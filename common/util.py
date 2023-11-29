import os
import sys
import copy
import inspect
import datetime
from typing import List, Tuple, Optional, Dict
import torch
from tqdm import tqdm


def glob_files(
    root_path: str,
    extensions: Tuple[str],
    recursive: bool = True,
    skip_hidden_directories: bool = True,
    max_directories: Optional[int] = None,
    max_files: Optional[int] = None,
    relative_path: bool = False,
) -> Tuple[List[str], bool, bool]:
    """glob files with specified extensions

    Args:
        root_path (str): _description_
        extensions (Tuple[str]): _description_
        recursive (bool, optional): _description_. Defaults to True.
        skip_hidden_directories (bool, optional): _description_. Defaults to True.
        max_directories (Optional[int], optional): max number of directories to search. Defaults to None.
        max_files (Optional[int], optional): max file number limit. Defaults to None.
        relative_path (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[List[str], bool, bool]: _description_
    """
    paths = []
    hit_max_directories = False
    hit_max_files = False
    for directory_idx, (directory, _, fnames) in enumerate(os.walk(root_path, followlinks=True)):
        if skip_hidden_directories and os.path.basename(directory).startswith("."):
            continue

        if max_directories is not None and directory_idx >= max_directories:
            hit_max_directories = True
            break

        paths += [
            os.path.join(directory, fname)
            for fname in sorted(fnames)
            if fname.lower().endswith(extensions)
        ]

        if not recursive:
            break

        if max_files is not None and len(paths) > max_files:
            hit_max_files = True
            paths = paths[:max_files]
            break

    if relative_path:
        paths = [os.path.relpath(p, root_path) for p in paths]

    return paths, hit_max_directories, hit_max_files


def get_time_string() -> str:
    x = datetime.datetime.now()
    return f"{(x.year - 2000):02d}{x.month:02d}{x.day:02d}-{x.hour:02d}{x.minute:02d}{x.second:02d}"


def get_function_args() -> Dict:
    frame = sys._getframe(1)
    args, _, _, values = inspect.getargvalues(frame)
    args_dict = copy.deepcopy({arg: values[arg] for arg in args})

    return args_dict


# - - - - - DDIM Inversion - - - - #

def next_step(model_output, timestep, sample, ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    prompt_embeds, _ = pipeline.encode_prompt(prompt)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = pipeline.unet(
            latent, t, encoder_hidden_states=prompt_embeds,
        ).sample

        if pipeline.scheduler.config.variance_type not in ["learned", "learned_range"]:
                noise_pred, _ = noise_pred.split(latent.shape[1], dim=1)

        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents

