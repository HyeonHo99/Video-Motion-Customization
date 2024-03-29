# Video-Motion-Customization (CVPR 2024)
This repository is the official implementation of [VMC](https://arxiv.org/abs/2312.00845).<br>
**[CVPR 2024] [VMC: Video Motion Customization using Temporal Attention Adaption for Text-to-Video Diffusion Models](https://arxiv.org/abs/2312.00845)**
<br>
[Hyeonho Jeong*](https://hyeonho99.github.io/),
[Geon Yeong Park*](https://geonyeong-park.github.io/),
[Jong Chul Ye](https://scholar.google.com/citations?user=HNMjoNEAAAAJ&hl=ko&oi=sra),

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://video-motion-customization.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2312.00845-b31b1b.svg)](https://arxiv.org/abs/2312.00845)

<p align="center">
<img src="https://video-motion-customization.github.io/static/images/figure-overview.png" width="100%"/>
<br>
<em>Given an input video with any type of motion patterns, our framework, VMC fine-tunes only the Keyframe Generation Module within hierarchical Video Diffusion Models for motion-customized video generation.</em>
</p>

## News
* [2023.11.30] Initial Code Release
  (Additional codes will be uploaded.)

## Setup
### Requirements

```shell
pip install -r requirements.txt
```

## Usage

The following command will run "train & inference" at the same time:

```bash
accelerate launch train_inference.py --config configs/car_forest.yml
```
*Additional scripts of 'train_only' and 'inference_with_pretrained' will be uploaded too.

## Data
* PNG files: [Google Drive Folder](https://drive.google.com/drive/u/2/folders/1L4dIqeK52lGBuxIKAEUzZgOEP95dz7AC)
* GIF files: [Google Drive Folder](https://drive.google.com/drive/u/2/folders/1GUDnosOkYQ50-1bHHIBitRMeamkd2qao)

## Results
<table class="center">
  <tr>
    <td style="text-align:center;"><b>Input Videos</b></td>
    <td style="text-align:center;" colspan="1"><b>Output Videos</b></td>
  </tr>
  <tr>
    <td><img src="https://video-motion-customization.github.io/static/gifs/sharks_swimming/input.gif"></td>
    <td><img src="https://video-motion-customization.github.io/static/gifs/sharks_swimming/spaceships_space.gif"></td>
  </tr>
  <tr>
    <td><img src="https://video-motion-customization.github.io/static/gifs3/car_forest/input.gif"></td>
    <td><img src="https://video-motion-customization.github.io/static/gifs3/car_forest/tank_snow.gif"></td>
  </tr>
  <tr>
    <td><img src="https://video-motion-customization.github.io/static/gifs4/plane_sky3/input.gif"></td>
    <td><img src="https://video-motion-customization.github.io/static/gifs4/plane_sky3/shark_under.gif"></td>
  </tr>
  <tr>
    <td><img src="https://video-motion-customization.github.io/static/gifs2/bird_forest/input.gif"></td>
    <td><img src="https://video-motion-customization.github.io/static/gifs2/bird_forest/phoenix_lava.gif"></td>
  </tr>
  <tr>
    <td><img src="https://video-motion-customization.github.io/static/gifs/skiing/input.gif"></td>
    <td><img src="https://video-motion-customization.github.io/static/gifs/skiing/astronaut_underwater.gif"></td>
  </tr>
  <tr>
    <td><img src="https://video-motion-customization.github.io/static/gifs/child_bike/input.gif"></td>
    <td><img src="https://video-motion-customization.github.io/static/gifs/child_bike/monkey.gif"></td>
  </tr>
  <tr>
    <td><img src="https://video-motion-customization.github.io/static/gifs4/pills_black/input.gif"></td>
    <td><img src="https://video-motion-customization.github.io/static/gifs4/pills_black/stars.gif"></td>
  </tr>
  <tr>
    <td><img src="https://video-motion-customization.github.io/static/gifs3/ink_spreading/input.gif"></td>
    <td><img src="https://video-motion-customization.github.io/static/gifs3/ink_spreading/flower.gif"></td>
  </tr>

</table>

### Video Style Transfer
<table class="center">
  <tr>
    <td><img src="https://video-motion-customization.github.io/static/gifs/child_bike/input.gif"></td>
    <td><img src="https://video-motion-customization.github.io/static/gifs/car_turn/input.gif"></td>
    <td><img src="https://video-motion-customization.github.io/static/gifs3/car_forest/input.gif"></td>
    <td><img src="https://video-motion-customization.github.io/static/gifs4/plane_sky2/input.gif"></td>
  </tr>
  <tr>
    <td><img src="https://video-motion-customization.github.io/static/style-transfer/child_bike/starry_van_gogh.gif"></td>
    <td><img src="https://video-motion-customization.github.io/static/style-transfer/car_turn/oil_flowers.gif"></td>
    <td><img src="https://video-motion-customization.github.io/static/gifs3/car_forest/anime.gif"></td>
    <td><img src="https://video-motion-customization.github.io/static/style-transfer/plane_sky2/starry_van_gogh.gif"></td>
  </tr>
</table>

### Backward Motion Customization
<table class="center">
  <tr>
    <td style="text-align:center;"><b>Reversed Videos</b></td>
    <td style="text-align:center;" colspan="1"><b>Output Videos</b></td>
  </tr>
  <tr>
    <td><img src="https://video-motion-customization.github.io/static/backward-motion/ink_backward/reverse.gif"></td>
    <td><img src="https://video-motion-customization.github.io/static/backward-motion/ink_backward/ink_water.gif"></td>
  </tr>
  <tr>
    <td><img src="https://video-motion-customization.github.io/static/backward-motion/car_backward/reverse.gif"></td>
    <td><img src="https://video-motion-customization.github.io/static/backward-motion/car_backward/tank_road.gif"></td>
  </tr>
  <tr>
    <td><img src="https://video-motion-customization.github.io/static/backward-motion/car_forest_backward/reverse.gif"></td>
    <td><img src="https://video-motion-customization.github.io/static/backward-motion/car_forest_backward/lamborghini_space.gif"></td>
  </tr>
  <tr>
    <td><img src="https://video-motion-customization.github.io/static/gifs2/seagull_skyline_backward/input.gif"></td>
    <td><img src="https://video-motion-customization.github.io/static/gifs2/seagull_skyline_backward/eagle_edge.gif"></td>
  </tr>
</table>


## Citation
If you find our work interesting, please cite our paper.
```bibtex
@article{jeong2023vmc,
  title={VMC: Video Motion Customization using Temporal Attention Adaption for Text-to-Video Diffusion Models},
  author={Jeong, Hyeonho and Park, Geon Yeong and Ye, Jong Chul},
  journal={arXiv preprint arXiv:2312.00845},
  year={2023}
}
```

## Shoutouts
- VMC directly employs an open-source project on cascaded Video Diffusion Models, [Show-1](https://github.com/showlab/Show-1),
  along with [DeepFloyd IF](https://github.com/deep-floyd/IF).
- This code builds upon [Diffusers](https://github.com/huggingface/diffusers) and we referenced the code logic of [Tune-A-Video](https://github.com/showlab/Tune-A-Video).
- We conducted evaluation against 4 great projects: [VideoComposer](https://arxiv.org/abs/2306.02018), [Gen-1](https://arxiv.org/abs/2302.03011), [Tune-A-Video](https://arxiv.org/abs/2212.11565), [Control-A-Video](https://arxiv.org/abs/2305.13840)

<br><i>Thanks all for open-sourcing!</i>


