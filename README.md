# Video-Motion-Customization
This repository is the official implementation of ["VMC: One-Shot Video Motion Customization using Temporal Attention Adaption for Text-to-Video Diffusion Models."](#)
<br>

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://video-motion-customization.github.io/)

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

## Shoutouts
VMC directly employs an open-source project on cascaded Video Diffusion Models, [Show-1](https://github.com/showlab/Show-1), along with [DeepFloyd IF](https://github.com/deep-floyd/IF).  
Additionally, this code builds upon [Diffusers](https://github.com/huggingface/diffusers) and we referenced the code logic of [Tune-A-Video](https://github.com/showlab/Tune-A-Video)
<br><i>Thanks all for open-sourcing!</i>


