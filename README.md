# Video-Motion-Customization
This repository is the official implementation of <strong>VMC</strong>, a novel framework for Video Motion Customization.
<br>

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://video-motion-customization.github.io/)


## News
* [2023.11.30] Initial Code Release  
  (Revisions may occur, so stay tuned!)

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
Other scripts will be uploaded.

## Data
* PNG files: [Link to Google Drive Folder](https://drive.google.com/drive/u/2/folders/1L4dIqeK52lGBuxIKAEUzZgOEP95dz7AC)
* GIF files: [Link to Google Drive Folder](https://drive.google.com/drive/u/2/folders/1GUDnosOkYQ50-1bHHIBitRMeamkd2qao)


## Shoutouts
VMC builds upon a great open-source project on Video Diffusion Models: [Show-1](https://showlab.github.io/Show-1/) 
<br>Thank you for open-sourcing!


