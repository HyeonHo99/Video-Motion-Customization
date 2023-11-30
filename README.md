# Video-Motion-Customization
This repository is the official implementation of ["One-Shot Video Motion Customization using Temporal Attention Adaption for Text-to-Video Diffusion Models."](#)
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
VMC builds upon an open-source project on cascaded Video Diffusion Models, namely [Show-1](https://showlab.github.io/Show-1/) 
<br><i>Thank you for open-sourcing!</i>


