# RebarCounting
A computer vision deep learning model to automatically count the number of rebars, based on https://arxiv.org/abs/1807.09856

## Installation
If you have pip installed, run
```
pip install -r requirements.txt
```
This code requires cuda 10.0 and corresponding cudnn to be installed on the system.
Code was ran and tested on python 3.6.8, torch 1.0.1 and torchvision 0.2.2

## Usage
Run
```
python3 train.py -h
usage: train.py [-h] [-r ROOT] [-e EPOCH] [-l LAYERS] [-ne NUM_EPOCH]
                [-se SAVE_EVERY] [-v VISDOM]

Train the counting model

optional arguments:
  -h, --help            show this help message and exit
  -r ROOT, --root ROOT  Root directory housing the data
  -e EPOCH, --epoch EPOCH
                        Epoch to start from (make sure epoch.pth exists in
                        model_data/model and model_data/opt)
  -l LAYERS, --layers LAYERS
                        resnet version to use
  -ne NUM_EPOCH, --num_epoch NUM_EPOCH
                        number of epochs to train the model
  -se SAVE_EVERY, --save_every SAVE_EVERY
                        save every - iterations
  -v VISDOM, --visdom VISDOM
                        Host visdom server
```
-r, --root :- specifies the directory that houses the images and the annotations. (it is expected that the xml files are in root/annotations/ in PASCAL VOC format and the images in JPG format are in root/training/)

-e, --epoch :- Epoch to start from (make sure epoch.pth exists in model_data/model and model_data/opt)

-l, --layers :- is the resnet version to use (either 50, 101 or 152)

-v, --visdom :- set this to true if you have visdom installed and would like to monitor your progress




