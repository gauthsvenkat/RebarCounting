# RebarCounting
A computer vision deep learning model to automatically count the number of rebars, based on https://arxiv.org/abs/1807.09856

## Installation
First clone the repository. Run
 ```
 git clone https://github.com/andohuman/RebarCounting.git
 ```
Since the repository also contains some large files (weights) you might additionally want to run 
 ```
 git lfs install
 git lfs pull
 ```
This will download the weights (for predition)

Then, if you have pip installed, run
```
pip install -r requirements.txt
```
This will install all the requirements for the code.

then run 
```
python3 train.py -r __misc/ -l 50 -ne 1
```
to see if the training script is working right. This should generate a bad prediction in _visualizing_dots/1/

This code requires cuda 10.0 and corresponding cudnn to be installed on the system.
Code was ran and tested on python 3.6.8, torch 1.0.1 and torchvision 0.2.2

## Usage
### For training 
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

-e, --epoch :- If resuming training, epoch to start from (make sure epoch.pth exists in model_data/model and model_data/opt)

-l, --layers :- is the resnet version to use (either 50, 101 or 152)

-v, --visdom :- set this to true if you have visdom installed and would like to monitor your progress

Example:-
```
python3 train.py -r Data/ -e model_data/model/570.pth -l 101 -ne 100 -se 10 -v True
```
### For predicting
run 
```
python3 predict.py -h 
usage: predict.py [-h] [-r ROOT] [-s SPLIT] [-sc SCALE] [-sl SAVE_LOCATION]
                  [-t THRESHOLD]
                  model_name

Count rebars in images

positional arguments:
  model_name            The full path of the trained model (.pth file)

optional arguments:
  -h, --help            show this help message and exit
  -r ROOT, --root ROOT  Root directory housing the data
  -s SPLIT, --split SPLIT
                        Either test or custom split
  -sc SCALE, --scale SCALE
                        Scale down image by this factor
  -sl SAVE_LOCATION, --save_location SAVE_LOCATION
                        Location to save the predicted images
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold to consider as positive, (predictions >
                        threshold) will be counted as 1
```
-r, --root :- root directory housing the data

-s, --split :- either test or a custom split (the directory should be structured as Data/test/ for test or Data/custom_name for a custom testing, which should contain all the images in jpg format to be tested)

-sc, --scale :- Use this argument to scale down the images. (sometimes larger images take a long time to predict with marginal improvements in accuracy)

-sl, --save_location :- Path to a directory to save the predictedi images

-t, --threshold :- consider predictions as positive if they are greater than threshold

Example:- 
```
python3 predict.py -r Data/ -s testing/ -sc 5 -sl predicted_images/ -t 0.95
```

## Acknowledgements

This project is done under the guidance of TATA innovation. 
The project is based on the paper https://arxiv.org/abs/1807.09856. If you would like to cite this work, please cite the original authors.

## Citations
```
@Article{laradji2018blobs,
    title={Where are the Blobs: Counting by Localization with Point Supervision},
    author={Laradji, Issam H and Rostamzadeh, Negar and Pinheiro, Pedro O and Vazquez, David and Schmidt, Mark},
    journal = {ECCV},
    year = {2018}
}
```


