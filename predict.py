from datagen_pytorch import DataGenerator
from models import Resnet
from torch.utils.data import DataLoader
import torch
import numpy as np
from utils import visualize_dots
from skimage import morphology as morph
import argparse
from tqdm import tqdm
import os, ntpath

parser = argparse.ArgumentParser(description='Count rebars in images')
parser.add_argument('model_name', help='The full path of the trained model (.pth file)')
parser.add_argument('-r', '--root', default='Data/', help='Root directory housing the data')
parser.add_argument('-s', '--split', default='test', help='Either test or custom split')
parser.add_argument('-sc', '--scale', default=5, type=int, help='Scale down image by this factor')
parser.add_argument('-sl', '--save_location', default='testing/', help='Location to save the predicted images')
parser.add_argument('-t', '--threshold', default=0.9, type=float, help='Threshold to consider as positive, (predictions > threshold) will be counted as 1')
args = parser.parse_args()


gen = DataGenerator(root=args.root, scale=args.scale, split=args.split, augment=False)
testloader = DataLoader(gen)

model = Resnet(n_classes=2, layers=101).cuda() #load model into gpu
model.load_state_dict(torch.load(args.model_name)) #load model weights

with tqdm(testloader) as t:
	for i, batch in enumerate(t):
		t.set_description('TESTING:') #set description for progress bar

		out = model(batch['images'].cuda()) #get output
		#softmax the output from model and load into cpu and convert to numpy array and get proper predictions
		out = (torch.nn.functional.softmax(out, 1).cpu().detach().numpy()[0][1] > args.threshold).astype(int)
		img = np.asarray(batch['OG_image'][0]) #get the original image

		if not os.path.exists(args.save_location) #make sure directory exists else make one
			os.makedirs(args.save_location)
		img_name = os.path.splitext(ntpath.basename(batch['image_path'][0]))[0] #get image name without extension and full path

		#save the images and append the count to filename
		visualize_dots(img, out, save=True, path=args.save_location+img_name+'_c-'+str(morph.label(out, return_num=True)[1])+'.jpg',size=1)