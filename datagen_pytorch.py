from torch.utils.data import Dataset
from glob import glob
import os, ntpath
import cv2
import numpy as np
import utils as ut
from utils import augmenter
import xml.etree.ElementTree as et
from torchvision.transforms import Compose, ToTensor, Normalize
import torch
import random


class DataGenerator(Dataset):

	def __init__(self, root="Data/", augment=True, debug_mode=False, scale=1, split=None):
		
		self.n_classes = 2 #Number of classes, either foreground or background
		self.augment = augment #Boolean value to decide if augmentation should be performed or not
		self.debug_mode = debug_mode
		self.scale = [scale] if isinstance(scale, int) else scale #Convert scale to list if integer
		self.split = split #train or test or custom
		#Convert image (HxWxC) to tensor of shape (CxHxW) and standardize
		self.transform = Compose([ToTensor(), Normalize(*ut.mean_std)])

		if self.split is 'train':
			self.anno_names = [x for x in glob(root+'annotations/*.xml')] #Complete locations of all the annotations files (XML)
			self.images_path = root+self.split+'/{}.JPG'
			self.length = len(self.anno_names)
			assert os.path.exists(self.anno_names[0]) #make sure annotations exists at that location.

		else:
			self.image_names = [x for x in glob(root+self.split+'/*.jpg')] #Complete location of all images in test or custom split
			self.length = len(self.image_names)
			assert os.path.exists(self.image_names[0]) #make sure the images exists


	def __len__(self):
		return self.length #Return number of images/annotations


	def __getitem__(self, index):

		if self.split is 'train': #train data
			return self.load_img_and_anno(index)
		
		else: #test or custom data
			return self.load_img(index)

	def load_img_and_anno(self, index):

		anno_name = self.anno_names[index] #get the file name
		scale = random.choice(self.scale) #choose the scale from list

		tree = et.parse(anno_name) #parse tree and
		root = tree.getroot() #get the root node
		
		width = int(root[4][0].text)//scale #initialize width and height
		height = int(root[4][1].text)//scale
		
		dots = np.zeros((height, width)) #Initialize dots matrix with zero (we'll populate this soon)
		
		for member in root.findall('object'): #Iterate over each bounding box

			xmin = int(member[4][0].text) #Get bounding box values
			ymin = int(member[4][1].text)
			xmax = int(member[4][2].text)
			ymax = int(member[4][3].text)

			x_cent = (xmin+xmax)//2//scale #Find the x,y coordinates of the centre according to scale
			y_cent = (ymin+ymax)//2//scale

			dots[y_cent][x_cent] = 1 #Set this location as 1 in the dots matrix (Corresponds to the point annotation in the image)

		
		img = cv2.imread(self.images_path.format(ntpath.basename(anno_name)[:-4])) #Load corresponding image file

		if scale is not 1: #scale down if self.scale != None
			img = cv2.resize(img, (0,0), fx=1/scale, fy=1/scale)			
		if self.augment: #augment image and annotations
			img, dots = augmenter(img, dots[...,np.newaxis], debug=self.debug_mode) #augment the image and/or display

		#return a dictionary
		return {
		"images":self.transform(np.ascontiguousarray(img)), #transformed image
		"points":torch.LongTensor(dots), #corresponding points converted to tensor
		"counts":torch.LongTensor(np.array([int(dots.sum())])), #the number of objects(rebars) in the picture
		"OG_image":np.ascontiguousarray(img) #the original image (useful for visualization purposes later)
		}

	def load_img(self, index):
		scale = random.choice(self.scale) #choose the scale

		img = cv2.imread(self.image_names[index]) #load image
		if scale is not 1: #resize
			img = cv2.resize(img, (0,0), fx=1/scale, fy=1/scale)

		return {
		"images":self.transform(np.ascontiguousarray(img)), #transformed image
		"OG_image":np.ascontiguousarray(img), #the original image (useful for visualization purposes later)
		"image_path":self.image_names[index] #the full path of the image (useful for making reports)
		}

