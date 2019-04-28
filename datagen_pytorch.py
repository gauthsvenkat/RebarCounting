from torch.utils.data import Dataset
from glob import glob
import os, ntpath
import cv2
import numpy as np
import utils as utp
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
		self.scale = [scale] if isinstance(scale, int) else scale
		self.split = split
		self.transform = Compose([ToTensor(), Normalize(*utp.mean_std)])

		if self.split is 'train':
			self.anno_names = [x for x in glob(root+'annotations/*.xml')] #Complete locations of all the annotations files (XML)
			self.images_path = root+self.split+'/{}.JPG'
			self.length = len(self.anno_names)
			assert os.path.exists(self.anno_names[0]) #make sure annotations exists at that location.

		elif self.split is 'test':
			self.image_names = [x for x in glob(root+self.split+'/*.jpg')]
			self.length = len(self.image_names)
			assert os.path.exists(self.image_names[0])


	def __len__(self):
		return self.length


	def __getitem__(self, index):

		if self.split is 'train':
			return self.load_img_and_anno(index)
		
		elif self.split is 'test':
			return self.load_img(index)

	def load_img_and_anno(self, index):

		anno_name = self.anno_names[index]
		scale = random.choice(self.scale)

		tree = et.parse(anno_name)
		root = tree.getroot()
		
		width = int(root[4][0].text)//scale
		height = int(root[4][1].text)//scale
		
		dots = np.zeros((height, width)) #Initialize dots matrix with zero (we'll populate this soon)
		
		for member in root.findall('object'): #Iterate over each bounding box

			xmin = int(member[4][0].text) #Get bounding box values
			ymin = int(member[4][1].text)
			xmax = int(member[4][2].text)
			ymax = int(member[4][3].text)

			x_cent = (xmin+xmax)//2//scale #Find the x,y coordinates of the centre
			y_cent = (ymin+ymax)//2//scale

			dots[y_cent][x_cent] = 1 #Set this location as 1 in the dots matrix (Corresponds to the point annotation in the image)

		
		img = cv2.imread(self.images_path.format(ntpath.basename(anno_name)[:-4])) #Load corresponding image file
		#dots = dots[...,np.newaxis] #Change shape from (H,W) to (H,W,1)

		if scale is not 1: #scale down if self.scale != None
			img = cv2.resize(img, (0,0), fx=1/scale, fy=1/scale)			
		if self.augment:
			img, dots = augmenter(img, dots[...,np.newaxis], debug=self.debug_mode) #augment the image and/or display

		
		return {
		"images":self.transform(np.ascontiguousarray(img)),
		"points":torch.LongTensor(dots),
		"counts":torch.LongTensor(np.array([int(dots.sum())])),
		"OG_image":np.ascontiguousarray(img)
		}

	def load_img(self, index):
		scale = random.choice(self.scale)

		img = cv2.imread(self.image_names[index])
		if scale is not 1:
			img = cv2.resize(img, (0,0), fx=1/scale, fy=1/scale)

		return {
		"images":self.transform(np.ascontiguousarray(img)),
		"OG_image":np.ascontiguousarray(img),
		"image_path":self.image_names[index]
		}

