import numpy as np
import cv2
from imgaug import augmenters as iaa
import imgaug as ia
import random, string
from skimage import morphology as morph
import torch
import visdom
from tqdm import tqdm


class GraphVisualization:
	def __init__(self, env_name='main'):
		self.env_name = env_name
		self.vis = visdom.Visdom(env=self.env_name)
		self.loss_win = 'Training'

	def plot_loss(self, loss, epoch):
		self.loss_win = self.vis.line(
			X=[epoch],
			Y=[loss],
			win=self.loss_win,
			update='append',
			opts=dict(
				xlabel='Epoch',
				ylabel='loss',
				title='Loss per Epoch'),
			)

class _tqdm(tqdm):
	def format_num(self, n):
		f = '{:.3f}'.format(n)

		return f




seq = iaa.Sequential([
	iaa.Fliplr(.4), #Flip horizontal 40% of the time
	iaa.Flipud(.4), #Flip Vertical 40% of the time
	iaa.Sometimes(.65, #Do these 65% of the time
		iaa.GaussianBlur(sigma=2.0) #Blur the Image a bit
		),
	iaa.Add((-40,40)) #Add or subtract pixels (Increase or Decrease brightness)
	], random_order=True
	)

def augmenter(img, anno, seq=seq, debug=False):
	
	seq = seq.to_deterministic() #Call this every time to make sure the img and anno are getting augmented the same way
	anno_ = ia.SegmentationMapOnImage(anno, shape=anno.shape) #Create a SegmentationMapObject

	img_aug = seq.augment_image(img) 
	anno_aug = seq.augment_segmentation_maps([anno_])[0].get_arr_int() #Augment annotation and get the values as an int array

	if debug: #If debug==True, display augmentation
		visualize_dots(img_aug, anno_aug)

	assert int(anno.sum()) == int(anno_aug.sum()) #Make sure the augmentation didn't screw with the counts
	assert img.shape == img_aug.shape #Make sure the img and anno dimensions are same before and after augmentations
	assert np.squeeze(anno).shape == anno_aug.shape

	return img_aug, anno_aug


def visualize_dots(img, points, save=False, name=None, path=None, size=5):

	points = points.squeeze()
	y,x= np.where(points==1) #Get locations of 1s

	for x_cent, y_cent in zip(x,y):
		img = cv2.circle(img, (x_cent,y_cent), size, (0,0,255), -1) #Draw a filled circle at these locations 
	img = cv2.putText(img, str(morph.label(points, return_num=True)[1]), (0,0), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,255,255))

	if save and path:
		cv2.imwrite(path, img)

	elif save and not path: #Write the augmented images to a temporary folder (For Debug Purposes) if save is True
		if name is None:
			cv2.imwrite("__visualizing_dots/"+''.join(random.choices(string.ascii_uppercase + string.digits, k=6))+".jpg", img)
		else:
			cv2.imwrite("__visualizing_dots/"+str(name)+".jpg", img)
	else: #Scale down and display
		#img = cv2.resize(img, dsize=None, fx=0.25, fy=0.25) #Scale down to 25% of the original size
		cv2.imshow("Dots Visualized", img)
		cv2.waitKey(0) 
		cv2.destroyWindow("Dots Visualized")


def t2n(x):
	if isinstance(x, torch.Tensor):
		x = x.cpu().detach().numpy()

	return x


mean_std = ([0.30852305, 0.33011866, 0.36652088], [0.22718432, 0.23224937, 0.23994817]) #rgb means and stds