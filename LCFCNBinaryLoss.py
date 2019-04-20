import torch
import torch.nn.functional as F
import numpy as np 
from skimage.morphology import label
from utils_pytorch import t2n, watersplit

def lcfcn_binary_loss(output, points, thres=0.5):
	#model.train()

	#import ipdb; ipdb.set_trace()

	predictions = torch.sigmoid(output) #Sigmoid activate the outputs
	points = points.cuda() #Load into GPU
	
	points_s = torch.squeeze(points) #Remove batch and channel dimension
	predictions_s = torch.squeeze(predictions)

	points_np = np.squeeze(t2n(points)) #Convert to numpy array and remove batch and channel dimension
	predictions_np = np.squeeze(t2n(predictions))


	pred_mask = (predictions_np > thres).astype(int) #create the output mask with 1's and 0's
	blobs = label(pred_mask) #convert to blobs with unique IDs

	#blob_uniques will now contain points that are intersecting with points in the target annotations
	#blob_counts will now contain number of occurances of every unique point in blob_counts
	#NOTE:- blob_uniques WILL HAVE UNIQUE LABELLED POINTS ONLY (after morph.label). DO NOT THINK IT ONLY CONTAINS 0,1's LIKE A MORON!
	blob_uniques, blob_counts = np.unique(blobs * (points_np), return_counts=True)

	#uniques will now contain points that are NOT intersecting with points in the target annotations (False positives) 
	uniques = np.delete(np.unique(blobs), blob_uniques)

	IMAGE_LEVEL_LOSS = F.binary_cross_entropy(torch.max(predictions_s), torch.max(points_s))

	#POINT_LEVEL_LOSS = torch.mean(F.binary_cross_entropy(predictions_s, points_s))
	POINT_LEVEL_LOSS = torch.mean(-points_s * torch.log(predictions_s))

	#FALSE POSITIVE LOSS
	#This loss penalizes the model for predicting false positives

	false_positive_mask = np.zeros(predictions_np.shape)

	for u in uniques:
		if u==0:
			continue
		false_positive_mask+= blobs == u
	assert (false_positive_mask <= 1).all()

	fp_scale = np.log(max(1., false_positive_mask.sum()))
	false_positive_target = 1 - false_positive_mask #Set false positives to background

	FALSE_POSITIVE_LOSS = -fp_scale * torch.mean(torch.FloatTensor(1 - false_positive_target).cuda() * torch.log(1 - predictions_s)) #WATCHOUT

	#SPLIT LEVEL LOSS
	#This loss penalizes the model for predicting blobs with more than 1 point annotation to force the model to split predicted blobs

	T = np.zeros(predictions_np.shape)

	scale_multi = 0.

	for i in range(len(blob_uniques)):
		if blob_counts[i] < 2 or blob_uniques[i] == 0: #ignore if blobs count is 1 (blob has only one point)
				continue

		blob_ind = blobs == blob_uniques[i] #Working with a particular blob

		T += watersplit(predictions_np, points_np*blob_ind)*blob_ind #Find locations of boundaries inside that blob and add it to T (To get boundaries in the entire image)

		scale_multi += float(blob_counts[i]+1) #Add blob_counts for overall scale

	assert (T <= 1).all() #make sure no intersecting boundaries

	multi_blob_target = 1 - T #set boundaries to 0

	SPLIT_LEVEL_LOSS = -scale_multi * torch.mean(torch.FloatTensor(1 - multi_blob_target).cuda() * torch.log(1 - predictions_s))

	#GLOBAL LOSS
	#This loss forces the model to make make as many split as can be.

	T = 1 - watersplit(predictions_np, points_np) #Find boundaries and set them to 0(background)
	scale = 1. #float(points_np.sum())

	GLOBAL_SPLIT_LOSS = -scale * torch.mean(torch.FloatTensor(1 - T).cuda() * torch.log(1 - predictions_s))

	return IMAGE_LEVEL_LOSS + POINT_LEVEL_LOSS + FALSE_POSITIVE_LOSS + SPLIT_LEVEL_LOSS + GLOBAL_SPLIT_LOSS








