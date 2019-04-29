import torch.nn as nn
import torchvision
import torch
from skimage import morphology as morph
import numpy as np
import torch.utils.model_zoo as model_zoo

resnet_dict = {
	'50':torchvision.models.resnet50,
	'101':torchvision.models.resnet101,
	'152':torchvision.models.resnet152,
}

class BaseModel(nn.Module):
	def __init__(self, n_classes):
		super().__init__() #initialize parent class
		self.n_classes = n_classes #number of classes
	@torch.no_grad()
	def predict(self, batch, method="probs"):
		#THIS SECTION SPANNING FROM HERE
		self.eval()
		if method == "counts":
			images = batch["images"].cuda()
			pred_mask = self(images).data.max(1)[1].squeeze().cpu().numpy()

			counts = np.zeros(self.n_classes-1)

			for category_id in np.unique(pred_mask):
				if category_id == 0:
					continue
				blobs_category = morph.label(pred_mask==category_id)
				n_blobs = (np.unique(blobs_category) != 0).sum()
				counts[category_id-1] = n_blobs

			return counts[None]
		#TO HERE IS NEVER USED but I'm afraid to remove it anyway.
		elif method == "blobs": 

			images = batch["images"].cuda() #load image into gpu memory
			pred_mask = self(images).data.max(1)[1].squeeze().cpu().numpy() #get the predictions and convert to numpy array

			h,w = pred_mask.shape #initialize h,w and empty blobs variable
			blobs = np.zeros((self.n_classes-1, h, w), int)

			for category_id in np.unique(pred_mask):
				if category_id == 0:
					continue
				blobs[category_id-1] = morph.label(pred_mask==category_id) #get the unique labelled blobs
                
			return blobs[None]

class Resnet(BaseModel):
	def __init__(self, n_classes, layers):
		super().__init__(n_classes) #number of classes

		resnet = resnet_dict[str(layers)](pretrained=True) #get the resnet version

		resnet_block_expansion_rate = resnet.layer1[0].expansion

		resnet.fc = nn.Sequential()

		self.resnet = resnet

		self.score_32s = nn.Conv2d(512 *  resnet_block_expansion_rate, self.n_classes, kernel_size=1)

		self.score_16s = nn.Conv2d(256 *  resnet_block_expansion_rate, self.n_classes, kernel_size=1)

		self.score_8s = nn.Conv2d(128 *  resnet_block_expansion_rate, self.n_classes, kernel_size=1)

		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.weight.requires_grad = True
				m.bias.requires_grad = True

	def forward(self, x):

		self.resnet.eval()
		input_spatial_dim = x.size()[2:]

		x = self.resnet.conv1(x)
		x = self.resnet.bn1(x)
		x = self.resnet.relu(x)
		x = self.resnet.maxpool(x)

		x = self.resnet.layer1(x)

		x = self.resnet.layer2(x)
		logits_8s = self.score_8s(x)

		x = self.resnet.layer3(x)
		logits_16s = self.score_16s(x)

		x = self.resnet.layer4(x)
		logits_32s = self.score_32s(x)

		logits_16s_spatial_dim = logits_16s.size()[2:]
		logits_8s_spatial_dim = logits_8s.size()[2:]

		logits_16s += nn.functional.interpolate(logits_32s, size=logits_16s_spatial_dim, mode="bilinear", align_corners=True)

		logits_8s += nn.functional.interpolate(logits_16s,  size=logits_8s_spatial_dim, mode="bilinear", align_corners=True)

		logits_upsampled = nn.functional.interpolate(logits_8s, size=input_spatial_dim, mode="bilinear", align_corners=True)

		return logits_upsampled