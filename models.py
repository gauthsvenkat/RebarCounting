import torch.nn as nn
import torchvision
import torch
from skimage import morphology as morph
import numpy as np
import torch.utils.model_zoo as model_zoo

class BaseModel(nn.Module):
	def __init__(self, n_classes):
		super().__init__()
		self.trained_images = set()
		self.n_classes = n_classes
	@torch.no_grad()
	def predict(self, batch, method="probs"):
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

		elif method == "blobs": 

			images = batch["images"].cuda()
			pred_mask = self(images).data.max(1)[1].squeeze().cpu().numpy()

			h,w = pred_mask.shape
			blobs = np.zeros((self.n_classes-1, h, w), int)

			for category_id in np.unique(pred_mask):
				if category_id == 0:
					continue
				blobs[category_id-1] = morph.label(pred_mask==category_id)
                
			return blobs[None]

'''
class Resnet50(BaseModel):
	def __init__(self, n_classes):
		super().__init__(n_classes)

		resnet50_32s = torchvision.models.resnet50(pretrained=True)

		resnet_block_expansion_rate = resnet50_32s.layer1[0].expansion

		resnet50_32s.fc = nn.Sequential()

		self.resnet50_32s = resnet50_32s

		self.score_32s = nn.Conv2d(512 *  resnet_block_expansion_rate, self.n_classes, kernel_size=1)

		self.score_16s = nn.Conv2d(256 *  resnet_block_expansion_rate, self.n_classes, kernel_size=1)

		self.score_8s = nn.Conv2d(128 *  resnet_block_expansion_rate, self.n_classes, kernel_size=1)

		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.weight.requires_grad = False
				m.bias.requires_grad = False

	def forward(self, x):

		self.resnet50_32s.eval()
		input_spatial_dim = x.size()[2:]

		x = self.resnet50_32s.conv1(x)
		x = self.resnet50_32s.bn1(x)
		x = self.resnet50_32s.relu(x)
		x = self.resnet50_32s.maxpool(x)

		x = self.resnet50_32s.layer1(x)

		x = self.resnet50_32s.layer2(x)
		logits_8s = self.score_8s(x)

		x = self.resnet50_32s.layer3(x)
		logits_16s = self.score_16s(x)

		x = self.resnet50_32s.layer4(x)
		logits_32s = self.score_32s(x)

		logits_16s_spatial_dim = logits_16s.size()[2:]
		logits_8s_spatial_dim = logits_8s.size()[2:]

		logits_16s += nn.functional.interpolate(logits_32s, size=logits_16s_spatial_dim, mode="bilinear", align_corners=True)

		logits_8s += nn.functional.interpolate(logits_16s,  size=logits_8s_spatial_dim, mode="bilinear", align_corners=True)

		logits_upsampled = nn.functional.interpolate(logits_8s, size=input_spatial_dim, mode="bilinear", align_corners=True)

		return logits_upsampled

class Resnet152(BaseModel):
	def __init__(self, n_classes):
		super().__init__(n_classes)

		resnet152_32s = torchvision.models.resnet152(pretrained=True)

		resnet_block_expansion_rate = resnet152_32s.layer1[0].expansion

		resnet152_32s.fc = nn.Sequential()

		self.resnet152_32s = resnet152_32s

		self.score_32s = nn.Conv2d(512 *  resnet_block_expansion_rate, self.n_classes, kernel_size=1)

		self.score_16s = nn.Conv2d(256 *  resnet_block_expansion_rate, self.n_classes, kernel_size=1)

		self.score_8s = nn.Conv2d(128 *  resnet_block_expansion_rate, self.n_classes, kernel_size=1)

		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.weight.requires_grad = False
				m.bias.requires_grad = False

	def forward(self, x):

		self.resnet152_32s.eval()
		input_spatial_dim = x.size()[2:]

		x = self.resnet152_32s.conv1(x)
		x = self.resnet152_32s.bn1(x)
		x = self.resnet152_32s.relu(x)
		x = self.resnet152_32s.maxpool(x)

		x = self.resnet152_32s.layer1(x)

		x = self.resnet152_32s.layer2(x)
		logits_8s = self.score_8s(x)

		x = self.resnet152_32s.layer3(x)
		logits_16s = self.score_16s(x)

		x = self.resnet152_32s.layer4(x)
		logits_32s = self.score_32s(x)

		logits_16s_spatial_dim = logits_16s.size()[2:]
		logits_8s_spatial_dim = logits_8s.size()[2:]

		logits_16s += nn.functional.interpolate(logits_32s, size=logits_16s_spatial_dim, mode="bilinear", align_corners=True)

		logits_8s += nn.functional.interpolate(logits_16s,  size=logits_8s_spatial_dim, mode="bilinear", align_corners=True)

		logits_upsampled = nn.functional.interpolate(logits_8s, size=input_spatial_dim, mode="bilinear", align_corners=True)

		return logits_upsampled
'''

resnet_dict = {
	'50':torchvision.models.resnet50,
	'101':torchvision.models.resnet101,
	'152':torchvision.models.resnet152,
}

class Resnet(BaseModel):
	def __init__(self, n_classes, layers):
		super().__init__(n_classes)

		resnet = resnet_dict[str(layers)](pretrained=True)

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