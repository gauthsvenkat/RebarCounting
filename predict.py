from datagen_pytorch import DataGenerator
from models import Resnet
from torch.utils.data import DataLoader
import torch
import numpy as np
from utils_pytorch import visualize_dots
from skimage import morphology as morph
import argparse
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument('mn', '--model_name')
parser.add_argument('-r', '--root', default='Data/')
parser.add_argument('-s', '--split', default='test')
parser.add_argument('-sl', '--save_location', default='testing/')
parser.add_argument('-t', '--threshold', default=0.9, type=float)
args = parser.parse_args()


gen = DataGenerator(root=args.root, scale=5, split=args.split, augment=False)
testloader = DataLoader(gen)

model = Resnet(n_classes=2, layers=101).cuda()
model.load_state_dict(torch.load(args.model_name))

with tqdm(testloader) as t:
	for i, batch in enumerate(t):
		t.set_description('TESTING:')

		out = model(batch['images'].cuda())
		out = torch.nn.functional.softmax(out, 1).cpu().detach().numpy()[0][1]
		img = np.asarray(batch['OG_image'][0])

		if not os.path.exists(args.save_location):
			os.makedirs(args.save_location)

		visualize_dots(img, (out>args.threshold).astype(int), save=True, path=args.save_location+batch['image_path'],size=1)