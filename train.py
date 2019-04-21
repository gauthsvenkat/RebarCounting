import os
import torch
from og_loss import lc_loss
import numpy as np
from torchvision import transforms
from datagen_pytorch import DataGenerator
from torch.utils.data import DataLoader
from models import Resnet
from torch.optim import Adam
from utils_pytorch import _tqdm as tqdm, GraphVisualization
import argparse
from utils_pytorch import visualize_dots

parser = argparse.ArgumentParser()
parser.add_argument('-epoch', '--epoch', default=0)
parser.add_argument('-layers', '--layers', default=101)
parser.add_argument('-num_epoch', '--num_epoch', default=100)

training_log = open('training_log.csv', 'a+')

args = parser.parse_args()

SCALE = [4,5,8,10]
EPOCHS = int(args.num_epoch)

save_location = 'model_data/'
vis = GraphVisualization()

train_set = DataGenerator(root='Data/', scale=SCALE, split='train')
trainloader = DataLoader(train_set, shuffle=True)

test_set = DataGenerator(root='Data/',scale=5, split='test', augment=False)
testloader = DataLoader(test_set)

model = Resnet(n_classes=2, layers=args.layers).cuda()
opt = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-3)

if int(args.epoch) > 0:
	print('Loading model and opt state_dict....')
	model.load_state_dict(torch.load(save_location+'model/'+str(args.epoch)+'.pth'))
	opt.load_state_dict(torch.load(save_location+'opt/'+str(args.epoch)+'.pth'))



r = range(int(args.epoch), EPOCHS) if int(args.epoch) > 0 else range(EPOCHS)

for epoch in r:
	model.train()
	loss_sum = 0.

	with tqdm(trainloader) as t:
		for i, batch in enumerate(t):
			t.set_description('EPOCH: %i'%(epoch+1))

			opt.zero_grad()
			loss = lc_loss(model, batch)
			loss.backward()
			opt.step()

			loss_sum += loss.item()
			t.set_postfix(loss=loss_sum/(i+1))

	vis.plot_loss(loss_sum/(i+1), epoch+1)

	if not (epoch+1)%10 or (epoch+1) is EPOCHS:

		save_path_model = save_location+'model/'
		if not os.path.exists(save_path_model):
			os.makedirs(save_path_model)
		torch.save(model.state_dict(), save_path_model+str(epoch+1)+'.pth')

		save_path_opt = save_location+'opt/'
		if not os.path.exists(save_path_opt):
			os.makedirs(save_path_opt)
		torch.save(opt.state_dict(), save_path_opt+str(epoch+1)+'.pth')

		model.eval()
		with tqdm(testloader) as t:
			for i, batch in enumerate(t):
				t.set_description('TESTING:')

				out = model(batch['images'].cuda())
				out = torch.nn.functional.softmax(out, 1).cpu().detach().numpy()[0][1]
				img = np.asarray(batch['OG_image'][0])

				img_save_path = '__visualizing_dots/'+str(epoch+1)+'/'

				if not os.path.exists(img_save_path):
					os.makedirs(img_save_path)

				visualize_dots(img, (out > 0.5).astype(int), save=True, path=img_save_path+str(i)+'.jpg', size=1)

	print('\n')











