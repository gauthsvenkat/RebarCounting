import os
import torch
from LCFCNloss import lc_loss
import numpy as np
from torchvision import transforms
from datagen_pytorch import DataGenerator
from torch.utils.data import DataLoader
from models import Resnet
from torch.optim import Adam
from utils import _tqdm as tqdm, GraphVisualization, visualize_dots
import argparse


parser = argparse.ArgumentParser(description='Train the counting model')
parser.add_argument('-r', '--root', default='Data/', help='Root directory housing the data')
parser.add_argument('-e', '--epoch', default=0, type=int, help='Epoch to start from (make sure epoch.pth exists in model_data/model and model_data/opt)')
parser.add_argument('-l', '--layers', default=101, help='resnet version to use')
parser.add_argument('-ne', '--num_epoch', default=100, type=int, help='number of epochs to train the model')
parser.add_argument('-se', '--save_every', default=10, type=int, help='save every - iterations')

args = parser.parse_args()

SCALE = [4,5,8,10] #Mutiple scale to train the model on (you can use more scale if your pc is strong enough)
EPOCHS = args.num_epoch

save_location = 'model_data/' #model save location
vis = GraphVisualization() #initialize the visdom object

train_set = DataGenerator(root=args.root, scale=SCALE, split='train')
trainloader = DataLoader(train_set, shuffle=True)

test_set = DataGenerator(root=args.root,scale=5, split='test', augment=False)
testloader = DataLoader(test_set)

model = Resnet(n_classes=2, layers=args.layers).cuda() #load model into gpu
opt = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-3) #set optimizer

if args.epoch > 0: #if initial epoch given load corresponding model and opt state
	print('Loading model and opt state_dict....')
	model.load_state_dict(torch.load(save_location+'model/'+str(args.epoch)+'.pth'))
	opt.load_state_dict(torch.load(save_location+'opt/'+str(args.epoch)+'.pth'))

r = range(args.epoch, EPOCHS) if args.epoch > 0 else range(EPOCHS) #Either start from beginning or continue

for epoch in r:
	model.train() #set model to training mode
	loss_sum = 0.

	with tqdm(trainloader) as t:
		for i, batch in enumerate(t):
			t.set_description('EPOCH: %i'%(epoch+1))

			opt.zero_grad() #zero the gradients
			loss = lc_loss(model, batch) #calculate loss
			loss.backward() #calculate gradients
			opt.step() #perform backprop step

			loss_sum += loss.item() #add current loss to overall loss
			t.set_postfix(loss=loss_sum/(i+1)) #display the average loss in the progress bar

	vis.plot_loss(loss_sum/(i+1), epoch+1) #plot loss for every epoch

	if not (epoch+1)%args.save_every or (epoch+1) is EPOCHS: #save every args.save_every epochs

		save_path_model = save_location+'model/'
		if not os.path.exists(save_path_model): #make directories if they don't exists
			os.makedirs(save_path_model)
		torch.save(model.state_dict(), save_path_model+str(epoch+1)+'.pth') #save the state_dict

		save_path_opt = save_location+'opt/' #DITTO
		if not os.path.exists(save_path_opt):
			os.makedirs(save_path_opt)
		torch.save(opt.state_dict(), save_path_opt+str(epoch+1)+'.pth')

		model.eval() #set model to evaluation mode (for visualizing)
		with tqdm(testloader) as t:
			for i, batch in enumerate(t):
				t.set_description('TESTING:')

				out = model(batch['images'].cuda()) #get output from model
				#softmax output and load in cpu and convert to numpy array
				out = torch.nn.functional.softmax(out, 1).cpu().detach().numpy()[0][1]
				img = np.asarray(batch['OG_image'][0]) #get original image

				img_save_path = '__visualizing_dots/'+str(epoch+1)+'/' #save location

				if not os.path.exists(img_save_path): #make directories if they don't exists
					os.makedirs(img_save_path)

				#save the pictures at location
				visualize_dots(img, (out > 0.5).astype(int), save=True, path=img_save_path+str(i)+'.jpg', size=1)

	print('\n')











