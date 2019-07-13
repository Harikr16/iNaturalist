import torch 
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data.dataset import Dataset


import numpy as np
import pdb
import json
from tqdm import tqdm
import os
import argparse
import models
from config import *
from PIL import Image
from tensorboardX import SummaryWriter
import csv


from utils import *
from config import *


class cutout(object):
	def __init__(self,mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
		self.mask_size = mask_size
		self.p = p
		self.cutout_inside = cutout_inside
		self.mask_color = (0,0,0)
		self.mask_size_half = mask_size // 2
		self.offset = 1 if mask_size % 2 == 0 else 0

	def __call__(self,image):
		# image = np.asarray(image).copy()
		image = torchvision.transforms.ToTensor()(image)

		if np.random.random() > self.p:
			image = torchvision.transforms.ToPILImage()(image)
			return image

		h, w = image.shape[1:]

		if self.cutout_inside:
			cxmin, cxmax = self.mask_size_half, w + self.offset - self.mask_size_half
			cymin, cymax = self.mask_size_half, h + self.offset - self.mask_size_half
		else:
			cxmin, cxmax = 0, w + self.offset
			cymin, cymax = 0, h + self.offset

		cx = np.random.randint(cxmin, cxmax)
		cy = np.random.randint(cymin, cymax)
		xmin = cx - self.mask_size_half
		ymin = cy - self.mask_size_half
		xmax = xmin + self.mask_size
		ymax = ymin + self.mask_size
		xmin = max(0, xmin)
		ymin = max(0, ymin)
		xmax = min(w, xmax)
		ymax = min(h, ymax)
		image[:,ymin:ymax, xmin:xmax] = 0 #self.mask_color
		image = torchvision.transforms.ToPILImage()(image)
		return image

def mixup_data(x, y, alpha=1.0, use_cuda=True):
	'''Returns mixed inputs, pairs of targets, and lambda'''
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1

	batch_size = x.size()[0]
	if use_cuda:
		index = torch.randperm(batch_size).cuda()
	else:
		index = torch.randperm(batch_size)

	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class SuperScheduler(object):
	def __init__(self, optimizer, cycle_size=100):
		self.cycle_size = cycle_size
		self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.triangle)		

	def triangle(self, epoch):
		return np.max([1 - 2*np.abs(float(epoch)/float(self.cycle_size) - 0.5) + 1e-2, 1e-2])

	def step(self, iter_):
		self.scheduler.step(iter_)

def validate(model,valid_loader,args):
	model.eval()
	with torch.no_grad():
		loader = tqdm(valid_loader, ncols=75)
		correct = np.array([])
		correct_5 = np.array([])
		for data,target in loader:

			if args.use_cuda ==1:
				data,target = data.cuda(),target.cuda()
			output = model(data)
			get_top_5_acc(output,target)*100

			correct = np.concatenate((correct,torch.eq(torch.argmax(output,dim=-1),target).cpu().numpy()))
			correct_5 = np.append(correct_5,get_top_5_acc(output,target)*100)

	top_1_acc = correct.mean()
	top_5_acc = correct_5.mean()
	tqdm.write("\nTop 1 Validation Accuracy : {0:2.3f}".format(top_1_acc*100))
	tqdm.write("Top 5 Validation Accuracy : {0:2.3f}".format(top_5_acc))
	
	return top_1_acc,top_5_acc

def test(model,test_loader,args):
	model.eval()
	with torch.no_grad():
		loader = tqdm(test_loader, ncols=75)
		correct = np.array([])
		correct_5 = np.array([])
		data_dict={}
		for iter_,(data,target,id_) in enumerate(loader):

			if args.use_cuda ==1:
				data,target = data.cuda(),target.cuda()
			# pdb.set_trace()
			output = model(data)
			get_top_5_acc(output,target)*100

			pred = torch.argmax(output,dim=-1)
			for p,prediction in zip(id_,pred):
				data_dict[int(p.item())] = int(prediction.item())

	tqdm.write("\n Testing Completed. Writing Result ..")

	filename =  os.path.join(args.save,'test_result.csv')
	with open(filename,'w') as f:
		writer = csv.writer(f)
		writer.writerow(['id','predicted'])
		for row in data_dict.iteritems():
			writer.writerow(row)

	print("Results written to : ",filename) 	

def train(model,optimizer,scheduler,criterion,dataloaders,args,cpt_folder,writer):

	nepochs = args.epochs
	train_loader = dataloaders[0]
	valid_loader =dataloaders[1]

	
	for n in range(args.start_epoch,nepochs):
		model.train()
		loader = tqdm(train_loader, ncols=75)
		loader.set_description("Epoch : {0}".format(n+1))
		correct = np.array([])
		for iter_,(data,target) in enumerate(loader):

			# verify_image(data)	
			n_iter = n*len(train_loader) + iter_
			if args.use_cuda ==1:
				data,target = data.cuda(),target.cuda()

			if args.mixup == "True":
				inputs, targets_a, targets_b, lam = mixup_data(data, target)
			else:
				inputs = data

			optimizer.zero_grad()
			output = model(inputs)
			if args.mixup == "True":
				loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
			else:
				loss = criterion(output,target)
			loss.backward()
			optimizer.step()
			if args.sup_conv == "True":
				scheduler.step(iter_)
			correct = torch.eq(torch.argmax(output,dim=-1),target).cpu().numpy()

			top_1_acc = float(correct.sum())/len(data)*100
			top_5_acc = get_top_5_acc(output,target)*100
			loader.set_postfix(Train_loss = loss.item(), Top_1_acc = top_1_acc, Top_5_acc= top_5_acc)
			writer.add_scalar('Accuracy/Top_1_training',top_1_acc,n_iter)
			writer.add_scalar('Accuracy/Top_5_training',top_5_acc,n_iter)
			writer.add_scalar('Learning_Rate/Per_batch',optimizer.param_groups[0]['lr'],n_iter)
		save_cpt(model = model,optimizer = optimizer,cpt_folder = cpt_folder,n = n)
		top_1_acc_val,top_5_acc_val = validate(model,valid_loader)
		writer.add_scalar('Accuracy/Top_5_validation',top_5_acc_val,n)
		writer.add_scalar('Accuracy/Top_1_validation',top_1_acc_val,n)

class Naturalist_Dset(torch.utils.data.Dataset):

	def __init__(self,root,annotation_fname,transform = None):
		super(Naturalist_Dset,self).__init__()

		self.transforms = transform
		with open(annotation_fname,'r') as f:
			data = json.load(f)

		self.image_name_id = {}
		self.image_list = []
		for elements in data['images']:			
			self.image_name_id[os.path.join(os.path.dirname(root),elements['file_name'])] = elements['id']
			if os.path.exists(os.path.join(os.path.dirname(root),elements['file_name'])):
				self.image_list.append(os.path.join(os.path.dirname(root),elements['file_name']))

		self.image_id_anno = {}		
		for elements in data['annotations']:
			self.image_id_anno[elements['id']] = elements['category_id']

	def __getitem__(self,index):

		# path,target = self.imgs[index]
		path = self.image_list[index]

		try :
			target = self.image_id_anno[self.image_name_id[path]]
		except:
			index = index - 1 if index > 0 else index + 1 
			return self.__getitem__(index)

		# img = self.loader(path)
		img = Image.open(path)

		if self.transforms is not None:
			img = self.transforms(img)

		return (img,target)


	def __len__(self): 

		return len(self.image_list)

class Test_Dset(torchvision.datasets.ImageFolder):
	def __init__(self,root,annotation_fname,transform = None):
		super(Test_Dset,self).__init__(root,transform=transform)
		with open(annotation_fname,'r') as f:
			data = json.load(f)
		self.image_name_id = {}
		self.image_list = []
		for elements in data['images']:
			image_name = os.path.join(os.path.dirname(root),'test2019_',elements['file_name'])			
			self.image_name_id[image_name] = elements['id']
			if os.path.exists(image_name):
				self.image_list.append(image_name)

		self.image_id_anno = {}	
	def __getitem__(self,index):

		path,target = self.samples[index]
		sample = self.loader(path)
		if self.transform is not None:
			sample = self.transform(sample)
		id_ = self.image_name_id[path]
		return (sample,target,id_)
	def __len__(self):
		return len(self.samples)

def main(args):

	# Set seed
	torch.manual_seed(args.seed)

	# Create folder for saving checkpoints
	cpt_log = args.checkpoint
	cpt_folder = args.save +"/Supervised_checkpoints/"

	if not os.path.exists(cpt_folder):
		os.makedirs(cpt_folder)

	
	# Define Network and Optimizer	
	model = models.__dict__[args.arch](out = args.num_classes, sobel=args.sobel)
	optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-04, weight_decay=10**args.wd, amsgrad=False)

	# If cuda is available, parallelise and  use cuda
	if args.use_cuda == 1:
		model.features = torch.nn.DataParallel(model.features)
		model.cuda()
		cudnn.benchmark = True
		# Loss function
		criterion = nn.CrossEntropyLoss().cuda()
	else:
		criterion = nn.CrossEntropyLoss()

	# If loading from checkpoints
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			# Correct starting epoch number and load netwrok,optimizer
			args.start_epoch = checkpoint['epoch']+1
			model.load_state_dict(checkpoint['model'])
			optimizer.load_state_dict(checkpoint['optimizer'])
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))
	
	if args.mixup == "True":
		print("Using Mixup")

	
	# Create a Tensorboard object
	writer = SummaryWriter(os.path.join(args.save,'Supervised_Learning_Logs'))


	if args.mode == "TRAIN":	

		# Define Transformations
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		transformations = [transforms.Resize((512,512)),transforms.RandomHorizontalFlip(p=0.5),
				  cutout(mask_size=100,p=0.6,cutout_inside=True),
				  transforms.RandomAffine(degrees = 60,scale=(0.5,1.5),shear=(20)), 		  
				  transforms.ToTensor(),
				  normalize]

		# Annotations and other details of images are given in json files
		train_json = TRAIN_JSON
		valid_json = VALID_JSON
		print("Creating Dataset")
		train_dataset = Naturalist_Dset(args.data + '/train_val2019',annotation_fname = train_json,transform=transforms.Compose(transformations))
		valid_dataset = Naturalist_Dset(args.data + '/train_val2019',annotation_fname = valid_json,transform=transforms.Compose(transformations))
		
		# Create Class balanced sampler
		weights = class_balanced_sampler(train_dataset,train_json)
		sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,len(weights))

		# Create Dataloaders
		train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler = sampler,
												 batch_size=args.batch,
												 num_workers=args.workers,
												 pin_memory=True)
		valid_dataloader = torch.utils.data.DataLoader(valid_dataset, shuffle=False,
												 batch_size=args.batch*5,
												 num_workers=args.workers,
												 pin_memory=True)

		if args.sup_conv == "True":
			print("Using Super Convergence")
			scheduler = SuperScheduler(optimizer, len(train_dataloader))
		else:
			scheduler = None

		dataloaders = [train_dataloader,valid_dataloader]
		print("Initial Learning Rate: ",optimizer.param_groups[0]['params']['lr'] )

		#Train the model
		train(model,optimizer,scheduler,criterion,dataloaders,args,cpt_folder,writer)

	elif args.mode == "TEST":

		# Define Transformations
		transformations= [transforms.Resize((512,512)),transforms.ToTensor(),normalize]
		# Json files containing annotations
		test_root = TEST_ROOT
		test_json = TEST_JSON
		# Create test dataset and dataloader
		test_dataset = Test_Dset(root = test_root,annotation_fname = test_json,transform = transforms.Compose(transformations))
		test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False,
												 batch_size=args.batch*10,    # Testing requires very less memory
												 num_workers=args.workers,
												 pin_memory=True)
		# Run the test
		test(model,test_loader,args)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster SUpervised Learning')

	parser.add_argument('--data', metavar='DIR', help='path to dataset')
	parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
						choices=['alexnet', 'vgg16','resnet','densenet'], default='resnet',
						help='CNN architecture (default: alexnet)')
	parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
	parser.add_argument('--lr', default=0.05, type=float,
						help='learning rate (default: 0.05)')
	parser.add_argument('--wd', default=-5, type=float,
						help='weight decay pow (default: -5)')
	parser.add_argument('--workers', default=4, type=int,
						help='number of data loading workers (default: 4)')
	parser.add_argument('--epochs', type=int, default=200,
						help='number of total epochs to run (default: 200)')
	parser.add_argument('--start_epoch', default=0, type=int,
						help='manual epoch number (useful on restarts) (default: 0)')
	parser.add_argument('--batch', default=256, type=int,
						help='mini-batch size (default: 256)')
	parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
	parser.add_argument('--resume', default='', type=str, metavar='PATH',
						help='path to checkpoint (default: None)')
	parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
	parser.add_argument('--use_cuda', type = int, default = 1, help ='1 for using cuda, 0 for cpu')
	parser.add_argument('--save', type=str, default='', help='save directory')
	parser.add_argument('--num_classes', type=int, default=1000, help='No: of classes (default: 1000)')
	parser.add_argument('--split', type=float, default=.25, help='Ratio of valid dataset to train dataset')
	parser.add_argument('--mode',type=str,default="TRAIN",help='Running mode - Test/Train')
	parser.add_argument('--sup_conv',type=str,default="FALSE",help='Use super convergence')
	parser.add_argument('--mixup',type=str,default="FALSE",help='Use MIXUP')

	args = parser.parse_args()

	print(args)

	main(args)
