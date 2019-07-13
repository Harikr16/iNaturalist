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

# from importlib import reload
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
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

def class_balanced_sampler(dataset,annotation_fname):
	print("Creating Sampler ..")
	with open(annotation_fname,'r') as f:
		data = json.load(f)
	cat_dict = {}
	for ann in data['annotations']:
		try:
			cat_dict[ann['category_id']]+=1
		except:
			 cat_dict[ann['category_id']]=1  
	weights = []
	for i in range(len(dataset.image_list)):
		target = dataset.image_id_anno[dataset.image_name_id[dataset.image_list[i]]]
		weights.append(float(1)/float(cat_dict[target]))
	return torch.DoubleTensor(weights)

def get_subset_sampler(dataset,split_ratio):
	""" Get Train and valid samplers from a single dataset """
	# SPLIT RATIO is the size of validation dataset
	
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	split = int(np.floor(split_ratio * dataset_size))
	np.random.shuffle(indices)
	train_indices, val_indices = indices[split:], indices[:split]
	samplers = (SubsetRandomSampler(train_indices),SubsetRandomSampler(val_indices))
	return samplers

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

def validate(model,valid_loader):
	model.eval()
	with torch.no_grad():
		loader = tqdm(valid_loader, ncols=75)
		correct = np.array([])
		correct_5 = np.array([])
		for data,target in loader:

			data,target = data.cuda(),target.cuda()
			# pdb.set_trace()
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

			data,target = data.cuda(),target.cuda()
			# pdb.set_trace()
			output = model(data)
			get_top_5_acc(output,target)*100

			pred = torch.argmax(output,dim=-1)
			for p,prediction in zip(id_,pred):
				data_dict[int(p.item())] = int(prediction.item())

	tqdm.write("\n Testing Completed. Writing Result ..")

	filename =  os.path.join(args.exp,'test_result.csv')
	with open(filename,'w') as f:
		writer = csv.writer(f)
		writer.writerow(['id','predicted'])
		for row in data_dict.iteritems():
			writer.writerow(row)

	print("Results written to : ",filename) 	

def save_cpt(model,cpt_folder,n,optimizer = None):
	# try:
	state_dict = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':n}
	# state_dict = {'optimizer':optimizer.state_dict(),'epoch':n}
	# torch.save(model,os.path.join(cpt_folder,str(n)+'_model.pt'))
	# except:
	# state_dict = {'model':model.state_dict(),'epoch':n}
	torch.save(state_dict,os.path.join(cpt_folder,str(n)+'.pt'))

def get_top_5_acc(output,target):
	count = 0
	top_k = torch.topk(output,dim=-1,k=5)
	for element in range(len(target)):
		if target[element] in top_k[1][element]: 
			count+=1
	return float(count)/float(target.size(0))

def verify_image(tensor):
	import cv2
	import PIL
	img = torchvision.transforms.ToPILImage()(tensor[0])
	img.show()

def train(model,optimizer,scheduler,criterion,dataloaders,args,cpt_folder,writer):

	nepochs = args.epochs
	train_loader = dataloaders[0]
	valid_loader =dataloaders[1]

	
	for n in range(args.start_epoch,nepochs):
		# print("Start Epoch:", args.start_epoch)
		model.train()
		loader = tqdm(train_loader, ncols=75)
		loader.set_description("Epoch : {0}".format(n+1))
		correct = np.array([])
		for iter_,(data,target) in enumerate(loader):

			# verify_image(data)	
			n_iter = n*len(train_loader) + iter_
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

			# if iter_>2:
			# 	break

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

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)

	cpt_log = args.checkpoint
	cpt_folder = args.exp +"/Supervised_checkpoints/"
	if not os.path.exists(cpt_folder):
		os.makedirs(cpt_folder)

	# Data Loading
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	tra = [transforms.Resize((512,512)),transforms.RandomHorizontalFlip(p=0.5),
			  cutout(mask_size=100,p=0.6,cutout_inside=True),
			  transforms.RandomAffine(degrees = 60,scale=(0.5,1.5),shear=(20)), #translate=(0.3,0.3),			  
			  transforms.ToTensor(),
			  normalize]
			  # ]
	if args.fp16 == 1:
		tra = tra.append(transforms.Lambda(lambda x: x.half()))
	

	# train_json = "C:/Dataset/Naturalist/inaturalist-2019-fgvc6/train2019.json"
	# valid_json = "C:/Dataset/Naturalist/inaturalist-2019-fgvc6/val2019.json"
	if args.mode == "TRAIN":
		train_json = "/media/photogauge/7E30E7C830E7858D/Dataset/Naturalist/inaturalist-2019-fgvc6/train2019.json"
		valid_json = "/media/photogauge/7E30E7C830E7858D/Dataset/Naturalist/inaturalist-2019-fgvc6/val2019.json"
		print("Creating Dataset")
		train_dataset = Naturalist_Dset(args.data + '/train_val2019',annotation_fname = train_json,transform=transforms.Compose(tra))
		valid_dataset = Naturalist_Dset(args.data + '/train_val2019',annotation_fname = valid_json,transform=transforms.Compose(tra))
		
		weights = class_balanced_sampler(train_dataset,train_json)
		sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,len(weights))
		# sampler = get_subset_sampler(dataset = dataset,split_ratio = args.split)
		print("Creating Loader")
		train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler = sampler, #shuffle = True
												 batch_size=args.batch,
												 num_workers=args.workers,
												 pin_memory=True)
		valid_dataloader = torch.utils.data.DataLoader(valid_dataset,#sampler = sampler #, shuffle=True,
												 batch_size=args.batch*5,
												 num_workers=args.workers,
												 pin_memory=True)

	model = models.__dict__[args.arch](out = args.num_classes, sobel=args.sobel)
	print("Parallelising ..")
	model.features = torch.nn.DataParallel(model.features)
	model.cuda()
	cudnn.benchmark = True
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			# args.start_epoch = 47
			# model = torch.load(args.resume)
			##########################
			if args.contd==0:
				#remove top_layer parameters from checkpoint
				model.top_layer = None
				import copy
				new_cpt = copy.deepcopy(checkpoint)
				# print(checkpoint['state_dict'])
				# print("******************")
				for key in checkpoint['state_dict']:
					if 'top_layer' in key:
						del new_cpt['state_dict'][key]
				model.load_state_dict(new_cpt['state_dict'])
				if args.arch == "densenet":
					model.top_layer = torch.nn.Linear(1920, args.num_classes).cuda()
				elif args.arch == "resnet":
					model.top_layer = torch.nn.Linear(2048, args.num_classes).cuda()
			###########################
			##########################

			else:
				
				args.start_epoch = checkpoint['epoch']+1
				model.load_state_dict(checkpoint['model'])
				# optimizer.load_state_dict(chec)
				# pdb.set_trace()	
				# save_cpt(model,'Runs',47)
				# checkpoint = torch.load(args.resume)
				# model = checkpoint['model']
			###########################			
				# args.start_epoch = checkpoint['epoch']+1
				# model = checkpoint['model']
				#################################
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))
	# model.features = torch.nn.DataParallel(model.features)
	# pdb.set_trace()
	# model.cuda()
	model.float()	
	# pdb.set_trace() 
	
	if args.mixup == "True":
		print("Using Mixup")
	if args.sup_conv == "True":
		print("Using Super Convergence")
	if args.fp16 ==1:
		print(" USing FP16")
		model.half()  # convert to half precision
		for layer in model.modules():
			if isinstance(layer, nn.BatchNorm2d):
				layer.float()

	print("Creating Optimizer ..")
	# create optimizer
	optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-04, weight_decay=10**args.wd, amsgrad=False)
	# optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),lr=args.lr,momentum=args.momentum,weight_decay=10**args.wd,)
	if args.resume and args.contd:
		print("Loading Optimizer ..")
		#*******************Load OPTIMIZER **************************
		# optimizer.load_state_dict(checkpoint['optimizer'])

	criterion = nn.CrossEntropyLoss().cuda()



		# for param in optimizer.param_groups[0]['params']:
		# 	param.half()
		# 	for key, val in optimizer.state[param].items():
		# 		try:
		# 			optimizer.state[param][key] = val.half()
		# 		except:
		# 			pass



	writer = SummaryWriter(os.path.join(args.exp,'Supervised_Learning_Logs'))
	if args.mode == "TRAIN":	
		if args.sup_conv == "True":
			scheduler = SuperScheduler(optimizer, len(train_dataloader))
		else:
			scheduler = None
		dataloaders = [train_dataloader,valid_dataloader]
		# print("Initial Learning Rate: ",optimizer.param_groups[0]['params']['lr'] )
		train(model,optimizer,scheduler,criterion,dataloaders,args,cpt_folder,writer)
	elif args.mode == "TEST":
		tra = [transforms.Resize((512,512)),transforms.ToTensor(),normalize]
		test_root = "/media/photogauge/7E30E7C830E7858D/Dataset/Naturalist/inaturalist-2019-fgvc6/test2019_"
		test_json = "/media/photogauge/7E30E7C830E7858D/Dataset/Naturalist/inaturalist-2019-fgvc6/test2019.json"
		test_dataset = Test_Dset(root = test_root,annotation_fname = test_json,transform = transforms.Compose(tra))
		test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True,#sampler = sampler[1],
												 batch_size=args.batch*10,
												 num_workers=args.workers,
												 pin_memory=True)
		test(model,test_loader,args)

# python supervised_learner.py --dir="~/Datasets/Gears/" --resume="~/Codes/DC/Runs/ResNet152_pretrained_kmeans_k_20/Checkpoints/checkpoint_196_.pth.tar"  --epochs= 50 --exp= --sobel


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
	parser.add_argument('--checkpoint', type=int, default=25000,
						help='how many iterations between two checkpoints (default: 25000)')
	parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
	parser.add_argument('--exp', type=str, default='', help='path to exp folder')
	parser.add_argument('--verbose', action='store_true', help='chatty')
	parser.add_argument('--num_classes', type=int, default=1000, help='No: of classes (default: 1000)')
	parser.add_argument('--split', type=float, default=.25, help='Ratio of valid dataset to train dataset')
	parser.add_argument('--contd',type=int,default=1,help ='set to 0 if loading from deepclustering cpt')
	parser.add_argument('--fp16',type = int, default=0,help = 'set to 1 if fp16 to be used')
	parser.add_argument('--mode',type=str,default="TRAIN",help='Running mode - Test/Train')
	parser.add_argument('--sup_conv',type=str,default="FALSE",help='Use super convergence')
	parser.add_argument('--mixup',type=str,default="FALSE",help='Use MIXUP')
	args = parser.parse_args()
	print(args)

	# args.data = DIR
	# args.arch = ARCH
	# args.lr = LR
	# args.wd = WD
	# args.workers = WORKERS
	# args.exp = EXP
	# args.epochs = EPOCHS
	# args.start_epoch = START_EPOCH
	# args.batch = BATCH
	# args.momentum = MOMENTUM
	# args.resume =RESUME
	# args.checkpoint = CHECKPOINT
	# args.seed = SEED
	# args.num_classes = NUM_CLASSES
	# args.split = SPLIT
	# args.contd = CONTD 
	# args.fp16 = FP16
	# args.mode = MODE
	main(args)
