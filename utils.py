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

def get_top_k_acc(output,target,k=5):
	''' Given the predicted labels and target label, it provides the top-k accuracy in prediction 
	Top-5 accuracy means the percentage of time the target label is present in the first 5 predictions
	arranged in descending order of softmax probability'''
	count = 0
	top_k = torch.topk(output,dim=-1,k=5)
	for element in range(len(target)):
		if target[element] in top_k[1][element]: 
			count+=1
	return float(count)/float(target.size(0))


def verify_image(tensor):
	''' Function to personally view a loaded image, as a verification process '''
	import cv2
	import PIL
	img = torchvision.transforms.ToPILImage()(tensor[0])
	img.show()

def save_cpt(model,cpt_folder,n,optimizer = None):
	''' Function to save the checkpoint 
	cpt_folder -> folder to save the checkpoint
	n -> epoch		'''
	state_dict = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':n}
	torch.save(state_dict,os.path.join(cpt_folder,str(n)+'.pt'))
	tqdm.write("Model Saved !!")


def class_balanced_sampler(dataset,annotation_fname):
	''' This function takes a dataset and name of json file containing the annotations.
	It outputs the weight for each image in the dataset to make in class balanced. '''
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

