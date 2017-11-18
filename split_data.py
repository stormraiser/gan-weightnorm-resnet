import argparse
import torch
import torchvision.datasets as datasets
import os
import os.path

parser = argparse.ArgumentParser()

parser.add_argument('--dataset',                           required = True,
	help = 'cifar10 | lsun | folder')
parser.add_argument('--lsun_class',                        default = 'bedroom',
	help = 'class of lsun dataset to use')
parser.add_argument('--dataroot',                          required = True,
	help = 'path to dataset')
parser.add_argument('--test_num',            type = int,   default = 0,
	help = 'number of samples to use in running test')

opt = parser.parse_args()

if opt.dataset == 'cifar10':
	dataset1 = datasets.CIFAR10(root = opt.dataroot, download = True)
	dataset2 = datasets.CIFAR10(root = opt.dataroot, train = False)
	fullsize = len(dataset1) + len(dataset2)
else:
	if opt.dataset == 'folder':
		dataset = datasets.ImageFolder(root = opt.dataroot)
	elif opt.dataset == 'lsun':
		dataset = datasets.LSUN(db_path = opt.dataroot, classes = [opt.lsun_class + '_train'])
	fullsize = len(dataset)

index_shuffle = torch.randperm(fullsize)

data_index = {}
if opt.test_num > 0:
	data_index['test'] = index_shuffle[:opt.test_num].clone()
else:
	data_index['test'] = None
data_index['train'] = index_shuffle[opt.test_num:].clone()
torch.save(data_index, os.path.join(opt.dataroot, 'data_index.pt'))
