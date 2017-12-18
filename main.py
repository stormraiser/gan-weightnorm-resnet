import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.autograd as autograd
from torch.autograd import Variable
import os
import os.path
from PIL import Image

from model import Discriminator, Generator

parser = argparse.ArgumentParser()

parser.add_argument('--dataset',                          required = True,
	help = 'cifar10 | lsun | folder')
parser.add_argument('--lsun_class',                       default = 'bedroom',
	help = 'class of lsun dataset to use')
parser.add_argument('--dataroot',                         required = True,
	help = 'path to dataset')
parser.add_argument('--batch_size',         type = int,   default = 32,
	help = 'input batch size')
parser.add_argument('--image_size',         type = int,   default = -1,
	help = 'image size')
parser.add_argument('--width',              type = int,   default = -1,
	help = 'image width')
parser.add_argument('--height',             type = int,   default = -1,
	help = 'image height')
parser.add_argument('--crop_size',          type = int,   default = -1,
	help = 'crop size before scaling')
parser.add_argument('--crop_width',         type = int,   default = -1,
	help = 'crop width before scaling')
parser.add_argument('--crop_height',        type = int,   default = -1,
	help = 'crop height before scaling')
parser.add_argument('--code_size',          type = int,   default = 128,
	help = 'dimension of latent space')
parser.add_argument('--dis_feature',        type = int,   required = True,    nargs = '+',
	help = 'number of features for discriminator')
parser.add_argument('--dis_block',          type = int,   required = True,    nargs = '+',
	help = 'number of residue blocks for discriminator')
parser.add_argument('--gen_feature',        type = int,   required = True,    nargs = '+',
	help = 'number of features for generator')
parser.add_argument('--gen_block',          type = int,   required = True,    nargs = '+',
	help = 'number of residue blocks for generator')
parser.add_argument('--save_path',                        default = None,
	help = 'path to save generated files')
parser.add_argument('--load_path',                        default = None,
	help = 'load to continue existing experiment')
parser.add_argument('--lr',                 type = float, default = 2e-5,
	help = 'learning rate')
parser.add_argument('--test_interval',      type = int,   default = 1000,
	help = 'how often to test reconstruction')
parser.add_argument('--test_lr',            type = float, default = 0.01,
	help = 'learning rate for reconstruction test')
parser.add_argument('--test_steps',         type = int,   default = 50,
	help = 'number of steps in running reconstruction test')
parser.add_argument('--sample_interval',    type = int,   default = 100,
	help = 'how often to save generated samples')
parser.add_argument('--sample_size',        type = int,   default = 10,
	help = 'size of visualization grid')
parser.add_argument('--sample_row',         type = int,   default = -1,
	help = 'height of visualization grid')
parser.add_argument('--sample_col',         type = int,   default = -1,
	help = 'width of visualization grid')
parser.add_argument('--save_interval',      type = int,   default = 20000,
	help = 'how often to save checkpoints')
parser.add_argument('--niter',              type = int,   default = 1000000,
	help = 'number of iterations to train')
parser.add_argument('--gpu',                type = int,   default = 0,
	help = 'ID of gpu to use')
parser.add_argument('--random_crop',                      default = False,    action = 'store_true',
	help = 'random crop')

opt = parser.parse_args()
print(opt)

load_location_map = lambda storage, location: storage.cuda(opt.gpu)

transform_list = []

if (opt.crop_height > 0) and (opt.crop_width > 0):
	if opt.random_crop:
		transform_list.append(transforms.RandomCrop((opt.crop_height, opt.crop_width)))
	else:
		transform_list.append(transforms.CenterCrop((opt.crop_height, opt.crop_width)))
elif opt.crop_size > 0:
	if opt.random_crop:
		transform_list.append(transforms.RandomCrop(opt.crop_size))
	else:
		transform_list.append(transforms.CenterCrop(opt.crop_size))

if (opt.height > 0) and (opt.width > 0):
	transform_list.append(transforms.Scale((opt.width, opt.height)))
elif opt.image_size > 0:
	transform_list.append(transforms.Scale(opt.image_size))
	if opt.random_crop:
		transform_list.append(transforms.RandomCrop(opt.image_size))
	else:
		transform_list.append(transforms.CenterCrop(opt.image_size))
	opt.height = opt.image_size
	opt.width = opt.image_size

transform_list.append(transforms.ToTensor())

if (opt.sample_row <= 0) or (opt.sample_col <= 0):
	opt.sample_row = opt.sample_size
	opt.sample_col = opt.sample_size

if opt.dataset == 'cifar10':
	dataset1 = datasets.CIFAR10(root = opt.dataroot, download = True,
		transform = transforms.Compose(transform_list))
	dataset2 = datasets.CIFAR10(root = opt.dataroot, train = False,
		transform = transforms.Compose(transform_list))
	def get_data(k):
		if k < len(dataset1):
			return dataset1[k][0]
		else:
			return dataset2[k - len(dataset1)][0]
else:
	if opt.dataset == 'folder':
		dataset = datasets.ImageFolder(root = opt.dataroot,
			transform = transforms.Compose(transform_list))
	elif opt.dataset == 'lsun':
		dataset = datasets.LSUN(db_path = opt.dataroot, classes = [opt.lsun_class + '_train'],
			transform = transforms.Compose(transform_list))
	def get_data(k):
		return dataset[k][0]

data_index = torch.load(os.path.join(opt.dataroot, 'data_index.pt'))
train_index = data_index['train']
test_index = data_index['test']

gen = Generator(opt.width, opt.height, opt.gen_feature, opt.gen_block, opt.code_size)
print(gen)
dis = Discriminator(opt.width, opt.height, opt.dis_feature, opt.dis_block)
print(dis)
dis.cuda(opt.gpu)
gen.cuda(opt.gpu)
gen_opt = optim.RMSprop(gen.parameters(), lr = opt.lr, eps = 1e-8, alpha = 0.9)
dis_opt = optim.RMSprop(dis.parameters(), lr = opt.lr, eps = 1e-8, alpha = 0.9)
loss_func = nn.BCEWithLogitsLoss()
test_func = nn.MSELoss()

state = {}

def load_state(path, prefix):
	gen.load_state_dict(torch.load(os.path.join(path, 'net_archive', '{0}_gen.pt'.format(prefix)), map_location = load_location_map))
	gen_opt.load_state_dict(torch.load(os.path.join(path, 'net_archive', '{0}_gen_opt.pt'.format(prefix)), map_location = load_location_map))
	dis.load_state_dict(torch.load(os.path.join(path, 'net_archive', '{0}_dis.pt'.format(prefix)), map_location = load_location_map))
	dis_opt.load_state_dict(torch.load(os.path.join(path, 'net_archive', '{0}_dis_opt.pt'.format(prefix)), map_location = load_location_map))
	state.update(torch.load(os.path.join(path, 'net_archive', '{0}_state.pt'.format(prefix))))
	for solver_state in gen_opt.state.values():
		for k, v in solver_state.items():
			if torch.is_tensor(v):
				solver_state[k] = v.cuda(opt.gpu)
	for solver_state in dis_opt.state.values():
		for k, v in solver_state.items():
			if torch.is_tensor(v):
				solver_state[k] = v.cuda(opt.gpu)

def save_state(path, prefix):
	torch.save(gen.state_dict(), os.path.join(path, 'net_archive', '{0}_gen.pt'.format(prefix)))
	torch.save(gen_opt.state_dict(), os.path.join(path, 'net_archive', '{0}_gen_opt.pt'.format(prefix)))
	torch.save(dis.state_dict(), os.path.join(path, 'net_archive', '{0}_dis.pt'.format(prefix)))
	torch.save(dis_opt.state_dict(), os.path.join(path, 'net_archive', '{0}_dis_opt.pt'.format(prefix)))
	state.update({
		'index_shuffle' : index_shuffle,
		'current_iter' : current_iter,
		'best_iter' : best_iter,
		'min_loss' : min_loss,
		'current_sample' : current_sample
	})
	torch.save(state, os.path.join(opt.save_path, 'net_archive', '{0}_state.pt'.format(prefix)))

def generate_noise(m):
	return torch.randn(m, opt.code_size)

def visualize(code, filename):
	generated = torch.Tensor(code.size(0), 3, opt.height, opt.width)
	for i in range((code.size(0) - 1) // opt.batch_size + 1):
		batch_size = min(opt.batch_size, code.size(0) - i * opt.batch_size)
		batch_code = Variable(code[i * opt.batch_size : i * opt.batch_size + batch_size])
		generated[i * opt.batch_size : i * opt.batch_size + batch_size].copy_(gen(batch_code).data)
	torchvision.utils.save_image(generated, filename, opt.sample_col)

def test():
	test_loss = 0
	for param in gen.parameters():
		param.requires_grad = False
	best_code = torch.Tensor(test_index.size(0), opt.code_size).cuda(opt.gpu)
	total_batch = (test_index.size(0) - 1) // opt.batch_size + 1

	for i in range(total_batch):
		batch_size = min(opt.batch_size, test_index.size(0) - i * opt.batch_size)
		batch_code = Variable(torch.zeros(batch_size, opt.code_size).cuda(opt.gpu))
		batch_code.requires_grad = True

		batch_target = torch.Tensor(batch_size, 3, opt.height, opt.width)
		for j in range(batch_size):
			batch_target[j].copy_(get_data(test_index[i * opt.batch_size + j]))
		batch_target = Variable(batch_target.cuda(opt.gpu))

		test_opt = optim.RMSprop([batch_code], lr = opt.test_lr, eps = 1e-6, alpha = 0.9)
		for j in range(opt.test_steps):
			loss = test_func(gen(batch_code), batch_target)
			loss.backward()
			test_opt.step()
			batch_code.grad.data.zero_()
		best_code[i * opt.batch_size : i * opt.batch_size + batch_size].copy_(batch_code.data)
		
		generated = gen(batch_code)
		loss = test_func(gen(batch_code), batch_target)
		test_loss = test_loss + loss.data[0] * batch_size

	visualize(best_code[0 : min(test_index.size(0), opt.sample_row * opt.sample_col)], os.path.join(opt.save_path, 'test', 'test_{0}.jpg'.format(current_iter)))

	for param in gen.parameters():
		param.requires_grad = True
	test_loss = test_loss / test_index.size(0)
	print('loss = {0}'.format(test_loss))
	return test_loss

if opt.load_path is not None:
	if opt.save_path is None:
		opt.save_path = opt.load_path
	vis_code = torch.load(os.path.join(opt.load_path, 'samples', 'vis_code.pt')).cuda(opt.gpu)

	load_state(opt.load_path, 'last')
	index_shuffle = state['index_shuffle']
	current_iter = state['current_iter']
	best_iter = state['best_iter']
	min_loss = state['min_loss']
	current_sample = state['current_sample']
else:
	if not os.path.exists(opt.save_path):
		os.makedirs(opt.save_path)
	for sub_folder in ('samples', 'test', 'net_archive', 'log'):
		if not os.path.exists(os.path.join(opt.save_path, sub_folder)):
			os.mkdir(os.path.join(opt.save_path, sub_folder))
	vis_code = generate_noise(opt.sample_row * opt.sample_col).cuda(opt.gpu)
	torch.save(vis_code, os.path.join(opt.save_path, 'samples', 'vis_code.pt'))

	index_shuffle = torch.randperm(train_index.size(0))
	current_iter = 0
	best_iter = 0
	min_loss = 1e100
	current_sample = 0

	if test_index is not None:
		vis_target = torch.Tensor(min(test_index.size(0), opt.sample_row * opt.sample_col), 3, opt.height, opt.width)
		for i in range(vis_target.size(0)):
			vis_target[i].copy_(get_data(test_index[i]))
		torchvision.utils.save_image(vis_target, os.path.join(opt.save_path, 'test', 'target.jpg'), opt.sample_col)

opt_file = open(os.path.join(opt.save_path, 'opt'), 'w')
print(opt, file = opt_file)
opt_file.close()

ones = Variable(torch.ones(opt.batch_size).cuda(opt.gpu))
zeros = Variable(torch.zeros(opt.batch_size).cuda(opt.gpu))

loss_record = torch.zeros(opt.test_interval, 3)

visualize(vis_code, os.path.join(opt.save_path, 'samples', 'sample_{0}.jpg'.format(current_iter)))

while current_iter < opt.niter:
	current_iter = current_iter + 1
	print('Iteration {0}:'.format(current_iter))
	current_loss_record = loss_record[(current_iter - 1) % opt.test_interval]

	gen.zero_grad()

	rand_code = Variable(generate_noise(opt.batch_size).cuda(opt.gpu))
	generated = gen(rand_code)
	generated_detach = generated.detach()
	generated_detach.requires_grad = True
	dis_fake_output = dis(generated_detach.mul(2).sub(1))

	gen_loss = loss_func(dis_fake_output, ones)
	current_loss_record[2] = gen_loss.data[0]
	generated.backward(autograd.grad([gen_loss], [generated_detach], retain_graph = True)[0])
	gen_opt.step()

	dis.zero_grad()

	dis_fake_loss = loss_func(dis_fake_output, zeros)
	current_loss_record[1] = dis_fake_loss.data[0]
	dis_fake_loss.backward()

	true_sample = torch.Tensor(opt.batch_size, 3, opt.height, opt.width)
	for i in range(opt.batch_size):
		true_sample[i].copy_(get_data(train_index[index_shuffle[current_sample]]))
		current_sample = current_sample + 1
		if current_sample == train_index.size(0):
			current_sample = 0
			index_shuffle = torch.randperm(train_index.size(0))
	true_sample = Variable(true_sample.cuda(opt.gpu))
	dis_real_loss = loss_func(dis(true_sample.mul(2).sub(1)), ones)
	current_loss_record[0] = dis_real_loss.data[0]
	dis_real_loss.backward()

	dis_opt.step()

	print('loss: dis-real:{0:.4f} dis-fake:{1:.4f} gen:{2:.4f}'.format(current_loss_record[0], current_loss_record[1], current_loss_record[2]))

	if current_iter % opt.sample_interval == 0:
		visualize(vis_code, os.path.join(opt.save_path, 'samples', 'sample_{0}.jpg'.format(current_iter)))

	if current_iter % opt.test_interval == 0:
		if test_index is not None:
			print('Testing ...')
			current_loss = test()
			if current_loss < min_loss:
				print('new best network!')
				min_loss = current_loss
				best_iter = current_iter
				save_state(opt.save_path, 'best')
		else:
			current_loss = 0
		log = {
			'training_loss' : loss_record,
			'test_loss' : current_loss
		}
		torch.save(log, os.path.join(opt.save_path, 'log', 'loss_{0}.pt'.format(current_iter)))
		save_state(opt.save_path, 'last')

	if current_iter % opt.save_interval == 0:
		save_state(opt.save_path, current_iter)