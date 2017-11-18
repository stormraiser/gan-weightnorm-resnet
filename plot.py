import argparse
import matplotlib.pyplot as plt
import torch
import numpy
import os
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('--paths', nargs='+', help='paths to experiments to plot')
parser.add_argument('--part', default='test', help='test | dis | dis-real | dis-fake | gen')
parser.add_argument('--window_size', type=int, default=100, help='window size for averaged discriminator loss')
opt = parser.parse_args()
print(opt)

for k, load_path in enumerate(opt.paths):
	logs = []
	log_files = os.listdir(os.path.join(load_path, 'log'))
	for log_file in log_files:
		niter = int(log_file[5:-3])
		log = torch.load(os.path.join(load_path, 'log', log_file))
		if opt.part == 'test':
			loss = log['test_loss']
			logs.append((niter, loss))
		else:
			losses = log['training_loss']
			for i in range(losses.size(0) // opt.window_size):
				avg_loss = losses[i * opt.window_size : (i + 1) * opt.window_size].mean(0)
				if opt.part == 'dis':
					loss = avg_loss[0] + avg_loss[1]
				elif opt.part == 'dis-real':
					loss = avg_loss[0]
				elif opt.part == 'dis-fake':
					loss = avg_loss[1]
				else:
					loss = avg_loss[2]
				logs.append((niter - losses.size(0) + (i + 1) * opt.window_size, loss))
	logs.sort()
	n = len(logs)
	x = torch.Tensor(n)
	y = torch.Tensor(n)
	for i in range(n):
		x[i] = logs[i][0]
		y[i] = logs[i][1]
	plot_name = '{0}-{1}'.format(k, os.path.basename(load_path))
	plt.plot(x.numpy(), y.numpy())
plt.show()