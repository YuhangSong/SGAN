import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
import cPickle as pickle
import dill

import torch
import visdom
vis = visdom.Visdom()

class logger(object):
	"""docstring for logger"""
	def __init__(self,LOGDIR,DSP):
		super(logger, self).__init__()

		self.LOGDIR = LOGDIR
		self.DSP = DSP

		self._iter = 0

		self._since_beginning = collections.defaultdict(lambda: {})
		self._since_last_flush = collections.defaultdict(lambda: {})

		self.restore()

	def restore(self):

		print('Try load previous plot....')
		try:
			self._since_beginning = dill.load(open(self.LOGDIR+'log.pkl', "r"))
			self._iter = dill.load(open(self.LOGDIR+'iteration.pkl', "r"))
			print('Restore plot from iter: '+str(self._iter))
		except Exception, e:
			print('Previous plot unfounded')
		print('')

	def tick(self):
		self._iter += 1

	def plot(self, name, value):
		value = np.asarray(value)
		self._since_last_flush[name][self._iter] = value

	def flush(self):

		for name, vals in self._since_last_flush.items():

			self._since_beginning[name].update(vals)

			x_vals = np.sort(self._since_beginning[name].keys())
			y_vals = [self._since_beginning[name][x] for x in x_vals]

			plt.clf()
			plt.plot(x_vals, y_vals)
			plt.xlabel('iteration')
			plt.ylabel(self.DSP+name)
			plt.savefig(self.LOGDIR+name+'.jpg')

			if len(x_vals) > 1:
				vis.line(   X=torch.from_numpy(np.asarray(x_vals)),
							Y=torch.from_numpy(np.asarray(y_vals)),
		                    win=self.DSP+name,
		                    opts=dict(title=self.DSP+name))

		self._since_last_flush.clear()

		with open(self.LOGDIR+'log.pkl', 'wb') as f:
			dill.dump(self._since_beginning, f)
		with open(self.LOGDIR+'iteration.pkl', 'wb') as f:
			dill.dump(self._iter, f)
