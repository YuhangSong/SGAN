import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
import cPickle as pickle

import torch
import visdom
vis = visdom.Visdom()

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]
def tick():
	_iter[0] += 1

def plot(name, value):
	_since_last_flush[name][_iter[0]] = value

def flush(LOGDIR,DSP):
	prints = []

	for name, vals in _since_last_flush.items():
		prints.append("{}\t{}".format(name, np.mean(vals.values())))
		_since_beginning[name].update(vals)

		x_vals = np.sort(_since_beginning[name].keys())
		y_vals = [_since_beginning[name][x] for x in x_vals]

		plt.clf()
		plt.plot(x_vals, y_vals)
		plt.xlabel('iteration')
		plt.ylabel(DSP+name)
		plt.savefig(LOGDIR+name+'.jpg')

		if len(x_vals) > 1:
			vis.line(   X=torch.from_numpy(np.asarray(x_vals)),
						Y=torch.from_numpy(np.asarray(y_vals)),
	                    win=DSP+name,
	                    opts=dict(title=DSP+name))

	_since_last_flush.clear()

	with open(LOGDIR+'log.pkl', 'wb') as f:
		pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)

def restore(LOGDIR,DSP):

	print('Try load previous plot....')

	with open(LOGDIR+'log.pkl', 'rb') as f:
		_since_beginning = pickle.load(f)
