import sys
import os
import h5py
import numpy as np
import json
import codecs

fileName = sys.argv[1]
f = h5py.File(fileName, mode='r')['model_weights']
layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
for layer_name in layer_names:
	g = f[layer_name]
	weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
	for weight_name in weight_names:
		weight_value = np.asarray(g[weight_name].value)
		json.dump(weight_value.tolist(), codecs.open('weight_' + weight_name + '.json','w',encoding = 'utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
