# -*- coding: utf-8 -*-
# @Author: yuchen
# @Date:   2019-05-01 09:49:48
# @Last Modified by:   yuchen
# @Last Modified time: 2019-05-01 13:09:16

import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
from IPython import embed

def dataset1(n, r1=5, r2=10):
	angle1 = nr.uniform(0.0, np.pi * 2, size=n)
	circle1 = np.vstack((r1 * np.cos(angle1), r1 * np.sin(angle1)))

	angle2 = nr.uniform(0.0, np.pi * 2, size=n).T
	circle2 = np.vstack((r2 * np.cos(angle2), r2 * np.sin(angle2)))

	return np.hstack((circle1, circle2)).T

def iterator(dataset, centers):
	assignment = [np.argmin(np.sum((centers - datapoint) ** 2., axis=1)) for datapoint in dataset]
	pos = [[] for _ in centers]
	for i, cidx in enumerate(assignment):
		pos[cidx].append(dataset[i])
	for i, c in enumerate(pos):
		pos[i] = np.vstack(pos[i])
	newcenters = np.array([np.mean(pos[i], axis=0) for i in range(len(pos))])
	return newcenters, np.array(assignment)

def plot(dataset, centers, assignment, name="plt_1.png"):
	plt.clf()
	colors = ['b', 'g']
	for i in range(2):
		idx = (assignment == i)
		plt.scatter(dataset[idx, 0], dataset[idx, 1], marker='.', alpha=0.2, color=colors[i])
	plt.scatter(centers[0:1, 0], centers[0:1, 1], marker='x', color=colors[0])
	plt.scatter(centers[1:2, 0], centers[1:2, 1], marker='x', color=colors[1])
	plt.savefig(name)

def main():
	n = 1000
	k = 2
	sig = 0.1
	r = 1.0
	MAXITER = 500
	dataset = dataset1(n)
	centers = nr.multivariate_normal((0, 0), cov=[[sig, 0.0], [0.0, sig]], size=(k)) * r 

	for it in range(MAXITER):
		if it % 20 == 0:
			print("Iteration {}".format(it))
		newcenters, assignment = iterator(dataset, centers)
		diff = np.sum((centers - newcenters) ** 2.)
		centers = newcenters
		if diff < 1e-9:
			print("Stopped at iter {}".format(it))
			break
	plot(dataset, centers, assignment)



if __name__ == "__main__":
	main()