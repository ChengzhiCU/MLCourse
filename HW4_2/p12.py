# -*- coding: utf-8 -*-
# @Author: yuchen
# @Date:   2019-05-01 10:41:36
# @Last Modified by:   yuchen
# @Last Modified time: 2019-05-01 16:56:35


import numpy as np
import numpy.random as nr
import scipy.linalg as linalg
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

def plot(dataset, centers, assignment, name="plt_2"):
	plt.clf()
	colors = ['b', 'g']
	for i in range(2):
		idx = (assignment == i)
		plt.scatter(dataset[idx, 0], dataset[idx, 1], marker='.', alpha=0.5, color=colors[i])
	plt.scatter(centers[0:1, 0], centers[0:1, 1], marker='x', color=colors[0])
	plt.scatter(centers[1:2, 0], centers[1:2, 1], marker='x', color=colors[1])
	plt.title(name)
	plt.savefig(name + ".png") 


def rnngraph(dataset, r):
	n = dataset.shape[0]
	distmat = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			distmat[i, j] = np.sum((dataset[i] - dataset[j]) ** 2.) if i != j else np.inf
	mat = np.zeros((n, n)).astype('int32')
	for i in range(n):
		r_nn = np.argsort(distmat[i])[:r]
		for j in r_nn:
			mat[i, j] = 1
	return mat

def findcc(graph):
	vecs = []
	n = graph.shape[0]
	found = set()
	inqueue = set()
	i = 0
	import queue
	q = queue.Queue(n)
	status = np.zeros((n, ))	# 0: not found, 1: in queue, 2: found
	while i < n:
		if status[i] == 2:
			i += 1
			continue
		elif status[i] == 1:
			raise ValueError("Impossible executing logic")
		else:
			pass

		vec = np.zeros((n, ))
		q.put(i)
		status[i] = 1

		while not q.empty():
			x = q.get()
			# print("Queue size: {}".format(q.qsize()))
			if status[x] == 2:
				continue
			status[x] = 2
			vec[x] = 1
			for j in range(n):
				if graph[x, j] == 1 and status[j] == 0:
					q.put(j)
					status[j] = 1
		vecs.append(vec)
	return vecs

def main():
	n = 400
	k = 2
	sig = 0.1
	r = 2.0
	rnum = 50
	MAXITER = 500
	dataset = dataset1(n)

	print("Computing relative distance")
	W = rnngraph(dataset, rnum)

	print("Calculating connected components")
	vecs = findcc(W)

	dataV = np.vstack(vecs).T
	print("#CC = {}".format(dataV.shape[1]))
	centers = []
	for i in range(k):
		tmp = np.zeros((k))
		tmp[i] = 1
		centers.append(tmp)

	for it in range(MAXITER):
		if it % 20 == 0:
			print("Iteration {}".format(it))
		newcenters, assignment = iterator(dataV, centers)
		diff = np.sum((centers - newcenters) ** 2.)
		centers = newcenters
		if diff < 1e-9:
			print("Stopped at iter {}".format(it))
			break

	plot(dataset, centers, assignment)



if __name__ == "__main__":
	main()