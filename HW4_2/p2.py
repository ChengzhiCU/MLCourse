# -*- coding: utf-8 -*-
# @Author: yuchen
# @Date:   2019-04-23 17:14:57
# @Last Modified by:   yuchen
# @Last Modified time: 2019-04-23 17:47:23

import numpy as np
import scipy.linalg as linalg
import csv

from IPython import embed

DATA_PATH = "hw4_data.csv"

n = 0
names = None
D = None

def init():
    global n, names, D
    csv_reader = csv.reader(open(DATA_PATH, 'r'), delimiter=',')
    csv_data = list(csv_reader)
    # print(csv_data)
    names = csv_data[0][1:]
    n = len(names)

    D = np.zeros((n, n))
    for i, row in enumerate(csv_data[1:]):
        D[i, :] = np.array(list(map(int, row[1:])))

    assert(((D - D.T) * (D - D.T)).sum() < 1e-15)   # Check symmetry
    D = D.astype('float32')


def approx(W, Vl, k=2):

    # Here W can be a small negative real number due to numerical errors
    # But we won't take it according to our approximated algorithm
    W = np.real(W)
    largest_idxs = W.copy().argsort()[::-1][:k]
    tmp1 = Vl[:, largest_idxs]
    tmp2 = np.sqrt(np.diag(W[largest_idxs]))
    alpha = np.matmul(tmp1, tmp2)
    return alpha

def main():
    init()

    H = -1. / n * np.ones((n, n)) + np.diag(np.ones((n, )))
    B = -0.5 * np.matmul(H.T, np.matmul(D, H))

    W, Vl = linalg.eig(B)
    for k in range(1, 9):
        alpha = approx(W, Vl, k)

        D1 = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D1[i, j] = np.dot(alpha[i] - alpha[j], alpha[i] - alpha[j])
        deltaD1 = D - D1
        print("With top {} eigen values, L2 error sum = {}".format(k, (deltaD1 ** 2).sum()))

if __name__ == "__main__":
    main()