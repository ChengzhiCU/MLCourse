
import numpy as np
from utils import *

# print("features", features)
# print("gt", y_gt)

from IPython import embed

def MLE(percent=1.0):

    trainfilename = 'propublicaTrain.csv'
    testfilename = 'propublicaTest.csv'

    # Note, for MLE, the normalization of the dataset is crucial.
    y_gt, _, features, type_0, type_1, mean = dataset_process(trainfilename, True, [], percent=percent)

    test_y_gt, _, test_features, test_type_0, test_type_1,_ = dataset_process(testfilename, False, mean)

    mean0 = np.mean(type_0, axis=0)
    mean1 = np.mean(type_1, axis=0)

    A0 = type_0 - mean0 # [len, 9]
    A1 = type_1 - mean1 # [l2, 9]
    #
    # print("****A0")
    # print(A0.shape)
    # print("****A1")
    # print(A1)
    # print("****")

    sigma0 = np.dot(np.transpose(A0), A0) / A0.shape[0] + np.diag([1e-7 for _ in range(9)])
    sigma1 = np.dot(np.transpose(A1), A1) / A1.shape[0] + np.diag([1e-7 for _ in range(9)])

    # embed()

    # print(sigma0)
    # print("***sigma1*")
    # print(sigma1)

    # print(np.linalg.det(sigma0))
    # print(np.linalg.det(sigma1))

    # print(sigma0, sigma1)
    # print(sigma0.shape, np.linalg.inv(sigma0))

    inv_simgma0 = np.linalg.inv(sigma0)
    inv_simgma1 = np.linalg.inv(sigma1)

    # for class 0

    correct = 0

    for i in range(test_features.shape[0]):
        data = test_features[i]
        p_y0 = - 0.5 * np.log(np.abs(np.linalg.det(sigma0))) - 0.5 * np.dot(np.dot((data-mean0),
                                                                                   np.linalg.inv(sigma0)), np.transpose(data-mean0))

        p_y1 = - 0.5 * np.log(np.abs(np.linalg.det(sigma1))) - 0.5 * np.dot(np.dot((data-mean1),
                                                                                   np.linalg.inv(sigma1)), np.transpose(data-mean1))

        if p_y0 >= p_y1:
            if test_y_gt[i] == 0:
                correct += 1
        else:
            if test_y_gt[i] == 1:
                correct += 1

    print("accuracy: {:8f}".format(correct * 1.0 / test_features.shape[0]))
    return correct * 1.0 / test_features.shape[0]

if __name__ == '__main__':

    xs = []
    ys = []
    for i in range(20):
        tmp = 0.0
        for j in range(10):
            percent = (i + 1) / 20.
            acc = MLE(percent=percent)
            tmp += acc
        xs.append(percent)
        ys.append(tmp / 10.)
    import matplotlib.pyplot as plt
    plt.plot(xs, ys)
    plt.xlabel("Dataset Percentage")
    plt.ylabel("Accuracy")
    plt.title("MLE")
    plt.savefig('MLE.png')
    plt.clf()