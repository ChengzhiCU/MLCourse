
import numpy as np
from utils import *

# print("features", features)
# print("gt", y_gt)

def KNN(percent=1.0):
    trainfilename = 'propublicaTrain.csv'
    testfilename = 'propublicaTest.csv'

    #filename, is_train, mean, int32=False, binfunc=None, Norm=False, std=None

    y_gt, _, features, type_0, type_1, mean = dataset_process(trainfilename, is_train= True, mean=[], Norm=False, std=[], percent=percent)
    test_y_gt, _, test_features, test_type_0, test_type_1, _ = dataset_process(testfilename, is_train=False, mean=mean, Norm=False)

    # y_gt, features, type_0, type_1, mean, std = dataset_process(trainfilename, is_train=True, mean=[], Norm=True, std=[])
    # test_y_gt, test_features, test_type_0, test_type_1, _, _ = dataset_process(testfilename, is_train=False, mean=mean,
    #                                                                         Norm=True, std=std)

    k=301
    print(k)
    correct = 0
    for row in range(test_features.shape[0]):
        x = test_features[row]

        dist = np.sum((x-features)**2, axis=1)

        x_norm = np.sum(x**2)**0.5
        # features_norm = np.sum(features**2, axis=1)**0.5
        # features_norm = np.expand_dims(features_norm, axis=1)
        # inner_product = x*features

        # dist = np.sum((inner_product)/x_norm/features_norm, axis=1)

        flat = dist.flatten()
        flat.sort()
        k_th_ele = flat[k-1]
        cnt_0=0
        cnt_1=0
        for i in range(dist.shape[0]):
            if dist[i] <= k_th_ele:
                if y_gt[i] == 0:
                    cnt_0 += 1
                elif y_gt[i] == 1:
                    cnt_1 += 1

        if cnt_0>=cnt_1:
            if test_y_gt[row] == 0:
                correct += 1
        else:
            if test_y_gt[row] == 1:
                correct += 1

    print("accuracy: {:8f}".format(correct * 1.0 / test_features.shape[0]))
    return correct * 1.0 / test_features.shape[0]

if __name__ == '__main__':

    xs = []
    ys = []
    for i in range(20):
        tmp = 0.0
        try:
            for j in range(10):
                percent = (i + 1) / 20.
                acc = KNN(percent=percent)
                tmp += acc
        except:
            # Sampled dataset size < k
            pass
        xs.append(percent)
        ys.append(tmp / 10.)
    import matplotlib.pyplot as plt
    plt.plot(xs[1:], ys[1:])
    plt.xlabel("Dataset Percentage")
    plt.ylabel("Accuracy")
    plt.title("KNN")
    plt.savefig('KNN.png')
    plt.clf()


