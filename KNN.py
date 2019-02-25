
import numpy as np
from utils import *

# print("features", features)
# print("gt", y_gt)

def KNN(percent=1.0):
    trainfilename = 'propublicaTrain.csv'
    testfilename = 'propublicaTest.csv'

    #filename, is_train, mean, int32=False, binfunc=None, Norm=False, std=None

    y_gt, _, features, type_0, type_1, mean = dataset_process(trainfilename, is_train= True, mean=[], Norm=False, std=[], percent=percent)
    test_y_gt, test_race_gt, test_features, test_type_0, test_type_1, _ = dataset_process(testfilename, is_train=False, mean=mean, Norm=False)

    # y_gt, features, type_0, type_1, mean, std = dataset_process(trainfilename, is_train=True, mean=[], Norm=True, std=[])
    # test_y_gt, test_features, test_type_0, test_type_1, _, _ = dataset_process(testfilename, is_train=False, mean=mean,
    #                                                                         Norm=True, std=std)

    k=301
    print(k)
    print("shape", test_race_gt.shape, test_features.shape)
    correct = 0

    r0_PP_0_0 = 0
    r0_PP_0_1 = 0
    r0_PP_1_1 = 0
    r0_PP_1_0 = 0
    r1_PP_0_0 = 0
    r1_PP_0_1 = 0
    r1_PP_1_1 = 0
    r1_PP_1_0 = 0

    r0_EO_0_0 = 0
    r0_EO_0_1 = 0
    r0_EO_1_1 = 0
    r0_EO_1_0 = 0
    r1_EO_0_0 = 0
    r1_EO_0_1 = 0
    r1_EO_1_1 = 0
    r1_EO_1_0 = 0

    DP_r0_0 = 0
    DP_r0_1 = 0
    DP_r1_0 = 0
    DP_r1_1 = 0
    r0_num = 0
    r1_num = 0


    for i in range(test_features.shape[0]):
        x = test_features[i]

        dist = np.sum((x-features)**2, axis=1)

        # x_norm = np.sum(x**2)**0.5
        # features_norm = np.sum(features**2, axis=1)**0.5
        # features_norm = np.expand_dims(features_norm, axis=1)
        # inner_product = x*features

        # dist = np.sum((inner_product)/x_norm/features_norm, axis=1)

        flat = dist.flatten()
        flat.sort()
        k_th_ele = flat[k-1]
        cnt_0=0
        cnt_1=0
        for e in range(dist.shape[0]):
            if dist[e] <= k_th_ele:
                if y_gt[e] == 0:
                    cnt_0 += 1
                elif y_gt[e] == 1:
                    cnt_1 += 1

        p_y0 = 0
        p_y1 = 0
        if cnt_0>=cnt_1:
            p_y0 = 1
            if test_y_gt[i] == 0:
                correct += 1

        else:
            p_y1 = 1
            if test_y_gt[i] == 1:
                correct += 1


        if test_race_gt[i] == 0:  # Race
            r0_num += 1
        else:
            r1_num += 1

        if p_y0 >= p_y1:    # predict  not recivism  y_hat = 0 PP
            if test_y_gt[i] == 0:
                # correct += 1
                if test_race_gt[i] == 0:  # Race
                    DP_r0_0 += 1
                else:
                    DP_r1_0 += 1

            if test_race_gt[i] == 0:  #Race
                if test_y_gt[i] == 0:
                    r0_PP_0_0 += 1
                else:
                    r0_PP_1_0 += 1
            else:
                if test_y_gt[i] == 0:
                    r1_PP_0_0 += 1
                else:
                    r1_PP_1_0 += 1
        else:             # y_hat =1
            if test_y_gt[i] == 1:
                # correct += 1
                if test_race_gt[i] == 0:  # Race
                    DP_r0_1 += 1
                else:
                    DP_r1_1 += 1

            if test_race_gt[i] == 0:
                if test_y_gt[i] == 1:
                    r0_PP_1_1 += 1
                else:
                    r0_PP_0_1 += 1
            else:
                if test_y_gt[i] == 1:
                    r1_PP_1_1 += 1
                else:
                    r1_PP_0_1 += 1


        if test_y_gt[0]==0:           # EO

            if test_race_gt[i] == 0:  # Race
                if p_y0 >= p_y1:
                    r0_EO_0_0 += 1
                else:
                    r0_EO_1_0 += 1
            if test_race_gt[i] == 1:  # Race
                if p_y0 >= p_y1:
                    r1_EO_0_0 += 1
                else:
                    r1_EO_1_0 += 1
        else:
            if test_race_gt[i] == 0:  # Race
                if p_y0 >= p_y1:
                    r0_EO_0_1 += 1
                else:
                    r0_EO_1_1 += 1
            if test_race_gt[i] == 1:  # Race
                if p_y0 >= p_y1:
                    r1_EO_0_1 += 1
                else:
                    r1_EO_1_1 += 1

    print("accuracy: {:8f}".format(correct * 1.0 / test_features.shape[0]))
    print("DP: R0 y=0 {:8f}".format(DP_r0_0 * 1.0 / r0_num))
    print("DP: R0 y=1 {:8f}".format(DP_r0_1 * 1.0 / r0_num))
    print("DP: R1 y=0 {:8f}".format(DP_r1_0 * 1.0 / r1_num))
    print("DP: R1 y=1 {:8f}".format(DP_r1_1 * 1.0 / r1_num))
    print()

    # print("debug", r0_EO_0_0, r0_EO_1_0)
    epis = 1e-7

    print("EO: R0 Y_hat=0 | Y=0 {:8f}".format(r0_EO_0_0 * 1.0 / (r0_EO_0_0 + r0_EO_1_0 + epis)))
    print("EO: R0 Y_hat=1 | Y=0 {:8f}".format(r0_EO_1_0 * 1.0 / (r0_EO_0_0 + r0_EO_1_0 + epis)))
    print("EO: R0 Y_hat=0 | Y=1 {:8f}".format(r0_EO_0_1 * 1.0 / (r0_EO_0_1 + r0_EO_1_1 + epis)))
    print("EO: R0 Y_hat=1 | Y=1 {:8f}".format(r0_EO_1_1 * 1.0 / (r0_EO_0_1 + r0_EO_1_1 + epis)))

    print("EO: R1 Y_hat=0 | Y=0 {:8f}".format(r1_EO_0_0 * 1.0 / (r1_EO_0_0 + r1_EO_1_0 + epis)))
    print("EO: R1 Y_hat=1 | Y=0 {:8f}".format(r1_EO_1_0 * 1.0 / (r1_EO_0_0 + r1_EO_1_0 + epis)))
    print("EO: R1 Y_hat=0 | Y=1 {:8f}".format(r1_EO_0_1 * 1.0 / (r1_EO_0_1 + r1_EO_1_1 + epis)))
    print("EO: R1 Y_hat=1 | Y=1 {:8f}".format(r1_EO_1_1 * 1.0 / (r1_EO_0_1 + r1_EO_1_1 + epis)))
    print()

    print("PP: R0 Y_hat=0 | Y=0 {:8f}".format(r0_PP_0_0 * 1.0 / (r0_PP_0_0 + r0_PP_1_0 + epis)))
    print("PP: R0 Y_hat=1 | Y=0 {:8f}".format(r0_PP_1_0 * 1.0 / (r0_PP_0_0 + r0_PP_1_0 + epis)))
    print("PP: R0 Y_hat=0 | Y=1 {:8f}".format(r0_PP_0_1 * 1.0 / (r0_PP_0_1 + r0_PP_1_1 + epis)))
    print("PP: R0 Y_hat=1 | Y=1 {:8f}".format(r0_PP_1_1 * 1.0 / (r0_PP_0_1 + r0_PP_1_1 + epis)))

    print("PP: R1 Y_hat=0 | Y=0 {:8f}".format(r1_PP_0_0 * 1.0 / (r1_PP_0_0 + r1_PP_1_0 + epis)))
    print("PP: R1 Y_hat=1 | Y=0 {:8f}".format(r1_PP_1_0 * 1.0 / (r1_PP_0_0 + r1_PP_1_0 + epis)))
    print("PP: R1 Y_hat=0 | Y=1 {:8f}".format(r1_PP_0_1 * 1.0 / (r1_PP_0_1 + r1_PP_1_1 + epis)))
    print("PP: R1 Y_hat=1 | Y=1 {:8f}".format(r1_PP_1_1 * 1.0 / (r1_PP_0_1 + r1_PP_1_1 + epis)))


    print("accuracy: {:8f}".format(correct * 1.0 / test_features.shape[0]))
    return correct * 1.0 / test_features.shape[0]

if __name__ == '__main__':

    # xs = []
    # ys = []
    # for i in range(20):
    #     tmp = 0.0
    #     try:
    #         for j in range(10):
    #             percent = (i + 1) / 20.
    #             acc = KNN(percent=percent)
    #             tmp += acc
    #     except:
    #         # Sampled dataset size < k
    #         pass
    #     xs.append(percent)
    #     ys.append(tmp / 10.)
    # import matplotlib.pyplot as plt
    # plt.plot(xs, ys)
    # plt.xlabel("Dataset Percentage")
    # plt.ylabel("Accuracy")
    # plt.title("KNN")
    # plt.savefig('KNN.png')
    # plt.clf()

    KNN(percent=1)
