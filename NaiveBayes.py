import numpy as np
from utils import *


def naive_bayes(percent=1.0):
    trainfilename = 'propublicaTrain.csv'
    testfilename = 'propublicaTest.csv'

    """
    def bin_feat1(x):   # age
        x = float(x)
        return int(x/5)

    def bin_feat6(x):   # count
        x = float(x)
        return int(x/3)

    binid = lambda x: x
    bins = None
    bins = [binid, bin_feat1, binid, binid, binid, binid, binid, binid, binid]
    """
    bins = None
    y_gt, _, features, type_0, type_1, _ = dataset_process(trainfilename, False, 0, int32=True, binfunc=bins,
                                                        percent=percent)
    test_y_gt, test_race_gt, test_features, test_type_0, test_type_1, _ = dataset_process(testfilename, False, 0, int32=True,
                                                                            binfunc=bins)

    # Training
    N = features.shape[0]
    M = features.shape[1]

    # Prior
    logp_Y0 = np.log(len(type_0) / N)
    logp_Y1 = np.log(len(type_1) / N)

    # Likelihood
    p_xiy0 = [{} for _ in range(M)]  # Count P(x|y)
    p_xiy1 = [{} for _ in range(M)]
    p_xy0 = [0 for _ in range(M)]  # Sum P(x|y)
    p_xy1 = [0 for _ in range(M)]
    diy0 = [0 for _ in range(M)]  # d in additive smoothing
    diy1 = [0 for _ in range(M)]

    for y0sample in type_0:
        for i in range(M):
            feati = y0sample[i]
            if feati not in p_xiy0[i]:
                p_xiy0[i][feati] = 0
            p_xiy0[i][feati] += 1
    for i in range(M):
        p_xy0[i] = sum(p_xiy0[i].values())
        diy0[i] = len(p_xiy0[i].keys())

    for y1sample in type_1:
        for i in range(M):
            feati = y1sample[i]
            if feati not in p_xiy1[i]:
                p_xiy1[i][feati] = 0
            p_xiy1[i][feati] += 1
    for i in range(M):
        p_xy1[i] = sum(p_xiy1[i].values())
        diy1[i] = len(p_xiy1[i].keys())

    correct = 0
    alpha = 1  # additive smoothing factor

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
        data = test_features[i]
        p_y0 = logp_Y0
        p_y1 = logp_Y1
        # p_y0, p_y1 = np.exp(logp_Y0), np.exp(logp_Y1)
        for j in range(1, M):
            featj = data[j]
            """
            if featj not in p_xiy0:
                p_y0 = -np.inf
            else:
                p_y0 += np.log(p_xiy0[j][featj] / p_xy0[j])
            if featj not in p_xiy1:
                p_y1 = -np.inf
            else:
                p_y1 += np.log(p_xiy1[j][featj] / p_xy1[j])
            """
            # p_y0 *= 1e-10 if featj not in p_xiy0[j] else p_xiy0[j][featj] / p_xy0[j]
            # p_y1 *= 1e-10 if featj not in p_xiy1[j] else p_xiy1[j][featj] / p_xy1[j]

            tmp = 0 if featj not in p_xiy0[j] else p_xiy0[j][featj]
            p_y0 += np.log((tmp + alpha) / (p_xy0[j] + diy0[j] * alpha))
            tmp = 0 if featj not in p_xiy1[j] else p_xiy1[j][featj]
            p_y1 += np.log((tmp + alpha) / (p_xy1[j] + diy1[j] * alpha))

        if p_y0 >= p_y1:
            if test_y_gt[i] == 0:
                correct += 1
        else:
            if test_y_gt[i] == 1:
                correct += 1


        if test_race_gt[i] == 0:  # Race
            r0_num += 1
        else:
            r1_num += 1

        if p_y0 >= p_y1:    # predict  not recivism  y_hat = 0 PP
            if test_y_gt[i] == 0:
                correct += 1
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
                correct += 1
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

    print("PP: R0 Y=0 | Y_hat=0 {:8f}".format(r0_PP_0_0 * 1.0 / (r0_PP_0_0 + r0_PP_1_0 + epis)))
    print("PP: R0 Y=1 | Y_hat=0 {:8f}".format(r0_PP_1_0 * 1.0 / (r0_PP_0_0 + r0_PP_1_0 + epis)))
    print("PP: R0 Y=0 | Y_hat=1 {:8f}".format(r0_PP_0_1 * 1.0 / (r0_PP_0_1 + r0_PP_1_1 + epis)))
    print("PP: R0 Y=1 | Y_hat=1 {:8f}".format(r0_PP_1_1 * 1.0 / (r0_PP_0_1 + r0_PP_1_1 + epis)))

    print("PP: R1 Y=0 | Y_hat=0 {:8f}".format(r1_PP_0_0 * 1.0 / (r1_PP_0_0 + r1_PP_1_0 + epis)))
    print("PP: R1 Y=1 | Y_hat=0 {:8f}".format(r1_PP_1_0 * 1.0 / (r1_PP_0_0 + r1_PP_1_0 + epis)))
    print("PP: R1 Y=0 | Y_hat=1 {:8f}".format(r1_PP_0_1 * 1.0 / (r1_PP_0_1 + r1_PP_1_1 + epis)))
    print("PP: R1 Y=1 | Y_hat=1 {:8f}".format(r1_PP_1_1 * 1.0 / (r1_PP_0_1 + r1_PP_1_1 + epis)))

    # print("percentage = {}, accuracy: {:8f}".format(percent, correct * 1.0 / test_features.shape[0]))
    return correct * 1.0 / test_features.shape[0]


if __name__ == '__main__':
    # xs = []
    # ys = []
    # for i in range(20):
    #     tmp = 0.0
    #     for j in range(10):
    #         percent = (i + 1) / 20.
    #         acc = naive_bayes(percent=percent)
    #         tmp += acc
    #     xs.append(percent)
    #     ys.append(tmp / 10.)
    # import matplotlib.pyplot as plt
    #
    # plt.plot(xs, ys)
    # plt.xlabel("Dataset Percentage")
    # plt.ylabel("Accuracy")
    # plt.title("Naive Bayes")
    # plt.savefig('naiveBayes.png')
    # plt.clf()

    naive_bayes(percent=1)
