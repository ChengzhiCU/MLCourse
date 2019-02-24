import csv
import numpy as np

def dataset_process(filename, is_train, mean, int32=False, binfunc=None):
    if is_train and int32:
        raise ValueError("Cannot set int32 mode on in train mode (with normalization)")

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        cnt = 0
        row_str_list=[]
        for i, row in enumerate(csv_reader):
            if len(row)>1 and i>0:
                row_str_list.append(row)
                cnt += 1

        y_gt = np.zeros((cnt, 1))
        features = np.zeros((cnt, 9))
        if int32:
            y_gt, features = y_gt.astype('int32'), features.astype('int32')

        for i, row_list in enumerate(row_str_list):
            y_gt[i] = int(row_list[0])
            for j in range(1, 10):
                features[i, j-1] = row_list[j] if binfunc is None else binfunc[j-1](row_list[j])

        if is_train:
            # print("shape", features.shape, np.mean(features, axis=0).shape)
            mean = np.mean(features, axis=0)
            features = features - np.mean(features, axis=0)
        else:
            features = features - mean

        type_0 = np.zeros((cnt - int(np.sum(y_gt)), 9))
        type_1 = np.zeros((int(np.sum(y_gt)), 9))
        if int32:
            type_0, type_1 = type_0.astype('int32'), type_1.astype('int32')

        c0 = 0
        c1 = 0
        for i in range(y_gt.shape[0]):
            if y_gt[i] == 0:
                type_0[c0] = features[i]
                c0 += 1
            elif y_gt[i] == 1:
                type_1[c1] = features[i]
                c1 += 1
    return y_gt, features, type_0, type_1, mean