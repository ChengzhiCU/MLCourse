import csv
import numpy as np



def dataset_process(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        cnt =0
        row_str_list=[]
        for row in csv_file:
            if len(row)>1 and len(row)<40:
                row_str_list.append(row)
                cnt += 1

        y_gt = np.zeros((cnt, 1))
        features = np.zeros((cnt, 9))

        for i, row in enumerate(row_str_list):
            row_list = row.split(',')
            y_gt[i] = row_list[0]

            for j in range(1, 10):
                features[i, j-1] = row_list[j]

        type_0 = np.zeros((cnt - int(np.sum(y_gt)), 9))
        type_1 = np.zeros((int(np.sum(y_gt)), 9))

        c0 = 0
        c1 = 0
        for i in range(y_gt.shape[0]):
            if y_gt[i] == 0:
                type_0[c0] = features[i]
                c0 += 1
            elif y_gt[i] == 1:
                type_1[c1] = features[i]
                c1 += 1
    return y_gt, features, type_0, type_1



# print("features", features)
# print("gt", y_gt)

if __name__ == '__main__':
    trainfilename = 'propublicaTrain.csv'
    testfilename = 'propublicaTrain.csv'
    y_gt, features, type_0, type_1 = dataset_process(trainfilename)

    test_y_gt, test_features, test_type_0, test_type_1 = dataset_process(testfilename)

    mean0 = np.mean(type_0, axis=0)
    mean1 = np.mean(type_1, axis=0)

    A0 = type_0 - mean0
    A1 = type_1 - mean1

    print(A0.shape)
    print(A1)


    sigma0 = np.dot(np.transpose(A0), A0)
    sigma1 = np.dot(np.transpose(A1), A1)

    print(np.linalg.det(sigma0))
    print(np.linalg.det(sigma1))

    # print(sigma0, sigma1)
    # print(sigma0.shape, np.linalg.inv(sigma0))

    inv_simgma0 = np.linalg.inv(sigma0)
    inv_simgma1 = np.linalg.inv(sigma1)

    print(mean0, mean1)

    # for class 0

    correct = 0

    for i in range(test_features.shape[0]):
        data = test_features[i]
        p_y0 = - 0.5 * np.log(np.abs(np.linalg.det(sigma0))) - 0.5 * np.dot(np.dot((data-mean0),
                                                                                   np.linalg.inv(sigma0)), np.transpose(data-mean0))

        p_y1 = - 0.5 * np.log(np.abs(np.linalg.det(sigma1))) - 0.5 * np.dot(np.dot((data-mean1),
                                                                                   np.linalg.inv(sigma1)), np.transpose(data-mean1))

        if p_y0 >= p_y1:
            if y_gt[i] == 0:
                correct += 1
        else:
            if y_gt[i] == 1:
                correct += 1

    print("accuracy: {:8f}".format(correct * 1.0 / test_features.shape[0]))



