import torch
import torch.nn as nn
import data_preprocess
import numpy as np
import csv
import torch.nn.functional as F

num_epochs = 200
concat = False
lambda_adv = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out1 = self.relu(out)
        out = self.fc2(out1)
        out = self.relu(out)
        out = self.fc3(out)
        out = F.softmax(out, dim=1)
        return out, out1


class Discri(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Discri, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        return out

mean, std, feature_mat_train, feature_mat_val, gender_vec_train, gender_vec_val, income_vec_train, income_vec_val = \
    data_preprocess.train_process("train.csv", 0.8, True)


feature_mat_test,  gender_vec_test = data_preprocess.test_process("test_no_income.csv", mean, std, True)

if concat:
    model = NeuralNet(14, 128, 1).to(device)
else:
    print("using no cat")
    model = NeuralNet(13, 128, 2).to(device)
    discri = Discri(128, 64, 2).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizerD = torch.optim.Adam(discri.parameters(), lr=1e-4)

batch_size = 500


if concat:
    feature_mat_train_all = np.concatenate((feature_mat_train, np.expand_dims(gender_vec_train, 1)), axis=1)
    feature_mat_val_all = np.concatenate((feature_mat_val, np.expand_dims(gender_vec_val, 1)), axis=1)
    feature_mat_test_all = np.concatenate((feature_mat_test, np.expand_dims(gender_vec_test, 1)), axis=1)
else:
    feature_mat_train_all = feature_mat_train
    feature_mat_val_all = feature_mat_val
    feature_mat_test_all = feature_mat_test

for epoch in range(num_epochs):
    cur_order = np.random.permutation(feature_mat_train.shape[0])
    loss_all = 0
    lossD_all = 0
    cnt = 0
    for iter in range(feature_mat_train.shape[0]//batch_size):
        ibatch_start = iter * batch_size
        ibatch_end = (iter + 1) * batch_size

        project_ind_batch = cur_order[ibatch_start:ibatch_end]

        feature_mat_train_batch = feature_mat_train_all[project_ind_batch]
        income_vec_train_batch = income_vec_train[project_ind_batch]
        gender_vec_train_batch = gender_vec_train[project_ind_batch]

        in_fea = torch.from_numpy(feature_mat_train_batch).float().to(device)
        targets = torch.from_numpy(income_vec_train_batch).long().to(device)
        gen_mask = torch.from_numpy(gender_vec_train_batch).long().to(device)

        # gen_mask = gen_mask.resize(gen_mask.size(0), 1)
        outputs, fea = model(in_fea)
        #
        # loss_1 = criterion(outputs * gen_mask, targets.resize(targets.size(0), 1) * gen_mask)
        # loss_0 = criterion(outputs * (1 - gen_mask), targets.resize(targets.size(0), 1) * (1 - gen_mask))
        #
        # loss = loss_0 + loss_1

        optimizerD.zero_grad()

        pred_gen = discri(fea.detach())
        lossD = F.nll_loss(torch.log(outputs), gen_mask)
        lossD.backward(retain_graph=True)
        optimizerD.step()

        optimizer.zero_grad()
        loss = F.nll_loss(torch.log(outputs), targets)
        loss  = loss - lambda_adv * lossD
        loss.backward()
        optimizer.step()

        # print("loss = ", loss.item())
        loss_all += loss.item()
        lossD_all += lossD.item()
        cnt += 1
    # test
    print("loss = ", loss_all / cnt)
    print("lossD = ", lossD_all / cnt)

    val_fea = torch.from_numpy(feature_mat_val_all).float().to(device)
    pred_income = model(val_fea)[0].cpu().data.numpy()[:, 0]


    gen1_pred = np.asarray((pred_income > 0.5), dtype=np.float32)
    gen1_correct = (gen1_pred == income_vec_val) * gender_vec_val
    # print("gen1_pred", gen1_pred.shape, income_vec_val.shape, gender_vec_val.shape)

    # print("gen1_correct", gen1_correct, gen1_correct.shape)
    gen1_correct_num = np.sum(gen1_correct)

    gen0_pred = np.asarray((pred_income > 0.5), dtype=np.float32)
    gen0_correct = (gen0_pred == income_vec_val) * (1 - gender_vec_val)
    gen0_correct_num = np.sum(gen0_correct)

    total = gender_vec_val.shape[0]
    num1 = np.sum(gender_vec_val)
    num0 = total - num1

    pred_income_binary = np.asarray((pred_income > 0.5), dtype=np.float32)
    cor_num = np.sum((pred_income_binary == income_vec_val))

    print("accuracy of A= {} \n accuracy of B= {} \n overall accuracy = {}".format(gen1_correct_num / num1,
                                                                                   gen0_correct_num / num0,
                                                                                   cor_num / total))


test_fea = torch.from_numpy(feature_mat_test_all).float().to(device)
pred_income = model(test_fea).cpu().data.numpy()[:, 0]
pred_income_binary = np.asarray((pred_income > 0.5), dtype=np.int)

with open('test_pred.csv', 'w') as csvfile:
    # spamwriter = csv.writer(csvfile, delimiter=' ',
    #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # spamwriter.writerow('Id', 'income Spam')
    csvfile.write('Id' + "," + 'income' + "\n")
    print(pred_income.shape[0])
    for i in range(pred_income.shape[0]):
        csvfile.write(str(int(i))+","+ str(pred_income_binary[i]) + "\n")



