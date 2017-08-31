import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from torch.autograd import Variable
from torchvision import datasets, transforms
from datautil import KaldiStreamDataloader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1140, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, 1024)
        self.fc7 = nn.Linear(1024, 1024)
        self.fc8 = nn.Linear(1024, 4000)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.dropout(x, p = 0.5, training = self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, p = 0.5, training = self.training)
        x = self.fc8(x)
        return x

    def eval(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.dropout(x, p = 0.5, training = self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, p = 0.5, training = self.training)
        return x.mean(0)

def test(model, enrollscp, testscp, feats_opts):
    list_dir = " "
    enroll_list = list_dir + " "
    test_list = list_dir + " "
    triallist = list_dir + " "
    enroll_dict = {}
    test_dict = {}
    torch.manual_seed(1)
    enroll_loader = KaldiStreamDataloader(enrollscp, enroll_list, feats_opts)
    test_loader = KaldiStreamDataloader(testscp, test_list, feats_opts)

    count = 0

    for batch_idx, (data,label) in enumerate(enroll_loader):
        label = int(label)
        label = np.array([label]) 
        label = torch.from_numpy(label)
        label = Variable(label)
        data = Variable(data[0])
        data = data.cuda()
        label = label.cuda()

        tem = model.eval(data)
        tem = tem.view(1024)
        tem = tem.data.cpu().numpy()

        if count == 0:
            result = tem
            count += 1
        elif count==1:
            result += tem
            count += 1
        elif count == 2:
            result += tem
            old_label = label.data.cpu().numpy().tolist()[0]
            count = 0
            result = result/3
            enroll_dict[old_label] = result
        else:
            pass

    ff = file("cos_score", "w+")

    for batch_idx, (data,label) in enumerate(test_loader):
        data = Variable(data[0])
        data = data.cuda()

        test_arr = model.eval(data)
        test_arr = test_arr.view(1024)
        test_arr = test_arr.data.cpu().numpy()

        test_dict[label] = test_arr


    for line in open(triallist):
        inp1 = line.split()[0]
        inp2 = line.split()[1]
        inp3 = line.split()[2]
        enroll_a = enroll_dict[int(inp1)]
        test_a = test_dict[inp2]
        
        lx = np.sqrt(test_a.dot(test_a))
        ly = np.sqrt(enroll_a.dot(enroll_a))
        cos = test_a.dot(enroll_a)/(lx * ly)
    
        ff.write(str(cos) + ' ' + inp3 + '\n')       

if __name__ == '__main__':
    data_dir = " "

    enroll_scp = data_dir + ' '
    test_scp = data_dir + ' '
    feats_opts = " "

    torch.manual_seed(1)

    model = Net()
    model.cuda()
    model.share_memory()
    model.load_state_dict(torch.load('final_dvector.pkl'))

    test(model, enroll_scp, test_scp, feats_opts)

