import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from kaldidata import KaldiDataset

torch.cuda.manual_seed(1)

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

model = Net()
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=args.momentum)

def train():
    accuracy = 0
    allnum = 0
    model.train()
    loss_func = nn.CrossEntropyLoss()
    loss_func.cuda()

    train_dataset =  KaldiDataset(scpfile=" ", labelfile=" ")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=256, shuffle=True)

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target[:,0]
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        pred = output.data.max(1)[1]
        allnum += 256
        accuracy += pred.eq(target.data).cpu().sum()
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('***set1,index:{:.0f},loss:{:.3f},accuracy:{}/{}({:.0f}%)'.format(batch_idx,loss.data[0],accuracy, allnum, 100. * accuracy / allnum)) 
    torch.save(model.state_dict(), 'final_dvector.pkl')


if  __name__ == '__main__':
    train()


