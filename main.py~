import torch
from torch.autograd import Variable

torch.manual_seed(1)

n_data = torch.ones(100, 2)
x0 = torch.normal(0*n_data, 1)
x1 = torch.normal(4*n_data, 1)
x2 = torch.normal(8*n_data, 1)
y0 = torch.zeros(100)
y1 = y0 + 1
y2 = y1 + 1
x = torch.cat((x0, x1, x2), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1, y2), ).type(torch.LongTensor)

x, y = Variable(x), Variable(y)

net = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 3),
    )

optim = torch.optim.Adam(net.parameters(), lr=0.05)
lossfunc = torch.nn.CrossEntropyLoss()

for i in range(100):
    output = net(x)
    prediction = torch.max(torch.nn.functional.softmax(output), 1)[1]
    loss = lossfunc(output, y)

    optim.zero_grad()
    loss.backward()
    optim.step()

    if i % 5 == 0:
        lossdata = loss.data.numpy().tolist()[0]

        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        accuracy = sum(pred_y == target_y)/300.
        print("step:{},loss:{:.3f},accuracy:{:.2f}%".format(i, lossdata, 100. * accuracy))

x = torch.FloatTensor([[0,0]])
x = Variable(x)
out = net(x)
predict = torch.max(torch.nn.functional.softmax(out), 1)[1]
print predict
tem = torch.LongTensor([[1]])
tem = Variable(tem)
if torch.equal(predict.data, tem.data):
    print "true"
else:
    print "false"
