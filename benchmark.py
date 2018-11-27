import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patheffects as PathEffects
from centerLoss import CenterLoss
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.cm as cm

trainset = torchvision.datasets.MNIST(root='../mnist', train=True, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
))

testset = torchvision.datasets.MNIST(root='../mnist', train=True, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
))

train_loader = Data.DataLoader(dataset=trainset, batch_size=128, shuffle=True, num_workers=4)
test_loader = Data.DataLoader(dataset=testset, batch_size=128, shuffle=True, num_workers=4)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.extract = torch.nn.Sequential(
            torch.nn.Linear(784, 512),
            torch.nn.PReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.PReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.PReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.PReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.PReLU(),
            torch.nn.Linear(32, 2),
        )
        self.predict = torch.nn.Sequential(
            torch.nn.PReLU(),
            torch.nn.Linear(2, 10),
        )

    def forward(self, x):
        feature = self.extract(x.view(-1, 784))
        pred = F.log_softmax(self.predict(feature))
        return feature, pred


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.extract = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            torch.nn.PReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=5, padding=2),
            torch.nn.PReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=5, padding=2),
            torch.nn.PReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=5, padding=2),
            torch.nn.PReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, kernel_size=5, padding=2),
            torch.nn.PReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=5, padding=2),
            torch.nn.PReLU(),
            torch.nn.MaxPool2d(2, 2),
        )
        self.feat = torch.nn.Linear(128*3*3, 2)
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(2, 10)
        )

    def forward(self, x):
        x = self.extract(x)
        x = x.view(-1, 128*3*3)
        feat = self.feat(x)
        pred = F.log_softmax(self.pred(feat))
        return feat, pred


model = Net().cuda()
# model = ConvNet().cuda()
optimizer4nn = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
scheduler = lr_scheduler.StepLR(optimizer4nn, 20, gamma=0.8)

centerloss = CenterLoss(10, 2, 0.1).cuda()
nllloss = torch.nn.NLLLoss().cuda()
#crossentropy = torch.nn.CrossEntropyLoss().cuda()
optimizer4center = torch.optim.SGD(centerloss.parameters(), lr=0.5)


def train(train_loader, model, epoch):
    print("Training Epoch: {}".format(epoch))
    model.train()
    for step, (data, target) in enumerate(train_loader):
        data = Variable(data).cuda()
        target = Variable(target).cuda()
        feat, pred = model(data)
        loss = nllloss(pred, target) + centerloss(target, feat)
        optimizer4nn.zero_grad()
        optimizer4center.zero_grad()
        loss.backward()
        optimizer4nn.step()
        optimizer4center.step()
        if step % 100 == 0:
            print("Epoch: {} step: {}".format(epoch, step))


def test(test_loader, model, epoch):
    print("Predicting Epoch: {}".format(epoch))
    model.eval()
    total_pred_label = []
    total_target = []
    total_feature = []
    for step, (data, target) in enumerate(test_loader):
        data = Variable(data).cuda()
        target = Variable(target).cuda()
        feature, pred = model(data)
        _, pred_label = pred.max(dim=1)
        total_pred_label.append(pred_label.data.cpu())
        total_target.append(target.data.cpu())
        total_feature.append(feature.data.cpu())

    total_pred_label = torch.cat(total_pred_label, dim=0)
    total_target = torch.cat(total_target, dim=0)
    total_feature = torch.cat(total_feature, dim=0)

    precision = torch.sum(total_pred_label == total_target).item() / float(total_target.shape[0])
    print("Validation accuracy: {}%".format(precision * 100))
    scatter(total_feature.numpy(), total_target.numpy(), epoch)


def scatter(feat, label, epoch):
    plt.ion()
    plt.clf()
    palette = np.array(sns.color_palette('hls', 10))
    ax = plt.subplot(aspect='equal')
    # sc = ax.scatter(feat[:, 0], feat[:, 1], lw=0, s=40, c=palette[label.astype(np.int)])
    for i in range(10):
        plt.plot(feat[label == i, 0], feat[label == i, 1], '.', c=palette[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    ax.axis('tight')
    for i in range(10):
        xtext, ytext = np.median(feat[label == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=18)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])

    plt.draw()
    plt.savefig('./benchmark/centerloss_{}.png'.format(epoch))
    plt.pause(0.001)


for epoch in range(50):
    scheduler.step()
    train(train_loader, model, epoch)
    test(test_loader, model, epoch)
