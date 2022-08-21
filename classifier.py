# Использованные ресурсы
# PyTorch = 'https://pytorch.org/'
# Help1 = 'https://neurohive.io/ru/tutorial/glubokoe-obuchenie-s-pytorch/'
# Help2 = 'https://docs.microsoft.com/ru-ru/windows/ai/windows-ml/tutorials/pytorch-train-model'
# Help3 = 'https://habr.com/ru/post/478208/'

from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import json

# try:
#     with open('settings.json') as f:
#         settings = json.load(f)                         # получаем данные файла настроек
# except IOError:
#     print('File "settings.json" is MISSING.')
#     wait = input("PRESS ENTER TO EXIT.")
#     raise SystemExit(1)
#
#
# WORK_DIRECTORY = settings['work_dir']
# POSITIVE_SET = settings['positive_set']
# NEGATIVE_SET = settings['negative_set']

WORK_DIRECTORY = "D:\Downloads\ComputerVision_test"
POSITIVE_SET = "00_hardworkers"
NEGATIVE_SET = "01_lazyobnes"
TRAIN_SET = "train_set"
TEST_SET = "test_set"
FRAME_WIDTH = 116
FRAME_HEIGHT = 116
BATCH_SIZE = 1000
LEARNING_RATE = 0.01                            # Скорость обучения
EPOCHS = 10                                     # Кол-во эпох обучения
LOG_INTERVAL = 10


class Net(nn.Module):
    def __init__(self):
        """На вход первого слоя приходят все пиксели изучаемых изображений, поэтому в качестве входного параметра
        принимаем перемножение высоты и ширины изображения
        Далее устанавливаем два скрытых слоя по 1000 нейронов в каждой
        На выход получаем 2 класса: рабочее место занято или свободно."""
        super(Net, self).__init__()
        self.fc1 = nn.Linear(FRAME_WIDTH * FRAME_HEIGHT, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 2)


def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return F.log_softmax(x)


# def train(epochs):
#     # запускаем главный тренировочный цикл
#     for epoch in tqdm(range(epochs)):
#         for batch_idx, (data, target) in enumerate(train_loader):
#             data, target = Variable(data), Variable(target)
#             # изменим размер с (batch_size, 1, 116, 116) на (batch_size, 116*116)
#             data = data.view(-1, FRAME_WIDTH * FRAME_HEIGHT)
#             optimizer.zero_grad()
#             net_out = net(data)
#             loss = criterion(net_out, target)
#             loss.backward()
#             optimizer.step()
#             if batch_idx % log_interval == 0:
#                 print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f}'.format(
#                     epoch, batch_idx * len(data), len(train_loader.dataset),
#                            100. * batch_idx / len(train_loader), loss.data[0]))


def train_test_loader():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=BATCH_SIZE, shuffle=True)
    return train_loader, test_loader



def main():
    print('Start classifier')
    # print('Work Dir: {}\n'
    #       'Positive Set: {}\n'
    #       'Negative Set: {}'.format(WORK_DIRECTORY, POSITIVE_SET, NEGATIVE_SET))
    net = Net()
    print("{}\n".format(net))
    # Осуществляем оптимизацию путем стохастического градиентного спуска
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # Создаем функцию потерь
    criterion = nn.NLLLoss()
    # запускаем главный тренировочный цикл
    for epoch in tqdm(range(EPOCHS)):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            # изменим размер с (batch_size, 1, 116, 116) на (batch_size, 116*116)
            data = data.view(-1, FRAME_WIDTH * FRAME_HEIGHT)
            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))


if __name__ == '__main__':
    main()

