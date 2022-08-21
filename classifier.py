from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
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
# PyTorch = 'https://pytorch.org/'
# Help = 'https://neurohive.io/ru/tutorial/glubokoe-obuchenie-s-pytorch/'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(116 * 116, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 2)


def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return F.log_softmax(x)


def main():
    print('Start classifier')
    # print('Work Dir: {}\n'
    #       'Positive Set: {}\n'
    #       'Negative Set: {}'.format(WORK_DIRECTORY, POSITIVE_SET, NEGATIVE_SET))
    net = Net()
    print("{}\n".format(net))


if __name__ == '__main__':
    main()

