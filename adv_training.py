from models import ResNet, ConvNet
import torch.nn as nn
import argparse
from utils import UcrDataset, UCR_dataloader, AdvDataset
import torch.optim as optim
import torch.utils.data
import os
import random
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--test', action='store_true', help='')
parser.add_argument('--query_one', action='store_true', help='query the probability of  target idx sample')
parser.add_argument('--idx', type=int, help='the index of test sample ')
parser.add_argument('--gpu', type=str, default='0', help='the index of test sample ')
parser.add_argument('--channel_last', type=bool, default=True, help='the channel of data is last or not')
parser.add_argument('--n_class', type=int, default=2, help='the class number of dataset')
parser.add_argument('--epochs', type=int, default=1500, help='number of epochs to train for')
parser.add_argument('--e', default=1499, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--checkpoints_folder', default='model_checkpoints', help='folder to save checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--run_tag', default='ECG200', help='tags for the current run')
parser.add_argument('--model', default='', help='the model type(ResNet,FCN)')
parser.add_argument('--normalize', action='store_true', help='')
parser.add_argument('--checkpoint_every', default=5, help='number of epochs after which saving checkpoints')
opt = parser.parse_args()

print(opt)
# configure cuda
if torch.cuda.is_available() and not opt.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    print("You have a cuda device, so you might want to run with --cuda as option")

device = torch.device("cuda:0" if opt.cuda else "cpu")
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


def train():
    os.makedirs(opt.checkpoints_folder, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_folder, opt.run_tag), exist_ok=True)

    dataset_path = 'data/' + opt.run_tag + '/' + opt.run_tag + '_TRAIN.txt'
    dataset = UcrDataset(dataset_path, channel_last=opt.channel_last, normalize=opt.normalize)

    attacked_data_path = 'final_result/' + opt.run_tag + '/' + 'attack_time_series.txt'
    attacked_dataset = AdvDataset(txt_file=attacked_data_path)
    batch_size = int(min(len(dataset) / 10, 16))
    print('dataset length: ', len(dataset))
    print('number of adv examples：', len(attacked_dataset))
    print('batch size：', batch_size)
    dataloader = UCR_dataloader(dataset, batch_size)
    adv_dataloader = UCR_dataloader(attacked_dataset, batch_size)

    seq_len = dataset.get_seq_len()
    n_class = opt.n_class
    print('sequence len:', seq_len)


    if opt.model == 'r':
        net = ResNet(n_in=seq_len, n_classes=n_class).to(device)
    if opt.model == 'f':
        net = ConvNet(n_in=seq_len, n_classes=n_class).to(device)
    net.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    print('############# Start to Train ###############')
    for epoch in range(opt.epochs):
        for i, (data, label) in enumerate(dataloader):
            if data.size(0) != batch_size:
                break
            data = data.float()
            data = data.to(device)
            label = label.long()
            label = label.to(device)

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, label.view(label.size(0)))
            loss.backward()
            optimizer.step()
            print('#######Train on trainset########')
            print('[%d/%d][%d/%d] Loss: %.4f ' % (epoch, opt.epochs, i + 1, len(dataloader), loss.item()))
        for i, (data, label) in enumerate(adv_dataloader):
            if data.size(0) != batch_size:
                break
            data = data.float()
            data = data.to(device)
            label = label.long()
            label = label.to(device)

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, label.view(label.size(0)))
            loss.backward()
            optimizer.step()
            print('####### Adversarial Training ########')
            print('[%d/%d][%d/%d] Loss: %.4f ' % (epoch, opt.epochs, i + 1, len(dataloader), loss.item()))
        # End of the epoch,save model
        if (epoch % (opt.checkpoint_every * 10) == 0) or (epoch == (opt.epochs - 1)):
            print('Saving the %dth epoch model.....' % epoch)
            torch.save(net, '%s/%s/adv_%s%depoch.pth' % (opt.checkpoints_folder, opt.run_tag, opt.model, epoch))


def test():
    data_path = 'data/' + opt.run_tag + '/' + opt.run_tag + '_TEST.txt'
    dataset = UcrDataset(data_path, channel_last=opt.channel_last, normalize=opt.normalize)
    batch_size = int(min(len(dataset) / 10, 16))
    print('dataset length: ', len(dataset))
    print('batch_size:', batch_size)
    dataloader = UCR_dataloader(dataset, batch_size)
    type = opt.model
    model_path = 'model_checkpoints/' + opt.run_tag + '/adv_' + type + str(opt.e) + 'epoch.pth'
    model = torch.load(model_path, map_location='cpu')

    with torch.no_grad():
        model.eval()
        total = 0
        correct = 0
        for i, (data, label) in enumerate(dataloader):

            data = data.float()
            data = data.to(device)
            label = label.long()
            label = label.to(device)
            label = label.view(label.size(0))
            total += label.size(0)
            out = model(data)
            softmax = nn.Softmax(dim=-1)
            prob = softmax(out)
            pred_label = torch.argmax(prob, dim=1)

            correct += (pred_label == label).sum().item()

        print('The TEST Accuracy of %s  is :  %.2f %%' % (data_path, correct / total * 100))


def query_one(idx):
    data_path = 'data/' + opt.run_tag + '/' + opt.run_tag + '_TEST.txt'
    test_data = np.loadtxt(data_path)
    test_data = torch.from_numpy(test_data)

    test_one = test_data[idx]

    X = test_one[1:].float()
    X = X.to(device)
    y = test_one[0].long() - 1
    y = y.to(device)
    if y < 0:
        y = opt.n_class - 1
    print('ground truth：', y)
    type = opt.model
    model_path = 'model_checkpoints/' + opt.run_tag + '/adv_' + type + str(opt.e) + 'epoch.pth'
    model = torch.load(model_path, map_location='cpu')
    model.eval()

    out = model(X)
    softmax = nn.Softmax(dim=-1)
    prob_vector = softmax(out)
    print('prob vector：', prob_vector)
    prob = prob_vector.view(opt.n_class)[y].item()

    print('Confidence in true class of the %d sample is  %.4f ' % (idx, prob))


if __name__ == '__main__':
    if opt.test:
        test()
    elif opt.query_one:
        query_one(opt.idx)
    else:
        train()
