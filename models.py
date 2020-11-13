import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, n_in, n_classes):
        super(ResNet, self).__init__()
        self.n_in = n_in
        self.n_classes = n_classes

        blocks = [1, 64, 128, 128]
        self.blocks = nn.ModuleList()
        for b, _ in enumerate(blocks[:-1]):
            self.blocks.append(ResidualBlock(*blocks[b:b + 2], self.n_in))

        self.fc1 = nn.Linear(blocks[-1], self.n_classes)

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)

        x = F.adaptive_avg_pool2d(x, 1)

        x = x.view(-1, 1, 128)
        x = self.fc1(x)
        # x = F.log_softmax(x,1)
        return x.view(-1, self.n_classes)


class ResidualBlock(nn.Module):
    def __init__(self, in_maps, out_maps, time_steps):
        super(ResidualBlock, self).__init__()
        self.in_maps = in_maps
        self.out_maps = out_maps
        self.time_steps = time_steps

        self.conv1 = nn.Conv2d(self.in_maps, self.out_maps, (7, 1), 1, (3, 0))
        self.bn1 = nn.BatchNorm2d(self.out_maps)

        self.conv2 = nn.Conv2d(self.out_maps, self.out_maps, (5, 1), 1, (2, 0))
        self.bn2 = nn.BatchNorm2d(self.out_maps)

        self.conv3 = nn.Conv2d(self.out_maps, self.out_maps, (3, 1), 1, (1, 0))
        self.bn3 = nn.BatchNorm2d(self.out_maps)

    def forward(self, x):
        x = x.view(-1, self.in_maps, self.time_steps, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        inx = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)) + inx)

        return x


# FCN model

class ConvNet(nn.Module):
    def __init__(self, n_in, n_classes):
        super(ConvNet, self).__init__()
        self.n_in = n_in
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(1, 128, (7, 1), 1, (3, 0))
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 256, (5, 1), 1, (2, 0))
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 128, (3, 1), 1, (1, 0))
        self.bn3 = nn.BatchNorm2d(128)

        self.fc4 = nn.Linear(128, self.n_classes)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 1, self.n_in, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(-1, 128)
        x = self.fc4(x)
        # return F.log_softmax(x,1)
        return x


if __name__ == '__main__':
    # resNet = ResNet(n_in=96, n_classes=3)
    # input = torch.randn(32, 96, 1)
    # out = resNet(input)
    # print(out.shape)
    torch.manual_seed(3)
    fcn = ConvNet(n_in=112, n_classes=3)
    input = torch.randn(32, 112, 1)
    out = fcn(input)
    print(out.size())
