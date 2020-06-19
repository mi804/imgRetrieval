import torch
import torchvision
import torchvision.transforms as transforms
from model import OneObjectClassifier
import torch.optim as optim
import torch.nn as nn
import argparse
import mmcv


def parse_args():
    parser = argparse.ArgumentParser(description='Test Classifier')
    parser.add_argument('--work_dirs',
                        type=str,
                        default='b48_64',
                        help='eval types')
    parser.add_argument('--resume-from',
                        help='the checkpoint file to resume from')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--binary_bits', type=int, default=48)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪224*224
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data/cifar',
                                            train=True,
                                            download=False,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = OneObjectClassifier(binary_bits=48)
    if args.resume_from is not None:
        checkpoint = args.resume_from
        net.load_state_dict(torch.load(checkpoint))
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=5e-4)
    epochs = args.epochs
    print('start training of {} epocs'.format(epochs))
    prog_bar = mmcv.ProgressBar(len(trainset) * epochs)
    for epoch in range(epochs + 1):  # 多批次循环
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # import ipdb; ipdb.set_trace()
            # 梯度置0
            optimizer.zero_grad()

            # 正向传播，反向传播，优化
            outputs = net(inputs).to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_size = inputs.size(0)
            for _ in range(batch_size):
                prog_bar.update()

            # 打印状态信息
            running_loss += loss.item()
            if i % 100 == 99:  # 每50批次打印一次
                print('[%d,%5d] loss: %.10f , learning rate: %f' %
                      (epoch + 1, i + 1, running_loss / 99,
                       optimizer.param_groups[0]['lr']))
                running_loss = 0.0
        if epoch % 10 == 9:
            mmcv.mkdir_or_exist('./work_dirs/' + args.work_dirs)
            torch.save(
                net.state_dict(), './work_dirs/' + args.work_dirs +
                '/resnet18_epoc{}.pth'.format(str(epoch + 1)))


if __name__ == '__main__':
    main()
