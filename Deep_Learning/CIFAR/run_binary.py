import argparse
from model import OneObjectClassifier
import torch
import torchvision.transforms as transforms
import torchvision
import mmcv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='store binary')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work_dirs',
                        type=str,
                        default='test',
                        help='work_dir')
    parser.add_argument('--run_test', action='store_true', help='run test set')
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--bit_size', type=int, default=48)
    args = parser.parse_args()
    return args


def run_train(model,
              device,
              bit_size=48,
              bs=64,
              work_dir='/work_dirs/train_binary'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data/cifar',
                                            train=True,
                                            download=False,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=bs,
                                              shuffle=False,
                                              num_workers=2)
    labels_train = np.empty(shape=[0, 1])
    print('Running train set for binary!')
    binarys = np.empty(shape=[0, bit_size])
    with torch.no_grad():
        prog_bar = mmcv.ProgressBar(len(trainset))
        for data in trainloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            binary = model.get_bianry_code(inputs)
            binarys = np.vstack((binarys, binary.cpu().numpy().astype(int)))

            la_np = labels.cpu().numpy()
            labels_train = np.vstack((labels_train, la_np.reshape(-1, 1)))

            batch_size = inputs.size(0)
            for i in range(batch_size):
                prog_bar.update()
    file_path = 'work_dirs/binary/train/' + work_dir + '.txt'
    label_file_path = 'work_dirs/binary/train/labels.txt'
    np.savetxt(file_path, binarys)
    np.savetxt(label_file_path, labels_train)


def run_test(model, device, bit_size=48, bs=64, work_dir='test'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data/cifar',
                                           train=False,
                                           download=False,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=bs,
                                             shuffle=False,
                                             num_workers=2)
    print('Running test set for binary!')
    binarys = np.empty(shape=[0, bit_size])
    with torch.no_grad():
        prog_bar = mmcv.ProgressBar(len(testset))
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            binary = model.get_bianry_code(inputs)
            np_binary = binary.cpu().numpy().astype(int)
            binarys = np.vstack((binarys, np_binary))
            batch_size = inputs.size(0)
            for i in range(batch_size):
                prog_bar.update()
    file_path = 'work_dirs/binary/test/' + work_dir + '.txt'
    np.savetxt(file_path, binarys)


def main():
    args = parse_args()

    checkpoint = args.checkpoint

    model = OneObjectClassifier(binary_bits=args.bit_size)
    model.load_state_dict(torch.load(checkpoint))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.run_test:
        run_test(model, device, args.bit_size, args.bs, args.work_dirs)
    else:
        run_train(model, device, args.bit_size, args.bs, args.work_dirs)


if __name__ == '__main__':
    main()
