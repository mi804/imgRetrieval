import argparse
from model import OneObjectClassifier
import torch
import torchvision.transforms as transforms
import torchvision
import mmcv
import numpy as np
from vgg_net import Net
import os
import torchvision.datasets as datasets


def parse_args():
    parser = argparse.ArgumentParser(description='store binary')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work_dirs',
                        type=str,
                        default='train_b128_8',
                        help='work_dir')
    parser.add_argument('--run_test', action='store_true', help='run test set')
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--bit_size', type=int, default=128)
    args = parser.parse_args()
    return args


def run_train(model,
              device,
              bit_size=128,
              bs=64,
              work_dir='/work_dirs/train_binary'):

    transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_dir = "./data/folder/102flowers"
    trainset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=8,
                                              shuffle=False,
                                              num_workers=4)
    # 记得改shuffle为False
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


def main():
    args = parse_args()

    checkpoint = args.checkpoint

    model = Net()
    model.load_state_dict(torch.load(checkpoint))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    run_train(model, device, args.bit_size, args.bs, args.work_dirs)


if __name__ == '__main__':
    main()
