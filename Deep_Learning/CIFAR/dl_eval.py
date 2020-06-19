import argparse
from model import OneObjectClassifier
import torch
import torchvision.transforms as transforms
import torchvision
import mmcv
import numpy as np
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='store binary')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--train_binary',
                        type=str,
                        default='work_dirs/binary/train/train_b48_64.txt',
                        help='train binary file')
    parser.add_argument('--work_dirs',
                        type=str,
                        default='work_dirs/retrieval_result/')
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--bit_size', type=int, default=48)
    parser.add_argument('--topk', type=int, default=1000)
    parser.add_argument('--show', action='store_true', help='show results')
    args = parser.parse_args()
    return args


def unnormalize(img):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    w_ = img.size()[1]
    h_ = img.size()[2]
    t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(3, w_, h_)
    t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, w_, h_)
    img = img * t_std + t_mean
    return img


def imshow(img, win_name='', wait_time=0, dir_=''):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    #         img_norm_cfg = dict(
    # mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    img = unnormalize(img)
    npimg = img.numpy().transpose((1, 2, 0))
    cv2.imshow(win_name, npimg)
    cv2.waitKey(wait_time)
    cv2.imwrite(dir_, npimg)


def main():
    args = parse_args()

    checkpoint = args.checkpoint

    model = OneObjectClassifier(binary_bits=args.bit_size)
    model.load_state_dict(torch.load(checkpoint))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if not args.show:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(3),
            transforms.RandomResizedCrop(32),  # 随机裁剪224*224
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
    testset = torchvision.datasets.CIFAR10(root='./data/cifar',
                                           train=False,
                                           download=False,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.bs,
                                             shuffle=False,
                                             num_workers=2)
    trainset = torchvision.datasets.CIFAR10(root='./data/cifar',
                                            train=True,
                                            download=False,
                                            transform=transform)

    print('Running test set for binary!')
    test_binarys = np.empty(shape=[0, args.bit_size])
    with torch.no_grad():
        prog_bar = mmcv.ProgressBar(len(testset))
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            binary = model.get_bianry_code(inputs)
            np_binary = binary.cpu().numpy().astype(int)
            test_binarys = np.vstack((test_binarys, np_binary))
            batch_size = inputs.size(0)
            for i in range(batch_size):
                prog_bar.update()
    print('')
    print('Reading train set binary!')
    train_binarys = np.loadtxt(args.train_binary)
    train_labels = np.loadtxt('work_dirs/binary/train/labels.txt')

    # calculate precision
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=2)
    print('running precision')
    K = args.topk
    AP = np.zeros((len(testset), 1))
    sum_tp = np.zeros((K, 1))
    with torch.no_grad():
        prog_bar = mmcv.ProgressBar(len(testset))

        def run_one_test(i):
            data = testset[i]
            image, query_label = data
            similarities = []
            # calculate similarity: hamming distance
            query_binary = test_binarys[i]
            for j in range(50000):
                train_binary = train_binarys[j]
                smstr = np.nonzero(query_binary - train_binary)
                similarities.append(np.shape(smstr[0])[0])
            similarities = np.array(similarities).reshape(-1, 1)
            index = np.argsort(similarities, axis=0)
            similarities = similarities[index.reshape(-1)]
            if args.show:
                dir_ = 'work_dirs/images/query' + str(i)
                mmcv.mkdir_or_exist(dir_)
                imshow(image, dir_=(dir_ + 'query_' + str(i) + '.png'))
                print(classes[query_label])
                indexes = index[0:10, 0]
                images = trainset[indexes[0]][0].unsqueeze(0)
                for kk in range(1, 10):
                    img = trainset[indexes[kk]][0].unsqueeze(0)
                    images = torch.cat([images, img], dim=0)
                imshow(torchvision.utils.make_grid(images, nrow=10),
                       dir_=(dir_ + 'result_' + str(kk) + '.png'))

            buffer_yes = np.zeros((K, 1))
            total_relevant = 0
            for j in range(K):
                retrieval_label = train_labels[index[j]][0]
                if query_label == retrieval_label:
                    buffer_yes[j] = 1
                    total_relevant = total_relevant + 1
            # compute precision
            P = np.divide(buffer_yes.cumsum().reshape(-1, 1),
                          np.arange(1, K + 1, 1).reshape(-1, 1))
            # return tuple((
            #     sum(P * buffer_yes) / sum(buffer_yes),
            #     buffer_yes.cumsum().reshape(-1, 1)))
            AP_ = np.zeros((1, ))
            if buffer_yes.sum() != 0:
                AP_ = sum(P * buffer_yes) / sum(buffer_yes)

            sum_tp_ = buffer_yes.cumsum().reshape(-1, 1)
            prog_bar.update()
            return (AP_, sum_tp_)

        for i in range(len(testset)):
            AP_, sum_tp_ = run_one_test(i)
            AP[i] = AP_
            sum_tp += sum_tp_

    precision_at_k = sum_tp / (np.arange(1, K + 1, 1).reshape(-1, 1) *
                               len(testset))
    precision_path = args.work_dirs + 'b' + str(args.bit_size) + '_' + str(
        args.bs) + '/precision.txt'
    Map_path = args.work_dirs + 'b' + str(args.bit_size) + '_' + str(
        args.bs) + '/Map.txt'
    for i in range(50):
        print('precision at top %d: %.4f' % (i + 1, precision_at_k[i][0]))
    Map = AP.mean()
    np.savetxt(precision_path, precision_at_k)
    np.savetxt(Map_path, np.array([Map]))
    print('map of cifar test set: %f ' % (Map))


if __name__ == '__main__':
    main()
