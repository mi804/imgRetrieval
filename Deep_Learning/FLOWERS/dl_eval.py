import argparse
from model import OneObjectClassifier
import torch
import torchvision.transforms as transforms
import torchvision
import mmcv
import numpy as np
import multiprocessing
from functools import partial
import cv2
import torchvision.datasets as datasets
import os


def parse_args():
    parser = argparse.ArgumentParser(description='store binary')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--train_binary',
                        type=str,
                        default='work_dirs/binary/train/train_b128_8.txt',
                        help='train binary file')
    parser.add_argument('--work_dirs',
                        type=str,
                        default='work_dirs/retrieval_result/')
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--bit_size', type=int, default=128)
    parser.add_argument('--topk', type=int, default=10)
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


def imshow(img, win_name='', wait_time=0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    img = unnormalize(img)
    npimg = img.numpy().transpose((1, 2, 0))
    cv2.imshow(win_name, npimg)
    cv2.waitKey(wait_time)
    cv2.imwrite(win_name, (npimg*255))
    cv2.destroyWindow(win_name)


def main():
    args = parse_args()
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    vis_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    data_dir = "./data/folder/102flowers"
    testset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                   test_transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=8,
                                             shuffle=False,
                                             num_workers=4)
    vis_testset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                       vis_transform)
    vis_trainset = datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                        vis_transform)

    checkpoint = args.checkpoint
    from vgg_net import Net
    model = Net()
    model.load_state_dict(torch.load(checkpoint))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

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
            for j in range(len(vis_trainset)):
                train_binary = train_binarys[j]
                smstr = np.nonzero(query_binary - train_binary)
                similarities.append(np.shape(smstr[0])[0])
            similarities = np.array(similarities).reshape(-1, 1)
            index = np.argsort(similarities, axis=0)
            similarities = similarities[index.reshape(-1)]
            if args.show:
                img_vis = vis_testset[i][0]
                path = 'work_dirs/image_result/' + 'query_' + str(i)
                mmcv.mkdir_or_exist(path)
                imshow(img_vis, win_name=(path + '/query.png'))
                indexes = index[0:K, 0]
                for kk in range(1, K):
                    img = vis_trainset[indexes[kk]][0]
                    imshow(img,
                           win_name=(path + '/result_' + str(kk) + '.png'))

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
    mmcv.mkdir_or_exist(args.work_dirs + 'b' + str(args.bit_size) + '_' +
                        str(args.bs))
    precision_path = args.work_dirs + 'b' + str(args.bit_size) + '_' + str(
        args.bs) + '/precision.txt'
    Map_path = args.work_dirs + 'b' + str(args.bit_size) + '_' + str(
        args.bs) + '/Map.txt'
    for i in range(K):
        print('precision at top %d: %.4f' % (i + 1, precision_at_k[i][0]))
    Map = AP.mean()
    np.savetxt(precision_path, precision_at_k)
    np.savetxt(Map_path, np.array([Map]))
    print('map of cifar test set: %f ' % (Map))


if __name__ == '__main__':
    main()
