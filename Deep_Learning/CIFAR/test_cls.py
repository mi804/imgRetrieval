import argparse
from model import OneObjectClassifier
import torch
import torchvision.transforms as transforms
import torchvision
import mmcv
import numpy as np
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Test Classifier')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--bit_size', type=int, default=48)
    args = parser.parse_args()
    return args


# 展示图像的函数
def imshow(img, win_name='', wait_time=0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    cv2.imshow(win_name, np.transpose(npimg, (1, 2, 0)))
    cv2.waitKey(wait_time)


def main():
    args = parse_args()

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
                                           download=True,
                                           transform=transform)
    bs = args.bs
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=bs,
                                             shuffle=False,
                                             num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')

    checkpoint = args.checkpoint
    if_show = args.show
    model = OneObjectClassifier(binary_bits=args.bit_size)
    model.load_state_dict(torch.load(checkpoint))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not args.skip_cls_acc:
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        print('Running classification top1 precision calculating!')
        with torch.no_grad():
            prog_bar = mmcv.ProgressBar(len(testset))
            for data in testloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                batch_size = inputs.size(0)
                if if_show:
                    imshow(torchvision.utils.make_grid(inputs.cpu()))
                    print(
                        'GroundTruth: ',
                        ' '.join('%5s' % classes[labels[j]] for j in range(4)))
                    _, predicted = torch.max(outputs, 1)
                    print(
                        'Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                                for j in range(4)))
                for i in range(batch_size):
                    prog_bar.update()
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        correct_num = 0
        total_num = 0
        print()
        for i in range(10):
            correct_num += class_correct[i]
            total_num += class_total[i]
            print('Accuracy of %5s : %2d %%' %
                  (classes[i], 100 * class_correct[i] / class_total[i]))
        print('Accuracy of all %.2f %%' % (100 * correct_num / total_num))


if __name__ == '__main__':
    main()
