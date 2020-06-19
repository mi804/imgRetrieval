import argparse
from model import OneObjectClassifier
import torch
import torchvision.transforms as transforms
import torchvision
import mmcv
import numpy as np
import cv2
from vgg_net import Net


def parse_args():
    parser = argparse.ArgumentParser(description='Test Classifier')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--skip_cls_acc', action='store_true')
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


# 展示图像的函数
def imshow(img, win_name='', wait_time=0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    img = unnormalize(img)
    npimg = img.numpy()
    cv2.imshow(win_name, np.transpose(npimg, (1, 2, 0)))
    cv2.waitKey(wait_time)


def main():
    args = parse_args()
    data_transforms = {
        'test':
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 数据目录
    data_dir = "./data/folder/102flowers"

    # 获取三个数据集
    from torchvision import datasets
    import os

    testset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                   data_transforms['test'])

    batch_size = 8
    # print(dataloaders)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=4)
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    #            'ship', 'truck')

    checkpoint = args.checkpoint
    if_show = args.show
    model = Net()
    model.load_state_dict(torch.load(checkpoint))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not args.skip_cls_acc:
        class_correct = list(0. for i in range(102))
        class_total = list(0. for i in range(102))
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
                for i in range(batch_size):
                    prog_bar.update()
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        correct_num = 0
        total_num = 0
        print()
        for i in range(102):
            correct_num += class_correct[i]
            total_num += class_total[i]
        print('Accuracy of all %.2f %%' % (100 * correct_num / total_num))


if __name__ == '__main__':
    main()
