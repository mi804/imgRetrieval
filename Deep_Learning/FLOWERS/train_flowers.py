import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from model import OneObjectClassifier
from vgg_net import Net
import torch.nn as nn
import time
import mmcv
device = torch.device("cuda:0")
# 对三种数据集进行不同预处理，对训练数据进行加强
data_transforms = {
    'train':
    transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val':
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
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

traindataset = datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                    data_transforms['train'])
valdataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                  data_transforms['val'])
testdataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                   data_transforms['test'])

batch_size = 8
# print(dataloaders)
traindataloader = torch.utils.data.DataLoader(traindataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4)

valdataloader = torch.utils.data.DataLoader(valdataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4)
testdataloader = torch.utils.data.DataLoader(testdataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=4)

dataset_sizes = {
    'train': len(traindataset),
    'test': len(testdataset),
    'val': len(valdataset)
}
print(dataset_sizes)
net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                   net.parameters()),
                            lr=0.0001,
                            momentum=0.9)


def val_model(model, criterion):
    best_acc = 0.0
    print('-' * 10)
    running_loss = 0.0
    running_corrects = 0
    model = model.to(device)
    for inputs, labels in valdataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels)
    epoch_loss = running_loss / dataset_sizes['val']
    print(running_corrects.double())
    epoch_acc = running_corrects.double() / dataset_sizes['val']
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('val', epoch_loss, epoch_acc))
    print('-' * 10)
    print()


def test_model(model, criterion):
    best_acc = 0.0
    print('-' * 10)

    running_loss = 0.0
    running_corrects = 0
    model = model.to(device)
    for inputs, labels in testdataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels)
    epoch_loss = running_loss / dataset_sizes['test']
    print(running_corrects.double())
    epoch_acc = running_corrects.double() / dataset_sizes['test']
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('test', epoch_loss, epoch_acc))
    print('-' * 10)
    print()


def train_model(model, criterion, optimizer, num_epochs=5):
    since = time.time()
    best_acc = 0.0
    loss_list = []
    acc_list = []
    prog_bar = mmcv.ProgressBar(len(traindataset) * num_epochs)
    for epoch in range(num_epochs):
        if (epoch + 1) % 5 == 0:
            test_model(model, criterion)
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        running_loss = 0.0
        running_corrects = 0
        model = model.to(device)
        for inputs, labels in traindataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels)
            batch_size = inputs.size(0)
            for _ in range(batch_size):
                prog_bar.update()
        epoch_loss = running_loss / dataset_sizes['train']
        print(dataset_sizes['train'])
        print(running_corrects.double())
        epoch_acc = running_corrects.double() / dataset_sizes['train']
        best_acc = max(best_acc, epoch_acc)
        loss_list.append(epoch_loss)
        acc_list.append(epoch_acc)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss,
                                                   epoch_acc))

        print()
        if epoch % 5 == 4 or epoch == num_epochs - 1:
            torch.save(
                net.state_dict(),
                './work_dirs/resnet152/resnet152_epoc{}.pth'.format(
                    str(epoch + 1)))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model


epochs = 48
model = train_model(net, criterion, optimizer, epochs)

val_model(model, criterion)

torch.save(model, 'model.pkl')