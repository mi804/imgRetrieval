import torch
from torchvision import transforms, datasets

data_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 把图片resize为256*256
    transforms.RandomResizedCrop(224),  # 随机裁剪224*224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                          0.225])  # 标准化
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])  # 标准化
trainset = datasets.ImageFolder(
    root='data/folder/102flowers/test',
    transform=test_transform)  # 标签为{'cats':0, 'dogs':1}
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=64,
                                          shuffle=True)

testset = datasets.ImageFolder(root='data/folder/102flowers/train',
                               transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
