import os
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torch.optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

root='./data'

transform_train=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
     ])

transform_val=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
     ])


f_mnist_train=datasets.CIFAR10(
    root=root,
    download=True,
    train=True,
    transform=transform_train
)


f_mnist_test=datasets.CIFAR10(
    root=root,
    download=True,
    train=False,
    transform=transform_val

)

batch_size=64


train_dataloader=DataLoader(f_mnist_train,
                            batch_size=batch_size,
                            shuffle=True
                            )

test_dataloader=DataLoader(f_mnist_test,
                            batch_size=batch_size,
                            shuffle=False
                            )

for (x,t) in train_dataloader:
    print(x.shape)
    print(t.shape)
    break

for (x,t) in test_dataloader:
    print(x.shape)
    print(t.shape)
    break

classes=('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(classes)
class CNN(nn.Module):

    def __init__(self):

        super().__init__()
        self.conv1=nn.Conv2d(in_channels=3,
                             out_channels=32,
                             kernel_size=3,
                             padding=True,
                             padding_mode='zeros')

        #正規化
        self.bn1=torch.nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               padding=True,
                               padding_mode='zeros')

        #正規化
        self.bn2 = torch.nn.BatchNorm2d(32)

        self.pool1=nn.MaxPool2d(2,2)
        self.dropout1=nn.Dropout2d(0.2)

        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               padding=True,
                               padding_mode='zeros')

        # 正規化
        self.bn3 = torch.nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               padding=True,
                               padding_mode='zeros')

        # 正規化
        self.bn4 = torch.nn.BatchNorm2d(64)

        self.pool2=nn.MaxPool2d(2,2)

        self.dropout2=nn.Dropout2d(0.2)

        self.conv5 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               padding=True,
                               padding_mode='zeros')

        # 正規化
        self.bn5 = torch.nn.BatchNorm2d(128)

        self.conv6= nn.Conv2d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=True,
                               padding_mode='zeros')

        # 正規化
        self.bn6 = torch.nn.BatchNorm2d(128)

        self.pool3=nn.MaxPool2d(2,2)

        self.dropout3=nn.Dropout2d(0.2)

        self.fc1=nn.Linear(128*4*4,128)

        self.dropout4=nn.Dropout2d(0.5)

        self.fc2=nn.Linear(128,10)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.bn1(x)
        x=F.relu(self.conv2(x))
        x=self.bn2(x)
        x=self.pool1(x)
        x=self.dropout1(x)
        x=F.relu(self.conv3(x))
        x=self.bn3(x)
        x=F.relu(self.conv4(x))
        x=self.bn4(x)
        x=self.pool2(x)
        x=self.dropout2(x)
        x=F.relu(self.conv5(x))
        x=self.bn5(x)
        x=F.relu(self.conv6(x))
        x=self.bn6(x)
        x=self.pool3(x)
        x=self.dropout3(x)

        x=x.view(-1,128*4*4)
        x=F.relu(self.fc1(x))
        x=self.dropout4(x)
        x=self.fc2(x)

        return x



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_dataloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(64)))

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=CNN().to(device)
print(model)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=0.0001)

def train_step(x,t):
    model.train()
    preds=model(x)
    loss=criterion(preds,t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss,preds

def test_step(x,t):
    model.eval()
    preds=model(x)
    loss=criterion(preds,t)

    return loss,preds


epochs=10
history={'loss':[],'accuracy':[],'test_loss':[],'test_accuracy':[]}


scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5,
    threshold=0.0001,
    verbose=True
)

for epoch in range(epochs):
    train_loss=0.
    train_acc=0.
    test_loss=0.
    test_acc=0.


    for(inputs,labels) in tqdm(train_dataloader):

        inputs,labels=inputs.to(device),labels.to(device)
        loss,preds=train_step(inputs,labels)
        train_loss+=loss.item()
        train_acc+=accuracy_score(
            labels.tolist(),
            preds.argmax(dim=-1).tolist()
        )

    for (inputs,labels) in tqdm(test_dataloader):

        inputs, labels = inputs.to(device), labels.to(device)
        loss, preds = test_step(inputs, labels)
        test_loss += loss.item()
        test_acc += accuracy_score(
            labels.tolist(),
            preds.argmax(dim=-1).tolist()
        )


    avg_train_loss = train_loss / len(train_dataloader)
    avg_train_acc = train_acc / len(train_dataloader)
    avg_test_loss = test_loss / len(test_dataloader)
    avg_test_acc = test_acc / len(test_dataloader)

    history['loss'].append(avg_train_loss)
    history['accuracy'].append(avg_train_acc)
    history['test_loss'].append(avg_test_loss)
    history['test_accuracy'].append(avg_test_acc)



    if (epoch + 1) % 1 == 0:
        print(
            "epoch{} train_loss:{:.4} train_acc:{:.4} val_loss:{:.4} val_acc:{:.4}"
                .format(
                epoch + 1,
                avg_train_loss,
                avg_train_acc,
                avg_test_loss,
                avg_test_acc
            ))


    scheduler.step(avg_test_acc)



# モデルを保存する
PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)



#モデルを使ってみる
dataiter = iter(test_dataloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(64)))
# 保存したモデルを読み込んで、予測する
model= CNN()
model.load_state_dict(torch.load(PATH))
outputs = model(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(64)))


#精度
correct = 0
total = 0
with torch.no_grad():
        for data in test_dataloader:
            inputs,labels=data
            #inputs = images.to(device)
            #labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))



#gpuにlabelとinputsとoutputs全部送るか
#cpudakedeyarebayoi


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for inputs,labels in test_dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        c=(predicted==labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


####精度完成け







plt.plot(history['loss'],
         marker='.',
         label='loss(Training)')

plt.plot(history['test_loss'],
         marker='.',
         label='loss(Test)')

plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()



plt.plot(history['accuracy'],
         marker='.',
         label='accuracy(Training)')

plt.plot(history['test_accuracy'],
         marker='.',
         label='accuracy(Test)')

plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()




