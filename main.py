import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from customDataset import DeepFakeSmallDataset
from vgg_lstm import VGGCNN, LSTM

use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = DeepFakeSmallDataset(root_dir='./frames_mouths', csv_file='labels.csv', transform=transform, frames=20)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

"""testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)"""

cnn = VGGCNN()
cnn.cuda()
cnn.eval()
lstm = LSTM()
lstm.cuda()
lstm.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lstm.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, (images, labels) in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = images.to(device), labels.to(device).long()

        print(images.size())
        print(labels)

        #labels = torch.autograd.Variable(labels.long())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = lstm(cnn(images))
        print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(loss.item())
        # print statistics
        running_loss += loss.item()
        if i % 2 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2))
            running_loss = 0.0

print('Finished Training')