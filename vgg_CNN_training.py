import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torch.nn as nn
from customDataset import DeepFakeSmallDataset
from vgg_architecture import VGGCNN
import numpy as np


transform = transforms.Compose([transforms.ToTensor()])

# Change path of dataset for experiment
trainset = DeepFakeSmallDataset(root_dir='../mouth-extraction-preprocessing/training_frames_mouths', csv_file='../mouth-extraction-preprocessing/training_labels_eyes.csv', transform=transform, frames=20)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                        num_workers=4, shuffle=True)

cnn = VGGCNN()
cnn.cuda()
cnn.train()

# Use unweighted loss for 100 video dataset
criterion = nn.CrossEntropyLoss()
# Use weighted loss for full dataset
#criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([30846/30846, 30846/3648]).to(device))
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

correct = 0

for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, (images, labels) in enumerate(trainloader, 0):
        try:
            # get the inputs; data is a list of [inputs, labels]
            images, labels = images.cuda(), labels.cuda().long()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnn(images)

            #Reshape sequence dimensions for labels and outputs to calculate loss on individual images
            #labels = labels.view(4, 1)
            labels = torch.repeat_interleave(labels, repeats=20, dim=1)
            labels = labels.view(20)
            outputs = outputs.contiguous().view(20, -1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            if i % 20 == 0:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 20))
                print("training accuracy for batch:{}".format((correct / (80*20)) * 100))
                running_loss = 0.0
                correct = 0
            # Save model name - this will be changed based on experiment
            torch.save(cnn.state_dict(), 'cnn_mouths_subset_data_epoch_{}.pth'.format(epoch+1))
        except ValueError:
            continue

print('Finished Training')
