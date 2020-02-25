import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torch.nn as nn
from customDataset import DeepFakeSmallDataset
from vgg import VGGCNN
import numpy as np


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = DeepFakeSmallDataset(root_dir='../mouth-extraction-preprocessing/training_frames_mouths', csv_file='../mouth-extraction-preprocessing/training_labels.csv', transform=transform, frames=20)
    
    # Split training data in half (one half for cnn, other half for lstm)
    indices = list(range(len(trainset)))
    split = int(np.floor(0.5*len(trainset)))
    random_seed=45
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices = indices[split:]
    
    """    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            num_workers=4, sampler=SubsetRandomSampler(train_indices))"""

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            num_workers=4, shuffle=True)

    cnn = VGGCNN()
    cnn.cuda()
    cnn.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

    correct = 0

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = images.cuda(), labels.cuda().long()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnn(images)

            #Reshape sequence dimensions for labels and outputs to calculate loss on individual images
            labels = labels.view(4, 1)
            labels = torch.repeat_interleave(labels, repeats=20, dim=1)
            labels = labels.view(80)
            outputs = outputs.contiguous().view(80, -1)

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
        torch.save(cnn.state_dict(), 'cnnmodel_for_lstm_{}.pth'.format(epoch+1))

    print('Finished Training')
    #torch.save(cnn.state_dict(), 'cnn_model_for_lstm.pth')