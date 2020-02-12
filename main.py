import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from customDataset import DeepFakeSmallDataset
from vgg_lstm import VGGCNN, LSTM


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = DeepFakeSmallDataset(root_dir='../mouth-extraction-preprocessing/testing_frames_mouths', csv_file='../mouth-extraction-preprocessing/testing_labels.csv', transform=transform, frames=20)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=4)

    cnn = VGGCNN()
    cnn.cuda()
    cnn.eval()
    lstm = LSTM()
    lstm.cuda()
    lstm.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(lstm.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001, nesterov=True)
    #optimizer = optim.Adam(lstm.parameters(), amsgrad=True)

    correct = 0

    #scheduler = optim.lr_scheduler_StepLR(optimizer, lr_lambda=0.00)

    for epoch in range(9):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = images.cuda(), labels.cuda().long()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = lstm(cnn(images))
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
                print("training accuracy for batch:{}".format((correct / 80) * 100))
                running_loss = 0.0
                correct = 0
        torch.save(lstm.state_dict(), 'lstm_model_batch_4_SGD_nesterov_epoch_{}.pth'.format(epoch+1))

    print('Finished Training')
    torch.save(lstm.state_dict(), 'lstm_model.pth')