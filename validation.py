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

    validationset = DeepFakeSmallDataset(root_dir='../mouth-extraction-preprocessing/validation_frames_mouths', csv_file='../mouth-extraction-preprocessing/validation_labels.csv', transform=transform, frames=20)
    validationloader = torch.utils.data.DataLoader(validationset, batch_size=4,
                                            shuffle=True, num_workers=4)

    cnn = VGGCNN()
    cnn.cuda()
    cnn.eval()
    lstm = LSTM()
    lstm.cuda()
    lstm.load_state_dict(torch.load('lstm_model_batch_4_lr_0.001_dropout_7_SGD_nesterov_epoch_8.pth'))
    #lstm.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    print(len(validationset))
    for i, (images, labels) in enumerate(validationloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = images.to(device), labels.to(device).long()

        # zero the parameter gradients
        # forward + backward + optimize
        outputs = lstm(cnn(images))
        # print statistics
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        running_loss = loss.item()
        print('[%5d] loss: %.3f' %
            (i + 1, running_loss))
        running_loss = 0.0
        print("accuracy:{}".format((predicted == labels).sum().item()))
    print(correct)
    print(len(validationset))
    print((correct / len(validationset)) * 100)

