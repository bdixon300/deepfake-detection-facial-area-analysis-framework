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

    validationset = DeepFakeSmallDataset(root_dir='../mouth-extraction-preprocessing/frames_mouths_testing_all', csv_file='../mouth-extraction-preprocessing/testing_labels_all.csv', transform=transform, frames=20)
    #validationset = DeepFakeSmallDataset(root_dir='../mouth-extraction-preprocessing/frames_eyes_validation_100_vid', csv_file='../mouth-extraction-preprocessing/validation_labels_eyes.csv', transform=transform, frames=20)
    validationloader = torch.utils.data.DataLoader(validationset, batch_size=4,
                                            shuffle=False, num_workers=4)

    cnn = VGGCNN()
    cnn.cuda()
    cnn.load_state_dict(torch.load('full_data_cnnmodel_for_lstm_2.pth'), strict=False)
    #cnn.load_state_dict(torch.load('cnn_eyes_epoch_5.pth'), strict=False)
    cnn.eval()
    lstm = LSTM()
    lstm.cuda()
    lstm.load_state_dict(torch.load('full_data_cnnmodel_lstm_epoch_7.pth'))
    #lstm.load_state_dict(torch.load('eye_cnn_lstm_epoch_10.pth'))
    lstm.eval()

    # Best model so far
    #cnn.load_state_dict(torch.load('full_data_cnnmodel_for_lstm_2.pth'), strict=False)
    #lstm.load_state_dict(torch.load('full_data_cnnmodel_lstm_epoch_8.pth'))

    criterion = nn.CrossEntropyLoss()
    correct = 0

    running_loss = 0.0
    
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0


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

        """     print("predicted:{}".format(predicted))
        print("labels:{}".format(labels))"""

        for tensor_value in range(0, predicted.size(0)):
            # TP
            if (predicted[tensor_value].item() == 1 and labels[tensor_value].item() == 1):
                true_positives += 1
            # FP
            if (predicted[tensor_value].item() == 0 and labels[tensor_value].item() == 1):
                false_positives += 1
            #TN
            if (predicted[tensor_value].item() == 0 and labels[tensor_value].item() == 0):
                true_negatives += 1
            #FN
            if (predicted[tensor_value].item() == 1 and labels[tensor_value].item() == 0):
                false_negatives += 1

        running_loss += loss.item()

        if i % 20 == 0:
            print("accuracy:{}".format((predicted == labels).sum().item()))
            print("TP: {} FP: {} TN: {} FN: {}".format(true_positives, false_positives, true_negatives, false_negatives))

    print(running_loss / (len(validationset) / 4))  
    print(correct)
    print(len(validationset))
    print((correct / len(validationset)) * 100)
    print("TP: {} FP: {} TN: {} FN: {}".format(true_positives, false_positives, true_negatives, false_negatives))


