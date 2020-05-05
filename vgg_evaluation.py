import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from customDataset import DeepFakeSmallDataset
from vgg import VGGCNN


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

    transform = transforms.Compose([transforms.ToTensor()])

    #validationset = DeepFakeSmallDataset(root_dir='../mouth-extraction-preprocessing/validation_frames_mouths', csv_file='../mouth-extraction-preprocessing/validation_labels.csv', transform=transform, frames=20)
    validationset = DeepFakeSmallDataset(root_dir='../mouth-extraction-preprocessing/frames_mouths_testing_all', csv_file='../mouth-extraction-preprocessing/testing_labels_all.csv', transform=transform, frames=20)
    #validationset = DeepFakeSmallDataset(root_dir='../mouth-extraction-preprocessing/frames_eyes_validation_100_vid', csv_file='../mouth-extraction-preprocessing/validation_labels_eyes.csv', transform=transform, frames=20)
    valloader = torch.utils.data.DataLoader(validationset, batch_size=4,
                                            shuffle=True, num_workers=4)

    cnn = VGGCNN()
    cnn.cuda()
    cnn.load_state_dict(torch.load('full_data_cnnmodel_for_lstm_2.pth'), strict=False)
    #cnn.load_state_dict(torch.load('cnn_eyes_epoch_5.pth'))
    cnn.eval()
    
    criterion = nn.CrossEntropyLoss()

    correct = 0

    running_loss = 0.0

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i, (images, labels) in enumerate(valloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = images.cuda(), labels.cuda().long()

        # forward + backward + optimize
        outputs = cnn(images)

        #Reshape sequence dimensions for labels and outputs to calculate loss on individual images
        labels = labels.view(4, 1)
        labels = torch.repeat_interleave(labels, repeats=20, dim=1)
        labels = labels.view(80)
        outputs = outputs.contiguous().view(80, -1)


        loss = criterion(outputs, labels)

        # print statistics
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

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
        if i % 20 == 0:    # print every 20 mini-batches
            print('[%5d] loss: %.3f' %
                (i + 1, running_loss / 20))
            #print(outputs)
            #print(labels)
            print("TP: {} FP: {} TN: {} FN: {}".format(true_positives, false_positives, true_negatives, false_negatives))
            print("validation accuracy for batch:{}".format((predicted == labels).sum().item()))
            running_loss = 0.0

    print('Finished validation')
    print((correct / (len(validationset) * 20)) * 100)
    print("TP: {} FP: {} TN: {} FN: {}".format(true_positives, false_positives, true_negatives, false_negatives))

