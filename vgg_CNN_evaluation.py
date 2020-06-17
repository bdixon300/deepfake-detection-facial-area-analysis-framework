import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from customDataset import DeepFakeSmallDataset
from vgg_architecture import VGGCNN

if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor()])

    # Alter path based on dataset being evaluated
    validationset = DeepFakeSmallDataset(root_dir='../../dataset/preprocessed_data/validation_frames_mouths', csv_file='../../dataset/preprocessed_data/validation_labels.csv', transform=transform, frames=20)
    valloader = torch.utils.data.DataLoader(validationset, batch_size=1,
                                            shuffle=False, num_workers=4)

    cnn = VGGCNN()
    cnn.cuda()
    # Change the model based on which experiment to run
    cnn.load_state_dict(torch.load('cnn_mouths_subset_data_epoch_8.pth'), strict=False)
    #cnn.load_state_dict(torch.load('trained_models_experiments/cnn_mouths_subset_data_epoch_8.pth'), strict=False)
    cnn.eval()

    criterion = nn.CrossEntropyLoss()

    correct = 0

    running_loss = 0.0

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i, (images, labels, sequence) in enumerate(valloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = images.cuda(), labels.cuda().long()

        # forward + backward + optimize
        outputs = cnn(images)

        #Reshape sequence dimensions for labels and outputs to calculate loss on individual images
        labels = torch.repeat_interleave(labels, repeats=20, dim=0)
        labels = labels.view(20)
        outputs = outputs.contiguous().view(20, -1)

        #print(labels)
        #print(outputs)

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

