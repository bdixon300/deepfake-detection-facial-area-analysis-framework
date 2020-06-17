import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from customDataset import DeepFakeSmallDataset
from vgg_architecture import VGGCNN as VGGCNNClassifier
from LRCN_architecture import VGGCNN, LSTM



if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])

    # These paths need to be changed to correspond to where the preprocessed datasets are stored

    validationsetface = DeepFakeSmallDataset(root_dir='../mouth-extraction-preprocessing/validation_frames_faces', csv_file='../mouth-extraction-preprocessing/validation_labels_faces.csv', transform=transform, frames=20)
    valloaderface = torch.utils.data.DataLoader(validationsetface, batch_size=1,
                                            shuffle=False, num_workers=4)
    
    validationseteye = DeepFakeSmallDataset(root_dir='../mouth-extraction-preprocessing/frames_eyes_validation_100_vid', csv_file='../mouth-extraction-preprocessing/validation_labels_eyes.csv', transform=transform, frames=20)
    valloadereye = torch.utils.data.DataLoader(validationseteye, batch_size=1,
                                            shuffle=False, num_workers=4)

    validationsetmouth = DeepFakeSmallDataset(root_dir='../mouth-extraction-preprocessing/validation_frames_mouths', csv_file='../mouth-extraction-preprocessing/validation_labels.csv', transform=transform, frames=20)
    valloadermouth = torch.utils.data.DataLoader(validationsetmouth, batch_size=1,
                                            shuffle=False, num_workers=4)
                                            
    # Change to this when using CNN evaluation not LRCN
    #cnnEye = VGGCNNClassifier()
    cnnEye = VGGCNN()
    cnnEye.cuda()
    # Swap model files for different experiments - these model files wont work with the final architecture used as the fc layers were altered
    cnnEye.load_state_dict(torch.load('trained_models_experiments/cnn_eyes_epoch_5.pth'), strict=False)
    cnnEye.eval()

    lstmEye = LSTM()
    # Swap model files for different experiments - these model files wont work with the final architecture used as the fc layers were altered
    lstmEye.load_state_dict(torch.load('trained_models_experiments/eye_cnn_lstm_epoch_8.pth'), strict=False)
    lstmEye.cuda()
    lstmEye.eval()
    
    # Change to this when using CNN evaluation not LRCN
    #cnnMouth = VGGCNNClassifier()
    cnnMouth = VGGCNN()
    cnnMouth.cuda()
    # Swap model files for different experiments - these model files wont work with the final architecture used as the fc layers were altered
    cnnMouth.load_state_dict(torch.load('trained_models_experiments/cnn_mouths_subset_data_epoch_8.pth'), strict=False)
    cnnMouth.eval()

    lstmMouth = LSTM()
    # Swap model files for different experiments - these model files wont work with the final architecture used as the fc layers were altered
    lstmMouth.load_state_dict(torch.load('trained_models_experiments/combined_vgg_lstm_better_architecture_epcoh_11.pth'), strict=False)
    lstmMouth.cuda()
    lstmMouth.eval()

    cnnFace = VGGCNNClassifier()
    cnnFace.cuda()
    # Swap model files for different experiments - these model files wont work with the final architecture used as the fc layers were altered
    cnnFace.load_state_dict(torch.load('trained_models_experiments/cnn_faces_epoch_6.pth'))
    cnnFace.eval()
    
    criterion = nn.CrossEntropyLoss()

    correct = 0

    running_loss = 0.0
    frame_count = 0
    voter_talley = 0

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    print("starting...")

    for i, (images, labels, sequence) in enumerate(valloaderface, 0):
        # get the inputs; data is a list of [inputs, labels]
        face_images, face_labels = images.cuda(), labels.cuda().long()
        mouth_images, mouth_labels, _ = validationsetmouth[i]
        eye_images, eye_labels, sequence = validationseteye[i]

        mouth_images = mouth_images.cuda().unsqueeze(0)
        eye_images = eye_images.cuda().unsqueeze(0)

        # forward + backward + optimize
        outputs_face = cnnFace(face_images)
        outputs_mouth = cnnMouth(mouth_images)
        outputs_eyes = cnnEye(eye_images)

        outputs_lstm_mouth = lstmMouth(cnnMouth(mouth_images))
        outputs_lstm_eyes = lstmEye(cnnEye(eye_images))

        # Alter this for different late fusion combinations
        #print(outputs_eyes)
        #outputs_late_fusion = torch.add(outputs_eyes, outputs_mouth) / 2
        #outputs_late_fusion = torch.add(outputs_eyes, outputs_face) / 2
        #outputs_late_fusion = torch.add(outputs_face, outputs_mouth) / 2
        #outputs_late_fusion = torch.add(outputs_eyes, outputs_mouth, outputs_face) / 3
        outputs_late_fusion = torch.add(outputs_lstm_eyes, outputs_lstm_mouth) / 2

        #Reshape sequence dimensions for labels and outputs to calculate loss on individual images
        #UNCOMMENT FOR CNN EVALUATION
        #face_labels = torch.repeat_interleave(face_labels, repeats=20, dim=0)

        loss = criterion(outputs_late_fusion, face_labels)

        _, predicted = torch.max(outputs_late_fusion.data, 1)
        correct += (predicted == face_labels).sum().item()
        voter_talley += predicted.sum().item()

        frame_count += 20

        if i % 35 == 0:
            print("evaluating video\: {}".format(sequence))
            #if voter_talley < frame_count / 2:
            if voter_talley < (frame_count / 20) / 2:
                print("fake")
            else:
                print("real")
            frame_count = 0
            voter_talley = 0

        for tensor_value in range(0, predicted.size(0)):
            # TP
            if (predicted[tensor_value].item() == 1 and face_labels[tensor_value].item() == 1):
                true_positives += 1
            # FP
            if (predicted[tensor_value].item() == 0 and face_labels[tensor_value].item() == 1):
                false_positives += 1
            #TN
            if (predicted[tensor_value].item() == 0 and face_labels[tensor_value].item() == 0):
                true_negatives += 1
            #FN
            if (predicted[tensor_value].item() == 1 and face_labels[tensor_value].item() == 0):
                false_negatives += 1

        running_loss += loss.item()
        if i % 20 == 0:    # print every 20 mini-batches
            print('[%5d] loss: %.3f' %
                (i + 1, running_loss / 20))
            print("TP: {} FP: {} TN: {} FN: {}".format(true_positives, false_positives, true_negatives, false_negatives))
            print("validation accuracy for batch:{}".format((predicted == face_labels).sum().item()))
            running_loss = 0.0

    print('Finished validation')
    print((correct / (len(validationset) * 20)) * 100)
    print("TP: {} FP: {} TN: {} FN: {}".format(true_positives, false_positives, true_negatives, false_negatives))

