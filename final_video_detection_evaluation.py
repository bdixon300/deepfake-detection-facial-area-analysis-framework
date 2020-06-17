import dlib
import cv2
import pandas as pd
import os
from PIL import Image
from imutils import face_utils, resize
import numpy as np
from LRCN_architecture import VGGCNN, LSTM
from vgg_architecture import VGGCNN as VGGCNNClassifier
import torch
import torchvision
import torchvision.transforms as transforms

if __name__ == '__main__':

    # Setup mouth extractor
    p = "./shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # Set path for video to be evaluated
    #path = ""
    path = "../../dataset/FaceForensicsDatasetMouthFiltered/validation/original_sequences/"
    #path = "../../dataset/FaceForensicsDatasetMouthFilteredFull/validation/manipulated_sequences/"
    #path = "../../dataset/FaceForensicsDatasetMouthFilteredFull/testing/manipulated_sequences/"
    #path = "../../dataset/FaceForensicsDatasetMouthFiltered/validation/manipulated_sequences/"
    #path = "../../dataset/FaceForensicsDatasetMouthFiltered/validation/original_sequences/"
    #path = "../../dataset/youtube_example/"
    #path = "../../dataset/DeepfakeDetectionChallengeSubsetReal/"

    # Best model setup loaded
    filename = ''
    cnn = VGGCNN()
    cnn.cuda()
    cnn.load_state_dict(torch.load('trained_models_experiments/full_data_cnnmodel_for_lstm_2.pth'), strict=False)
    cnn.eval()
    lstm = LSTM()
    lstm.load_state_dict(torch.load('trained_models_experiments/extended_lstm_architecture_cnn_lstm_epoch_3.pth'), strict=False)
    lstm.cuda()
    lstm.eval()

    print("Evaluating cnn performance on its own at video level")

    real_vids = 0
    fake_vids = 0

    print("Evaluating real videos")

    # Iterate over videos to evaluate
    for filename in os.listdir(path):
        print("Evaluating video: {}".format(filename))
        voter_tally = 0

        vidcap = cv2.VideoCapture(path+filename)
        success = True

        frame_count = 1

        X = []

        while success:
            success,image = vidcap.read()
            # Only evaluate first 30 seconds to save time -- this may be altered to analyse the full video
            if not success or frame_count > 720:
                break
            # Extract faces
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = detector(gray, 0)
            largest_face_size = 0

            # Skip frame if no faces are found
            if len(faces) == 0:
                continue

            # Extract mouth area from largest face
            for (i, face) in enumerate(faces):
                # Make the prediction and transfom it to numpy array
                #face = face.rect
                shape = predictor(gray, face)
                shape = face_utils.shape_to_np(shape)
                size = face.width() * face.height()
                if largest_face_size < size:
                    largest_face_size = size

                    # Mouth region uses these indices for dlib
                    (i, j) = (48, 68)
                    # Eye region: 36, 48
                    # Face region: 1, 68

                    # clone the original image so we can draw on it, then
                    # display the name of the face part on the image
                    clone = image.copy()

                    # loop over the subset of facial landmarks, drawing the
                    # specific face part
                    for (x, y) in shape[i:j]:
                        cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                        roi = image[y:y + h, x:x + w]
                        roi = cv2.resize(roi, (224,224))
            X.append(transforms.ToTensor()(roi))
            # Evaluate sequences of 20 frames
            if frame_count % 20 == 0:
                X = torch.stack(X, dim=0)
                X = X.unsqueeze(0)
                outputs = lstm(cnn(X.cuda()))
                _, predicted = torch.max(outputs.data, 1)
                voter_tally += predicted.sum().item()
                X = []
            frame_count += 1
        # Aggregate predictions from all sequences for final prediction
        if voter_tally < (frame_count / 20) / 2:
            fake_vids += 1
            print("fake: video: {}".format(filename))
        else:
            real_vids += 1
            print("real: video: {}".format(filename))
    print("real vids: {}".format(real_vids))
    print("fake vids: {}".format(fake_vids))


