import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# This is the LRCN architecture design, the CNN takes in 20 frames and outputs feature maps for 20 frames. This is
# then fed to the lstm and fully connected layers.

# 2D CNN encoder using pretrained VGG16 (input is sequence of images)
class VGGCNN(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=25088):
        """Load the pretrained vgg 16 and replace top fc layer."""
        super(VGGCNN, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        vgg = models.vgg16(pretrained=True)
        modules = list(vgg.children())[:-1]      # delete the last fc layer.
        self.vgg = nn.Sequential(*modules)
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # VGG CNN
            with torch.no_grad():
                x = self.vgg(x_3d[:, t, :, :, :])  # VGG
                x = x.view(x.size(0), -1)             # flatten output of conv

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class LSTM(nn.Module):
    def __init__(self, CNN_embed_dim=25088, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.5, num_classes=2):
        super(LSTM, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, 128)
        self.bn1 = nn.BatchNorm1d(self.h_FC_dim, momentum=0.01)
        self.fc2 = nn.Linear(self.h_FC_dim, 64)
        self.bn2 = nn.BatchNorm1d(64, momentum=0.01)
        self.fc3 = nn.Linear(64, self.num_classes)

    def forward(self, x_RNN):
        
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  

        x = self.bn1(self.fc1(RNN_out[:, -1, :])) # Use value at last time step in sequence
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x