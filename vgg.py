import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 2D CNN encoder using pretrained VGG16 (input is sequence of images)
class VGGCNN(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3):
        """Load the pretrained vgg 16 and replace top fc layer."""
        super(VGGCNN, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        # Initialise pretrained model
        vgg = models.vgg16(pretrained=True)
        # Freeze feature layers
        """for param in vgg.features.parameters(): 
            param.requires_grad = False"""

        modules = list(vgg.children())[:-1]
        self.vgg = nn.Sequential(*modules)

        self.fc1 = nn.Linear(25088, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, 2)
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # VGG CNN
            with torch.no_grad():
                x = self.vgg(x_3d[:, t, :, :, :])
                x = x.view(x.size(0), -1)            

            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)
            x = torch.sigmoid(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq

cnn = VGGCNN()