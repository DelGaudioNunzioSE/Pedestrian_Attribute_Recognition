import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ultralytics.nn.modules import CBAM
from torchvision.models import ConvNeXt_Base_Weights


class denseBlock(nn.Module):
    def __init__(self, hidden_dim=1024):
        super(denseBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, hidden_dim // 8)
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        return x




class CNNWithAttention(nn.Module):
    def __init__(self, num_classes=1, attention_heads=2, hidden_dim=256, channel='RGB'):
        super(CNNWithAttention, self).__init__()

        # ResNet Backbone
        #resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        #if(channel == 'L'): # use black and white images <---
        #    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #    print('Model: black and white mode')
        #self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Rimuove l'ultimo FC e la pool
        #self.resnet_out_channels = 2048  # Per resnet18/34, resnet50 usa 2048    


        # Carica ResNet-34 pre-addestrato
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Gestione delle immagini in bianco e nero (1 canale)
        if channel == 'L':  # Se le immagini sono in scala di grigio
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            print('Model: black and white mode')

        # Rimuovi l'ultimo layer fully connected e la parte di pooling finale
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Rimuove l'ultimo FC e la pool

        # Uscita della rete, la dimensione di uscita per ResNet-34 è 512 (come ResNet-18)
        self.resnet_out_channels = 512  # Per ResNet-34, l'output dei canali è 512




        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Multi-head Attention Layer
        self.cbam1 = CBAM(c1=self.resnet_out_channels)
        self.cbam2 = CBAM(c1=self.resnet_out_channels)
        self.cbam3 = CBAM(c1=self.resnet_out_channels)

        #self.proj = nn.Linear(self.resnet_out_channels, hidden_dim)

        
        # Fully Connected Layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)

        # Classifier
        self.classifier1 = nn.Linear(hidden_dim // 2, num_classes)
        self.classifier2 = nn.Linear(hidden_dim // 2, num_classes)
        self.classifier3 = nn.Linear(hidden_dim // 2, num_classes)
        
        # Activation and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Passaggio attraverso la CNN (Backbone)
        x = self.backbone(x)             # [B, 256, H/8, W/8]

        x1 = self.cbam1(x)
        x2 = self.cbam2(x)
        x3 = self.cbam3(x)

        # Pooling globale per ridurre le dimensioni spaziali
        x1 = F.adaptive_avg_pool2d(x1, (1, 1))  # Global Average Pooling to (batch_size, 2048, 1, 1)
        x1 = x1.view(x1.size(0), -1)  # Flatten to (batch_size, 2048)

        x2 = F.adaptive_avg_pool2d(x2, (1, 1))  # Global Average Pooling to (batch_size, 2048, 1, 1)
        x2 = x2.view(x2.size(0), -1)  # Flatten to (batch_size, 2048)

        x3 = F.adaptive_avg_pool2d(x3, (1, 1))  # Global Average Pooling to (batch_size, 2048, 1, 1)
        x3 = x3.view(x3.size(0), -1)  # Flatten to (batch_size, 2048)

        #x1 = self.proj(x1)  # [B, hidden_dim]
        #x2 = self.proj(x2)  # [B, hidden_dim]
        #x3 = self.proj(x3)  # [B, hidden_dim]


        # Passaggio attraverso i fully connected layers
        x1 = self.fc1(x1)  # [B, hidden_dim // 2]
        x2 = self.fc2(x2)  # [B, hidden_dim // 2]
        x3 = self.fc3(x3)  # [B, hidden_dim // 2]
        
        x1 = self.relu(x1)
        x2 = self.relu(x2)
        x3 = self.relu(x3)

        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x3 = self.dropout(x3)

        # Classifier
        out1 = self.classifier1(x1)  # [B, num_classes], num_classes perchè mi dà la probabilità
        out2 = self.classifier2(x2)  # [B, num_classes]
        out3 = self.classifier3(x3)  # [B, num_classes]


        return out1, out2, out3