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
    def __init__(self, num_classes=1, attention_heads=2, hidden_dim=512, channel='RGB'):
        super(CNNWithAttention, self).__init__()

        # ResNet Backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        if(channel == 'L'): # use black and white images <---
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            print('Model: black and white mode')

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # remove FC e la pool

        self.resnet_out_channels = 2048  #  
        self.proj = nn.Linear(self.resnet_out_channels, hidden_dim)

        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Multi-head Attention Layer
        self.attention1 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=attention_heads, batch_first=True)
        self.attention2 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=attention_heads, batch_first=True)
        self.attention3 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=attention_heads, batch_first=True)

        
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

        # Prepara l'input per il livello di attenzione
        B, C, _, _ = x.size()         # B = Batch size, C = 256, H = H/8, W = W/8
        x = x.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, 256]
        x = self.proj(x)

        # Attention layers
        attn1, _= self.attention1(x, x, x)   # [B, H*W, 256]
        attn2, _ = self.attention2(x, x, x)   # [B, H*W, 256]
        attn3, _ = self.attention3(x, x, x)   # [B, H*W, 256]
        

        x1 = attn1.max(dim=1).values
        x2 = attn2.max(dim=1).values
        x3 = attn3.max(dim=1).values


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