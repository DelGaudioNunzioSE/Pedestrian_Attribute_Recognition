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
    def __init__(self, num_classes=1, hidden_dim=512, channel='RGB'):
        super(CNNWithAttention, self).__init__()

        # ResNet Backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        if(channel == 'L'): # use black and white images
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            print('Model: black and white mode')
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Removes the last FC and pooling layers
        self.resnet_out_channels = 2048  # For ResNet50, it uses 2048 channels    
        
        
        self.proj = nn.Conv2d(self.resnet_out_channels, hidden_dim, kernel_size=1, stride=1)

        # CBAM layers (use one for each attention layer)
        self.cbam1 = CBAM(c1=hidden_dim)
        self.cbam2 = CBAM(c1=hidden_dim)
        self.cbam3 = CBAM(c1=hidden_dim)

        # Global Average Pooling to reduce HxW to 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully Connected Layers
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
        # Pass through the CNN Backbone
        x = self.backbone(x)  # [B, 2048, H/8, W/8]

        x = self.proj(x)  # [B, H*W, hidden_dim] - Adjust this to match hidden_dim

        # Apply CBAM layers (spatial + channel attention)
        attn1 = self.cbam1(x)  # [B, H*W, hidden_dim]
        attn2 = self.cbam2(x)  # [B, H*W, hidden_dim]
        attn3 = self.cbam3(x)  # [B, H*W, hidden_dim]

        x1 = self.global_avg_pool(attn1)  # [B, hidden_dim, 1, 1]
        x2 = self.global_avg_pool(attn2)  # [B, hidden_dim, 1, 1]
        x3 = self.global_avg_pool(attn3)  # [B, hidden_dim, 1, 1]

        x1 = x1.view(x1.size(0), -1)  # [B, hidden_dim]
        x2 = x2.view(x2.size(0), -1)  # [B, hidden_dim]
        x3 = x3.view(x3.size(0), -1)  # [B, hidden_dim]

        # Pass through fully connected layers
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
        out1 = self.classifier1(x1)  # [B, num_classes]
        out2 = self.classifier2(x2)  # [B, num_classes]
        out3 = self.classifier3(x3)  # [B, num_classes]

        return out1, out2, out3
