import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ultralytics.nn.modules import CBAM
from torchvision.models import ConvNeXt_Base_Weights




class CNNWithAttention(nn.Module):
    def __init__(self, num_classes=1, attention_heads=2, hidden_dim=256, channel='RGB'):
        super(CNNWithAttention, self).__init__()

        # ResNet Backbone
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # remove FC e la pool

        self.resnet_out_channels = 512 # 
        

        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Multi-head Attention Layer
        self.attention1 = nn.MultiheadAttention(self.resnet_out_channels, self.resnet_out_channels//8, batch_first=True)
        self.attention2 = nn.MultiheadAttention(self.resnet_out_channels, self.resnet_out_channels//8, batch_first=True)
        self.attention3 = nn.MultiheadAttention(self.resnet_out_channels, self.resnet_out_channels//8, batch_first=True)


        # Classifier
        self.classifier1 = nn.Linear(self.resnet_out_channels, num_classes)
        self.classifier2 = nn.Linear(self.resnet_out_channels, num_classes)
        self.classifier3 = nn.Linear(self.resnet_out_channels, num_classes)




    def forward(self, x):
        # Passaggio attraverso la CNN (Backbone)
        x = self.backbone(x)             # [B, 256, H/8, W/8]

        # Prepara l'input per il livello di attenzione
        B, C, _, _ = x.size()         # B = Batch size, C = 256, H = H/8, W = W/8
        x = x.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, 256]

        
        attn1, _ = self.attention1(x, x, x)  # [B, H*W, 512]
        x1 = attn1.mean(dim=1)  # Global average pooling (B, 512)

        # Apply the second attention layer for classifier 2
        attn2, _ = self.attention2(x, x, x)  # [B, H*W, 512]
        x2 = attn2.mean(dim=1)  # Global average pooling (B, 512)

        # Apply the third attention layer for classifier 3
        attn3, _ = self.attention3(x, x, x)  # [B, H*W, 512]
        x3 = attn3.mean(dim=1)  # Global average pooling (B, 512)


        # Classifier
        out1 = self.classifier1(x1)  # [B, num_classes], num_classes perchè mi dà la probabilità
        out2 = self.classifier2(x2)  # [B, num_classes]
        out3 = self.classifier3(x3)  # [B, num_classes]


        return out1, out2, out3