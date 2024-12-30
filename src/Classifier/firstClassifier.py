import torch
import torch.nn as nn
import torchvision.models as models

class CNNWithAttention(nn.Module):
    def __init__(self, num_classes=1, attention_heads=2, hidden_dim=256):
        super(CNNWithAttention, self).__init__()

        # Backbone con VGG16 (congelato)
        resnet = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Rimuove l'ultimo FC e la pool
        self.resnet_out_channels = 512  # Per VGG16
        self.proj = nn.Linear(self.resnet_out_channels, hidden_dim)

        # Multi-head Attention Layer con Normalizzazione
        self.attention1 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=attention_heads, batch_first=True)
        self.attention2 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=attention_heads, batch_first=True)
        self.attention3 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=attention_heads, batch_first=True)


        # Fully Connected Layer con Batch Normalization
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)

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
        x = self.backbone(x)  # [B, 512, H/8, W/8]

        # Prepara l'input per il livello di attenzione
        B, C, H, W = x.size()  # B = Batch size, C = 512, H = H/8, W = W/8
        x = x.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, 512]
        x = self.proj(x)  # [B, H*W, hidden_dim]

        # Attention layers con LayerNorm
        attn1, _ = self.attention1(x, x, x)  # [B, H*W, hidden_dim]
        attn2, _ = self.attention2(x, x, x)
        attn3, _ = self.attention3(x, x, x)

        # Riduzione delle dimensioni (es. max pooling)
        x1 = attn1.max(dim=1).values  # [B, hidden_dim]
        x2 = attn2.max(dim=1).values
        x3 = attn3.max(dim=1).values

        # Fully Connected layers con BatchNorm
        x1 = self.fc1(x1)
        x1 = self.bn1(x1)  # BatchNorm
        x2 = self.fc2(x2)
        x2 = self.bn2(x2)  # BatchNorm
        x3 = self.fc3(x3)
        x3 = self.bn3(x3)  # BatchNorm

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
