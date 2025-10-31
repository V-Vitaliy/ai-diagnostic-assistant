import torch.nn as nn
from torchvision import models

class PretrainedDensenet(nn.Module):
    def __init__(self, num_class=1):
        super().__init__()
        self.channels = 1664
        densenet_169 = models.densenet169(pretrained=True)
        # --- Freeze parameters ---
        for param in densenet_169.parameters():
            param.requires_grad = False

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=4)
        self.features = nn.Sequential(*list(densenet_169.features.children()))
        # Define relu instance
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.channels, num_class)
        # --- Sigmoid layer is removed ---
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        features = self.features(x)
        # --- Use the relu instance correctly ---
        out = self.relu(features)
        # ------------------------------------
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(-1, self.channels)

        # --- Return raw logits, NOT sigmoid probability ---
        return self.fc1(out)