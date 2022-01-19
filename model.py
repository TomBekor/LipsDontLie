import torch
from torch import nn
import torchvision.models as models
import math

TRANSFORMER_D_MODEL = 128

pretrained_vgg = models.vgg11(pretrained=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        # Match the number of channels to 3 (RGB)
        self.up_conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        # Use pretrained convlution network feature extractor
        self.feature_extractor = pretrained_vgg.features
        # Apply linear network to match d_model features
        self.feed_forward = nn.Linear(1024, TRANSFORMER_D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_of_frames = x.size(0), x.size(1)
        # Reshape x to [batch_size*num_frames, channels, h, w] to extract feature maps
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.up_conv(x)
        x = self.feature_extractor(x)
        x = self.feed_forward(x.view(x.size(0), -1))
        # Reshape x back to [batch_size, num_frames, d_model]
        x = x.view(batch_size, num_of_frames, -1)
        return x


