import torch
import torch.nn as nn


class MSDC(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # Use half-kernel padding to preserve spatial dims
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 9), padding=(0, 4), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (9, 1), padding=(4, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 15), padding=(0, 7), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (15, 1), padding=(7, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)


        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
   
        u = x.clone()
   
        attn = self.conv0(x)
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
   
        attn = attn + attn_0 + attn_1 + attn_2
     
        attn = self.conv3(attn)

        return attn * u
