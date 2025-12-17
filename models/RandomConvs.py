import torch
import torch.nn as nn
import random

class RandomConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(RandomConv, self).__init__()
        self.in_channels = in_channels
        self.selected_channels = in_channels // 2

        self.conv = nn.Sequential(
            nn.Conv2d(self.selected_channels, self.selected_channels, kernel_size=kernel_size,
                      stride=1, padding=1, dilation=1, groups=self.selected_channels, bias=False),
            nn.SiLU()
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.SiLU()
        )

    def forward(self, x):
        B, C, H, W = x.size()
        indices = list(range(C))
        selected_indices = random.sample(indices, self.selected_channels)
        remaining_indices = list(set(indices) - set(selected_indices))

        selected = x[:, selected_indices, :, :]
        remaining = x[:, remaining_indices, :, :]

        selected_conv = self.conv(selected)

        combined = torch.cat([selected_conv, remaining], dim=1)
        out = self.fuse_conv(combined).permute(0, 2, 3, 1)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 144, 56, 56).float().to(device)
    model = RandomConv(144).to(device)
    print(model)

    output = model(x)
    print("Output shape:", output.shape)
