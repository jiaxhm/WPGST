import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
import matplotlib.pyplot as plt
import numpy as np
import datetime

def show_tensorll(a: torch.Tensor, fig_num=None, title=None):
    """Display a 2D tensor.
    args:
        fig_num: Figure number.
        title: Title of figure.
    """
    a_np = a.squeeze().cpu().clone().detach().numpy()
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # save_path = f"result_visual/image_{timestamp}.png"
    # plt.imsave(save_path, a_np, format='png')
    # a_np = a
    if a_np.ndim == 3:
        a_np = np.transpose(a_np, (1, 2, 0))
    plt.figure(fig_num)
    plt.tight_layout()
    # plt.tight_layout(pad=0)  # 也可以设为默认的，但这样图大点
    plt.cla()
    plt.imshow(a_np)
    plt.axis('off')
    plt.axis('equal')
    # plt.colorbar()  # 创建颜色条
    if title is not None:
        plt.title(title)
    plt.draw()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"result_visual/image_{timestamp}.png"
    plt.imsave(save_path, a_np, format='png')
    # plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.show()  # imshow是对图像的处理，show是展示图片
    plt.pause(0.1)
def show_tensor(a: torch.Tensor, fig_num=None, title=None):
    """Display a 2D tensor.
    args:
        fig_num: Figure number.
        title: Title of figure.
    """
    a_np = a.squeeze().cpu().clone().detach().numpy()
    if a_np.ndim == 3:
        a_np = np.transpose(a_np, (1, 2, 0))
    plt.figure(fig_num)
    plt.tight_layout()
    # plt.tight_layout(pad=0)  # 也可以设为默认的，但这样图大点
    plt.cla()
    plt.imshow(a_np, cmap='gray')
    plt.axis('off')
    plt.axis('equal')
    # a_np = (a_np - np.min(a_np)) / (np.max(a_np) - np.min(a_np))
    a_np[a_np < 0] = 0
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"result_visual/image_{timestamp}.png"
    plt.imsave(save_path, a_np, format='png',cmap='gray')
    # a_np = a
    if a_np.ndim == 3:
        a_np = np.transpose(a_np, (1, 2, 0))
    plt.figure(fig_num)
    plt.tight_layout()
    # plt.tight_layout(pad=0)  # 也可以设为默认的，但这样图大点
    plt.cla()
    plt.imshow(a_np, cmap='gray')
    plt.axis('off')
    plt.axis('equal')
    # plt.colorbar()  # 创建颜色条
    if title is not None:
        plt.title(title)
    plt.draw()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"result_visual/image_{timestamp}.png"
    plt.imsave(save_path, a_np, format='png',cmap='gray')
    # plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.show()  # imshow是对图像的处理，show是展示图片
    plt.pause(0.1)



# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
#         self.pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#
#     def forward(self, x):
#         return self.pointwise(self.depthwise(x))

class WaveletFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')

        self.conv_lh = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), groups=in_channels)
        self.conv_hl = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0), groups=in_channels)
        self.conv_hh = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)

        self.high_reduce = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1)

        self.low_reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

        self.softmax = nn.Softmax2d()

    def forward(self, x):
        # for k in x:
        #     show_tensorll(k.sum(dim=0))
        Yl, Yh = self.dwt(x)
        hl, lh, hh = Yh[0][:, :, 0, :, :], Yh[0][:, :, 1, :, :], Yh[0][:, :, 2, :, :]
        # for k in Yl:
        #     show_tensorll(k.sum(dim=0))
        # for k in hl:
        #     show_tensorll(k.sum(dim=0))
        # for k in lh:
        #     show_tensorll(k.sum(dim=0))
        # for k in hh:
        #     show_tensorll(k.sum(dim=0))

        hl = self.conv_hl(hl)
        lh = self.conv_lh(lh)
        hh = self.conv_hh(hh)

        high = torch.cat([hl, lh, hh], dim=1)
        high = self.high_reduce(high)
        high = self.softmax(high)

        low = self.low_reduce(Yl)

        fusion_mul = torch.mul(low, high)
        fusion_add = torch.add(low, fusion_mul)

        x_down = self.downsample(x)

        out = torch.add(x_down, fusion_add)
        # for k in out:
        #     show_tensorll(k.sum(dim=0))
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 36, 224, 224).float().to(device)
    model = WaveletFeatureFusion(36, 36)
    print(model)
    model = model.to(device)

    output = model(x)