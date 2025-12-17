# coding=gbk

import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from models.model_util import *
from train_util import *
from timm.models.layers import trunc_normal_
import math
from models.local_trans3 import *
from models.WAdemo2 import WaveletFeatureFusion
from models.RandomConvs import RandomConv
import datetime

# define the function includes in import *
__all__ = [
    'lformerr3'
]

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
    save_path = f"result_visual2/image0823_{timestamp}.png"
    plt.imsave(save_path, a_np, format='png', cmap='gray')
    # plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.show()  # imshow是对图像的处理，show是展示图片
    plt.pause(0.1)

class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=nn.SiLU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        # drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        # self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        # self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop1(x)
        x = self.fc2(x)
        # x = self.drop2(x)
        return x

class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

def conv3x3(in_planes, out_planes):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)

class sam_residual(nn.Module):
    def __init__(self, inplanes, planes):
        super(sam_residual, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Sequential(conv3x3(planes, planes))#, simam_module(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.silu(out)

        return out

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, upsize, mid_size):  #channel or spatial
        super(unetUp, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(upsize, mid_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(mid_size),
            nn.SiLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_size),
            nn.SiLU(True)
        )

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        return outputs

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes=9):
        super(Unet, self).__init__()
        self.conv0 = nn.Sequential(
                        nn.Conv2d(3, 36, kernel_size=3, stride=1, padding=1),  # 208
                        sam_residual(36, 36))

        self.down1 = nn.Sequential(
                        WaveletFeatureFusion(36, 72),  # 104
                        sam_residual(72, 72))

        self.down2 = nn.Sequential(
                        WaveletFeatureFusion(72, 144),  # 52
                        LocalFormerBlock11(144),
        )
        # LocalFormerBlock11(144),

        self.down3 = nn.Sequential(
                        WaveletFeatureFusion(144, 288),  # 26
                        LocalFormerBlock22(288),
        )
        # LocalFormerBlock22(288),
        # LocalFormerBlock22(288),

        self.down4 = nn.Sequential(
                        WaveletFeatureFusion(288, 576)  # 13
                        )

        self.supertrans111 = LocalFormerBlock33(576)#PVMLayer(576, 576)
        self.supertrans222 = LocalFormerBlock33(576)#PVMLayer(576, 576)

        self.dlformer1 = LocalFormerBlock11(144)
        # self.dlformer12 = LocalFormerBlock11(144)
        self.dlformer2 = LocalFormerBlock22(288)
        # self.dlformer22 = LocalFormerBlock22(288)
        # self.dlformer23 = LocalFormerBlock22(288)
        self.dsam1 = sam_residual(36, 36)
        self.dsam2 = sam_residual(72, 72)

        self.up_concat3 = unetUp(576, 288, 576, 288)

        self.up_concat2 = unetUp(288, 144, 288, 144)

        self.up_concat1 = unetUp(144, 72, 144, 72)

        self.up_concat0 = unetUp(72, 36, 72, 36)

        self.final = nn.Conv2d(39, num_classes, 1)
        self.softmax = nn.Softmax(1)

        self.edge = nn.Sequential(
            nn.Conv2d(39, 32, kernel_size=3, padding=1, dilation=1, bias=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, dilation=1, bias=True),
            nn.Conv2d(16, 1, kernel_size=1, padding=0, dilation=1, bias=True)
        )

    def forward(self, inputs):
        h, w = inputs.shape[2], inputs.shape[3]
        feat0 = self.conv0(inputs) # (32, 208)

        # for k in feat0:
        #     show_tensorll(k.sum(dim=0))

        feat1 = self.down1(feat0)  # (64, 104)
        feat2 = self.down2(feat1)  # (128, 52)
        feat3 = self.down3(feat2)  # (252, 26)
        feat4 = self.down4(feat3)  # (261, 13)
        feat4 = self.supertrans111(feat4)  # (261, 13)
        feat4 = self.supertrans222(feat4)  # (261, 13)
        # t0, t1, t2, t3 = self.psc(feat0, feat1, feat2, feat3)
        # feat4_r = F.interpolate(feat4, size=(h, w), mode='bilinear', align_corners=False)

        up3 = self.up_concat3(feat3, feat4)#256
        up3 = self.dlformer2(up3)
        # up3 = self.dlformer22(up3)
        # up3 = self.dlformer23(up3)
        up2 = self.up_concat2(feat2, up3)#128
        up2 = self.dlformer1(up2)
        # up2 = self.dlformer12(up2)
        up1 = self.up_concat1(feat1, up2)
        up1 = self.dsam2(up1)
        up0 = self.up_concat0(feat0, up1)
        # for k in up0:
        #     show_tensorll(k.sum(dim=0))
        up0 = self.dsam1(up0)

        # for k in up0:
        #     show_tensorll(k.sum(dim=0))

        up0 = torch.cat([up0, inputs], 1)
        final = self.final(up0)
        prob0 = self.softmax(final)

        # for k in final:
        #     show_tensorll(k.sum(dim=0))

        lat = self.edge(up0)
        # edge2, _ = torch.max(prob0, dim=1, keepdim=True)
        # edge2_min = edge2.min()
        # edge2_max = edge2.max()
        # edge2 = (edge2 - edge2_min) / (edge2_max - edge2_min + 1e-8)
        # edge2 = 1 - edge2

        edge2, _ = torch.min(1-prob0, dim=1, keepdim=True)

        # show_tensorll(edge2[0])

        if self.training:
            return prob0, lat, edge2
        else:
            return prob0


    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.Localformer = LocalFormerBlock3(576)
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
        ##---------------------------------------------------
        self.skip_scale1 = nn.Parameter(torch.ones(1))
        self.skip_scale2 = nn.Parameter(torch.ones(1))
        self.mlp = Mlp(input_dim)
        self.randomconv = RandomConv(input_dim)
        #---------------------------------------------
        num_channels_reduced = input_dim // 16
        self.fc1 = nn.Linear(input_dim, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, input_dim, bias=True)
        self.relu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        self.norm33 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x = x.permute(0, 2, 3, 1)
        xx = x.flatten(2).permute(0, 2, 1)
        xx = self.norm33(xx)
        z = xx.clone().permute(0, 2, 1).mean(dim=2)
        # z_max = xx.permute(0, 2, 1).max(dim=2)

        fc_out_1 = self.relu(self.fc1(z))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        # --------------------------
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        H, W = x.shape[2:]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4, x5, x6, x7, x8, x9 = torch.chunk(x_norm, 9, dim=2)

        # x1test = self.mamba(x1)
        # x_reversed = x.transpose(-1, -2).reshape(B, C, n_tokens)

        x_mamba1 = self.Localformer(x1.transpose(-1, -2).reshape(B, C//9, H, W)) #+ self.skip_scale * x1
        x_mamba2 = self.Localformer(x2.transpose(-1, -2).reshape(B, C//9, H, W)) #+ self.skip_scale * x2
        x_mamba3 = self.Localformer(x3.transpose(-1, -2).reshape(B, C//9, H, W)) #+ self.skip_scale * x3
        x_mamba4 = self.Localformer(x4.transpose(-1, -2).reshape(B, C//9, H, W)) #+ self.skip_scale * x4
        x_mamba5 = self.Localformer(x5.transpose(-1, -2).reshape(B, C//9, H, W)) #+ self.skip_scale * x5
        x_mamba6 = self.Localformer(x6.transpose(-1, -2).reshape(B, C//9, H, W)) #+ self.skip_scale * x6
        x_mamba7 = self.Localformer(x7.transpose(-1, -2).reshape(B, C//9, H, W)) #+ self.skip_scale * x7
        x_mamba8 = self.Localformer(x8.transpose(-1, -2).reshape(B, C//9, H, W)) #+ self.skip_scale * x8
        x_mamba9 = self.Localformer(x9.transpose(-1, -2).reshape(B, C//9, H, W)) #+ self.skip_scale * x9

        x_mamba = torch.cat(
            [x_mamba1, x_mamba2, x_mamba3, x_mamba4, x_mamba5, x_mamba6, x_mamba7, x_mamba8, x_mamba9], dim=2)

        #------------------
        x_mamba = self.skip_scale1 * x + self.randomconv(x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)).permute(0, 3, 1, 2)
        x_mamba = x_mamba.reshape(B, C, n_tokens).transpose(-1, -2)
        x_mamba = x_mamba * fc_out_2.unsqueeze(1)
        #---------------------

        x_mamba = self.skip_scale2 * x_mamba + self.mlp(self.norm(x_mamba))
        # x_mamba = self.proj(x_mamba)
        #-----------

        #---------------
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out

def lformerr3(data=None):
    model = Unet()
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model

if __name__ == "__main__":
    # with torch.no_grad():
        input = torch.randn(4, 3, 208, 208)
        input = input.cuda()
        model = Unet().cuda()

        out_result = model(input)
        print(out_result.shape)