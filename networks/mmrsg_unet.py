
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np


try:
    from .csgat_utils import LGAG, CAB, SAB
except ImportError:
    from csgat_utils import LGAG, CAB, SAB


try:
    from .MS_SSM import MS_SSM
except ImportError:
    try:
        from MS_SSM import MS_SSM
    except ImportError as e:
        raise ImportError("Failed to import MSCSM module. Please ensure networks/MSCSM.py exists.") from e


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size, dim_out=None, num_heads=9, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        

        if isinstance(resolution, (tuple, list)):
            res_h, res_w = resolution
        else:
            res_h = res_w = resolution
            
        if idx == -1:
            H_sp, W_sp = res_h, res_w
        elif idx == 0:
            H_sp, W_sp = res_h, self.split_size
        elif idx == 1:
            W_sp, H_sp = res_w, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        qkv: tuple of (q, k, v) where each is B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        B, L, C = q.shape
        H = W = int(np.sqrt(L))  
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x


class CSWinBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 2, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x

class CARAFE(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, up_factor=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(dim, dim // 4, 1)
        self.encoder = nn.Conv2d(dim // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(dim, dim_out, 1)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)


            # N,C,H,W -> N,C,delta*H,delta*W
            # kernel prediction module
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor,
                                        self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(B, self.kernel_size ** 2, H, W,
                                                  self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

            # content-aware reassembly module
            # tensor.unfold: dim, size, step
        w = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                              self.kernel_size // 2, self.kernel_size // 2),
                              mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        w = w.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        w = w.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        w = w.reshape(B, C, H, W, -1)  # (N, C, H, W, Kup^2)
        w = w.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        x = torch.matmul(w, kernel_tensor)  # (N, H, W, C, S^2)
        x = x.reshape(B, H, W, -1)
        x = x.permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, self.up_factor)
        x = self.out(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()

        return x


class CARAFE4(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, up_factor=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(dim, dim // 4, 1)
        self.encoder = nn.Conv2d(dim // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(dim, dim_out, 1)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)


            # N,C,H,W -> N,C,delta*H,delta*W
            # kernel prediction module
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor,
                                        self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(B, self.kernel_size ** 2, H, W,
                                                  self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

            # content-aware reassembly module
            # tensor.unfold: dim, size, step
        w = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                              self.kernel_size // 2, self.kernel_size // 2),
                              mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        w = w.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        w = w.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        w = w.reshape(B, C, H, W, -1)  # (N, C, H, W, Kup^2)
        w = w.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        x = torch.matmul(w, kernel_tensor)  # (N, H, W, C, S^2)
        x = x.reshape(B, H, W, -1)
        x = x.permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, self.up_factor)
        x = self.out(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()

        return x


class CSGAT(nn.Module):
    def __init__(self, dim_main, dim_skip, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.dim_main = dim_main
        self.dim_skip = dim_skip
        self.cab = CAB(in_channels=dim_skip)
        self.sab = SAB()
        F_int = min(dim_main, dim_skip) // 2
        self.lgag = LGAG(F_g=dim_skip, F_l=dim_main, F_int=F_int)
        if dim_skip != dim_main:
            self.skip_adjust = nn.Conv2d(dim_skip, dim_main, 1, bias=False)
        else:
            self.skip_adjust = nn.Identity()
        self.concat_linear = nn.Linear(2 * dim_main, dim_main)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, F_carafe_out, F_skip_raw):
        if self.enabled:
            channel_attn = self.cab(F_skip_raw)
            spatial_attn = self.sab(F_skip_raw * channel_attn)
            F_skip_refined = F_skip_raw * channel_attn * spatial_attn
            F_gated = self.lgag(F_skip_refined, F_carafe_out)
            additive_feature = F_carafe_out + (self.gamma * F_gated)
            F_skip_adjusted = self.skip_adjust(F_skip_refined)
        else:
            additive_feature = F_carafe_out
            F_skip_adjusted = self.skip_adjust(F_skip_raw)
        B, C_main, H, W = additive_feature.shape
        L = H * W
        F_skip_flat = F_skip_adjusted.view(B, C_main, L).transpose(1, 2)
        additive_flat = additive_feature.view(B, C_main, L).transpose(1, 2)
        concat_feature = torch.cat([F_skip_flat, additive_flat], dim=2)
        output_flat = self.concat_linear(concat_feature)
        output = output_flat.transpose(1, 2).view(B, self.dim_main, H, W)
        return output



class MambaBottleneckWrapper(nn.Module):

    def __init__(self, dim, reso, norm_layer=nn.LayerNorm, drop_path=0., d_state=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.H = self.W = reso  
        self.norm = norm_layer(dim)


        self.mamba_block = MS_SSM(
            hidden_dim=dim,
            d_state=d_state,
            **kwargs
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):

        x_residual = x

        x = self.norm(x)

        B, L, C = x.shape
        assert L == self.H * self.W, "Input tensor length L does not match H*W"
        x_4d = x.reshape(B, self.H, self.W, C)

        x_mamba_out = self.mamba_block(x_4d) 


        x_flat_out = x_mamba_out.reshape(B, L, C)


        x = x_residual + self.drop_path(x_flat_out)

 
        return x




class CSWinTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=8, embed_dim=64, depth=[1, 2, 9, 1],
                 split_size=[1, 2, 7, 7],
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0, hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False):
        super().__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        
       
        if isinstance(num_heads, int):
            # stage1: dim=64, stage2: dim=128, stage3: dim=256, stage4: dim=512
            heads = [2, 4, 8, 16] 
        else:
            heads = num_heads

        #encoder

        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 7, 4, 2),
            Rearrange('b c h w -> b (h w) c', h=img_size // 4, w=img_size // 4),
            nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
        print("depth",depth)
        self.stage1 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[0], reso=img_size // 4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.merge1 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.stage2 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size // 8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer)
                for i in range(depth[1])])
        self.merge2 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        temp_stage3 = []
        temp_stage3.extend(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], reso=img_size // 16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer)
                for i in range(depth[2])])

        self.stage3 = nn.ModuleList(temp_stage3)
        self.merge3 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2

        self.stage4 = nn.ModuleList(
            [MambaBottleneckWrapper(
                dim=curr_dim,
                reso=img_size // 32,  # 224 // 32 = 7
                norm_layer=norm_layer,
                drop_path=dpr[np.sum(depth[:-1]) + i],
                d_state=16
            ) for i in range(depth[-1])])


        self.norm = norm_layer(curr_dim)




  
        self.stage_up4 = nn.ModuleList(
            [MambaBottleneckWrapper(
                dim=curr_dim,
                reso=img_size // 32,
                norm_layer=norm_layer,
                drop_path=dpr[np.sum(depth[:-1]) + i],
                d_state=16
            ) for i in range(depth[-1])])


        self.carafe4 = CARAFE(curr_dim, curr_dim // 2)
        self.csgat4 = CSGAT(dim_main=curr_dim // 2, dim_skip=curr_dim // 2)
        curr_dim = curr_dim // 2

        self.stage_up3 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], reso=img_size // 16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer)
                for i in range(depth[2])]
        )

        self.carafe3 = CARAFE(curr_dim, curr_dim // 2)
        self.csgat3 = CSGAT(dim_main=curr_dim // 2, dim_skip=curr_dim // 2)
        curr_dim = curr_dim // 2

        self.stage_up2 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size // 8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer)
                for i in range(depth[1])])
                
        
        self.carafe2 = CARAFE(curr_dim, curr_dim // 2)
        self.csgat2 = CSGAT(dim_main=curr_dim // 2, dim_skip=curr_dim // 2)
        curr_dim = curr_dim // 2

        self.concat_linear2 = nn.Linear(128, 64)
        self.stage_up1 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[0], reso=img_size // 4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])

        self.upsample1 = CARAFE4(curr_dim, 64)
        self.norm_up = norm_layer(embed_dim)
        self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.out_head2 = nn.Conv2d(128, self.num_classes, kernel_size=1, bias=False)  
        self.out_head3 = nn.Conv2d(256, self.num_classes, kernel_size=1, bias=False)  
        self.out_head4 = nn.Conv2d(512, self.num_classes, kernel_size=1, bias=False)  
        self.latest_feature_maps = {}
        # Classifier head

        

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    #Encoder and Bottleneck
    def forward_features(self, x):
        x = self.stage1_conv_embed(x)

        x = self.pos_drop(x)

        for blk in self.stage1:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        self.x1 = x
        x = self.merge1(x)

        for blk in self.stage2:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        self.x2 = x
        x = self.merge2(x)

        for blk in self.stage3:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                    x = blk(x)
        self.x3 = x
        x = self.merge3(x)

        for blk in self.stage4:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.norm(x)

        return x

    def forward_up_features(self, x):
        # ---- STAGE 4 ----
        for blk in self.stage_up4:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x4_feat = x  # (B, HW, 512)

        B_skip, HW_skip, C_skip = self.x3.shape
        H_skip = W_skip = int(np.sqrt(HW_skip))
        x3_converted = self.x3.reshape(B_skip, H_skip, W_skip, C_skip).permute(0, 3, 1, 2)

        F_carafe_out_flat = self.carafe4(x)
        B, HW_up, C_out = F_carafe_out_flat.shape
        H_up = W_up = int(np.sqrt(HW_up))
        F_carafe_out_bchw = F_carafe_out_flat.transpose(1, 2).view(B, C_out, H_up, W_up)
        x = self.csgat4(F_carafe_out_bchw, x3_converted)

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # ---- STAGE 3 ----
        for blk in self.stage_up3:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x3_feat = x  # (B, HW, 256)

        B_skip, HW_skip, C_skip = self.x2.shape
        H_skip = W_skip = int(np.sqrt(HW_skip))
        x2_converted = self.x2.reshape(B_skip, H_skip, W_skip, C_skip).permute(0, 3, 1, 2)

        F_carafe_out_flat = self.carafe3(x)
        B, HW_up, C_out = F_carafe_out_flat.shape
        H_up = W_up = int(np.sqrt(HW_up))
        F_carafe_out_bchw = F_carafe_out_flat.transpose(1, 2).view(B, C_out, H_up, W_up)
        x = self.csgat3(F_carafe_out_bchw, x2_converted)

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # ---- STAGE 2 ----
        for blk in self.stage_up2:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x2_feat = x  # (B, HW, 128)

        B_skip, HW_skip, C_skip = self.x1.shape
        H_skip = W_skip = int(np.sqrt(HW_skip))
        x1_converted = self.x1.reshape(B_skip, H_skip, W_skip, C_skip).permute(0, 3, 1, 2)

        F_carafe_out_flat = self.carafe2(x)
        B, HW_up, C_out = F_carafe_out_flat.shape
        H_up = W_up = int(np.sqrt(HW_up))
        F_carafe_out_bchw = F_carafe_out_flat.transpose(1, 2).view(B, C_out, H_up, W_up)
        x = self.csgat2(F_carafe_out_bchw, x1_converted)

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # ---- STAGE 1 ----
        for blk in self.stage_up1:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.norm_up(x)  # B, HW, 64
        x1_feat = x  # (B, HW, 64)


        return x1_feat, x2_feat, x3_feat, x4_feat

    def up_x4(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = self.upsample1(x)
        x = x.view(B, 4 * H, 4 * W, -1)
        x = x.permute(0, 3, 1, 2)  # B,C,H,W
        x = self.output(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)

        x1_feat, x2_feat, x3_feat, x4_feat = self.forward_up_features(x)


        def to_bchw(feat_flat):
            B, L, C = feat_flat.shape
            H = W = int(np.sqrt(L))
            return feat_flat.transpose(1, 2).view(B, C, H, W)


        p1 = self.up_x4(x1_feat)


        p2 = self.out_head2(to_bchw(x2_feat))
        p3 = self.out_head3(to_bchw(x3_feat))
        p4 = self.out_head4(to_bchw(x4_feat))
        self.latest_feature_maps = {
            "p1": p1.detach(),
            "p2": p2.detach(),
            "p3": p3.detach(),
            "p4": p4.detach(),
        }


        target_hw = p1.shape[2:]
        p2 = F.interpolate(p2, size=target_hw, mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, size=target_hw, mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=target_hw, mode='bilinear', align_corners=False)

        return [p1, p2, p3, p4]


