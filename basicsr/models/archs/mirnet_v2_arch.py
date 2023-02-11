
## Learning Enriched Features for Fast Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
## https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/
#%%
# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as stx
import math

##########################################################################
##---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3,reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_V        



class ContextBlock(nn.Module):

    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)
        # channel_add_conv = tf.keras.Sequential([tf.keras.layers.Conv2D(80, 1, 1, input_shape=[1,1,80]),tf.keras.layers.LeakyReLU(alpha=0.2) ,tf.keras.layers.Conv2D(80, 1, 1)]) ->
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )

    def modeling(self, x):
        # x -> [N, C, H, W
        batch, channel, height, width = x.size() # tf.shape(x) -> [  2,  80, 256, 256]
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width) # input_x = tf.reshape(input_x, [batch, channel, height*width]) -> [    2,    80, 65536]
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1) # input_x = tf.expand_dims(input_x, 1) -> [2, 1, 80, 65536]
        # [N, 1, H, W]
        context_mask = self.conv_mask(x) # tf.keras.layers.Conv2D(1, 1)(tf.transpose(x, [0, 2, 3, 1])) -> [2, 256, 256, 1]
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width) # tf.reshape(context_mask, [batch, height * width, 1])
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask) # tf.keras.layers.Softmax(axis=1)(context_mask) -> [2, 65536, 1]
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3) # tf.expand_dims(context_mask, 1) -> [2, 1, 65536, 1]
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask) # tf.linalg.matmul(input_x, context_mask) -> [2, 1, 80, 1]
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1) # tf.reshape(context, [batch, channel, 1, 1]) -> [2, 80, 1, 1]

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        # import pdb; pdb.set_trace()
        context = self.modeling(x)
        # context = tf.transpose(context, [0, 2,3,1]) -> [2, 1, 1, 80]
        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context) # channel_add_term = channel_add_conv(context) -> [2, 1, 1, 80]
        # x = tf.transpose(x, [0, 2,3,1]) -> [2, 256, 256, 80]
        x = x + channel_add_term # same -> [2, 256, 256, 80]

        return x

##########################################################################
### --------- Residual Context Block (RCB) ----------
class RCB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCB, self).__init__()
        
        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act, 
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups)
        )

        self.act = act
        
        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x):
        # x -> [N, H, W, C]
        import pdb; pdb.set_trace()
        # feat = tf.shape(x_)[-1]
        # body = tf.keras.Sequential([tf.keras.layers.Conv2D(feat, 3, 1, padding='same', input_shape=tf.shape(x_)[1:]), tf.keras.layers.LeakyReLU(0.2) ,tf.keras.layers.Conv2D(feat, 3, 1, padding='same')]) -> same shape as x
        res = self.body(x)
        res = self.act(self.gcnet(res))
        res += x
        return res


##########################################################################
##---------- Resizing Modules ----------    
class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels*chan_factor), 1, stride=1, padding=0, bias=bias)
            )

    def forward(self, x):
        return self.bot(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels//chan_factor), 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
            )

    def forward(self, x):
        return self.bot(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


##########################################################################
##---------- Multi-Scale Resiudal Block (MRB) ----------
class MRB(nn.Module):
    def __init__(self, n_feat, height, width, chan_factor, bias,groups):
        super(MRB, self).__init__()

        self.n_feat, self.height, self.width = n_feat, height, width

        self.dau_top = RCB(int(n_feat*chan_factor**0), bias=bias, groups=groups)
        self.dau_mid = RCB(int(n_feat*chan_factor**1), bias=bias, groups=groups)
        self.dau_bot = RCB(int(n_feat*chan_factor**2), bias=bias, groups=groups)

        self.down2 = DownSample(int((chan_factor**0)*n_feat),2,chan_factor)
        self.down4 = nn.Sequential(
            DownSample(int((chan_factor**0)*n_feat),2,chan_factor), 
            DownSample(int((chan_factor**1)*n_feat),2,chan_factor)
        )

        self.up21_1 = UpSample(int((chan_factor**1)*n_feat),2,chan_factor)
        self.up21_2 = UpSample(int((chan_factor**1)*n_feat),2,chan_factor)
        self.up32_1 = UpSample(int((chan_factor**2)*n_feat),2,chan_factor)
        self.up32_2 = UpSample(int((chan_factor**2)*n_feat),2,chan_factor)

        self.conv_out = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, bias=bias)

        # only two inputs for SKFF
        self.skff_top = SKFF(int(n_feat*chan_factor**0), 2)
        self.skff_mid = SKFF(int(n_feat*chan_factor**1), 2)

    def forward(self, x):
        x_top = x.clone()
        x_mid = self.down2(x_top)
        x_bot = self.down4(x_top)

        x_top = self.dau_top(x_top)
        x_mid = self.dau_mid(x_mid)
        x_bot = self.dau_bot(x_bot)

        x_mid = self.skff_mid([x_mid, self.up32_1(x_bot)])
        x_top = self.skff_top([x_top, self.up21_1(x_mid)])

        x_top = self.dau_top(x_top)
        x_mid = self.dau_mid(x_mid)
        x_bot = self.dau_bot(x_bot)

        x_mid = self.skff_mid([x_mid, self.up32_2(x_bot)])
        x_top = self.skff_top([x_top, self.up21_2(x_mid)])

        out = self.conv_out(x_top)
        out = out + x

        return out

##########################################################################
##---------- Recursive Residual Group (RRG) ----------
class RRG(nn.Module):
    def __init__(self, n_feat, n_MRB, height, width, chan_factor, bias=False, groups=1):
        super(RRG, self).__init__()
        modules_body = [MRB(n_feat, height, width, chan_factor, bias, groups) for _ in range(n_MRB)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
##---------- MIRNet  -----------------------

# ============================================================
#                         ☢ called by basicsr/models/archs/__init__.py L35
# ============================================================
class MIRNet_v2(nn.Module):
    """
    🔥
    network_g:
          type: MIRNet_v2
          inp_channels: 6
          out_channels: 3
          n_feat: 80
          chan_factor: 1.5
          n_RRG: 4
          n_MRB: 2
          height: 3
          width: 2
          scale: 1
          task: 'defocus_deblurring'
    """
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        n_feat=80,
        chan_factor=1.5,
        n_RRG=4,
        n_MRB=2,
        height=3,
        width=2,
        scale=1,
        bias=False,
        task= None
    ):
        super(MIRNet_v2, self).__init__()

        kernel_size=3
        self.task = task
        # n_feat ➡ output_channels
        self.conv_in = nn.Conv2d(inp_channels, n_feat, kernel_size=3, padding=1, bias=bias)

        modules_body = []
        
        modules_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=1))
        modules_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=2))
        modules_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=4))
        modules_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=4))

        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=bias)
        

    def forward(self, inp_img):
        # import pdb; pdb.set_trace()
        shallow_feats = self.conv_in(inp_img)
        # (Pdb) shallow_feats.size()
        # torch.Size([4, 80, 512, 512])
        deep_feats = self.body(shallow_feats)
        # (Pdb) deep_feats.size()
        # torch.Size([4, 80, 512, 512])
        if self.task == 'defocus_deblurring':
            deep_feats += shallow_feats
            out_img = self.conv_out(deep_feats)

        else:
            out_img = self.conv_out(deep_feats)
            out_img += inp_img

        return out_img

#%%
import yaml
import json
with open(r"T:\deblur_tf2_mirnetv2\MIRNetv2\Defocus_Deblurring\Options\DefocusDeblur_DualPixel_16bit_MIRNet_v2.yml", "r") as f:
    params = yaml.safe_load(f)
#%%
model_params = params['network_g']
model_params['inp_channels'] = 3
# remove type from model dictionary
model_params.pop('type')
print(json.dumps(model_params, indent=4))
#%%
sample_input_batch = torch.randn([2, 3, 256, 256])
model = MIRNet_v2(**model_params)
output = model(sample_input_batch)