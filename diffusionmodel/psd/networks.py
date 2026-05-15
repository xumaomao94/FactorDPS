import torch.nn as nn
import torch
import math
from inspect import isfunction

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim # inner_channel; correspond to d_model in "Attention is All You Need" (dimension of position encoding)

    def forward(self, noise_level): # noise_level: [batch, 1] # using noise_level as position
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count # [0, 1, 2, ..., count-1] / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding # [batch, 1, self.dim=count*2]
    
class Upsample(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv1d(in_channel, in_channel, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(self.up(x))
    

class Downsample(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, in_channel, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, groups=32, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.block(x)
    

class FeatureWiseAffine(nn.Module):
    """Noise embed affine transformation to mix with the input features """
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels *(1+use_affine_level))
        )
    
    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(batch, -1, 1).chunk(2, dim=1)
            x = (1+gamma)*x + beta
        else:
            x = x+ self.noise_func(noise_embed).view(batch, -1, 1)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, noise_level_embed_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()

        # Process noise by noise_func and feature by block1 to result in the same shape of out_channels
        self.noise_func = FeatureWiseAffine(in_channels=noise_level_embed_dim,
                                            out_channels=out_channels,
                                            use_affine_level=use_affine_level)
        
        self.block1 = Block(in_channels=in_channels, out_channels=out_channels,groups=norm_groups)

        self.block2 = Block(in_channels=out_channels, out_channels=out_channels,groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv1d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity() # Use convolution to amtch the channel size for residual
    
    def forward(self, x, noise_embed):

        '''Process input signal to match the resulting size of noise embed inside FeatureWiseAffine for mixing with noise'''
        h = self.block1(x)
        h = self.noise_func(h, noise_embed)
        h = self.block2(h)

        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channels, n_heads=1, norm_groups=32):
        super().__init__()
        
        self.n_heads = n_heads
        self.fnorm = nn.GroupNorm(norm_groups, in_channels)
        self.qkv = nn.Conv1d(in_channels=in_channels,
                             out_channels=in_channels*3,
                             kernel_size=1,
                             bias=False)
        self.out = nn.Conv1d(in_channels=in_channels,out_channels=in_channels, kernel_size=1)

    def forward(self, input):
        b, c, k = input.shape
        head_dim = c//self.n_heads  # the qkv each will have channel dimension. divide the features among multiple heads

        normalized_input = self.fnorm(input)
        qkv = self.qkv(normalized_input).view(b, self.n_heads, head_dim*3 , k)
        query,key,value = qkv.chunk(3, dim=2) #[b, n_heads, head_dim, k]

        attn = torch.einsum("bndk,bndl->bnkl", query, key).contiguous()/math.sqrt(head_dim) # [b, n_heads, k, k] #Todo: using head_dim here
        attn = torch.softmax(attn, dim=-1)

        out_val = torch.einsum("bnkl,bndl->bndk", attn, value).contiguous().view(b, c, k)
        return input + self.out(out_val)

class ResNetBlockWithAttn(nn.Module):
    def __init__(self, in_channels, out_channels, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResNetBlock(in_channels=in_channels,
                                     out_channels=out_channels,
                                     noise_level_embed_dim=noise_level_emb_dim,
                                     dropout=dropout,
                                     norm_groups=norm_groups)
        if with_attn:
            self.attn = SelfAttention(in_channels=out_channels, norm_groups=norm_groups)
    
    def forward(self, x, noise_embed):
        h = self.res_block(x, noise_embed)
        if self.with_attn:
            h = self.attn(h)
        return h



class PSDNet(nn.Module):
    def __init__(self,
                 K=64,
                 in_channel = 1,
                 out_channel = 1,
                 channel_mults=(1,2,4,8,8),
                inner_channel=32,
                attention_resolutions=(16,8,4,2,1),
                res_blocks=2,
                with_noise_level_emb=True,
                dropout=0.0,
                norm_groups=32,
                ):
        super().__init__()

        if with_noise_level_emb:
            noise_embed_dim = inner_channel
            self.noise_level_mlp = nn.Sequential(PositionalEncoding(inner_channel),
                                                 nn.Linear(inner_channel, inner_channel*4),
                                                 nn.SiLU(),
                                                 nn.Linear(inner_channel*4, inner_channel),
                                                 )
        else:
            noise_embed_dim=None
            self.noise_level_mlp = None
        
        num_mults = len(channel_mults)

        downs =[nn.Conv1d(in_channel, inner_channel, kernel_size=3, padding=1)]
        current_K = K
        feat_channels = [inner_channel]

        for ind in range(num_mults):
            is_last_layer = (ind == (num_mults-1))
            use_attention = (current_K in attention_resolutions)

            in_channel_num = feat_channels[-1]
            out_channel_num = inner_channel * channel_mults[ind]

            '''Attach ResNet blocks'''
            for _ in range(0, res_blocks): # Same K but different channel_nums
                downs.append(ResNetBlockWithAttn(in_channel_num, out_channel_num,
                                                 noise_level_emb_dim = noise_embed_dim,
                                                 norm_groups=norm_groups,
                                                 dropout=dropout,
                                                 with_attn=use_attention,
                                                 ))
                feat_channels.append(out_channel_num)
                in_channel_num = feat_channels[-1]
            if not is_last_layer: # Half K but same channel nums
                downs.append(Downsample(in_channel_num))
                feat_channels.append(in_channel_num)
                current_K = current_K // 2
        self.downs = nn.ModuleList(downs)


        self.mid = nn.ModuleList([
            ResNetBlockWithAttn(in_channel_num, in_channel_num,
                                noise_level_emb_dim = noise_embed_dim,
                                norm_groups=norm_groups,
                                dropout=dropout,
                                with_attn=True),
            ResNetBlockWithAttn(in_channel_num, in_channel_num,
                                noise_level_emb_dim = noise_embed_dim,
                                norm_groups=norm_groups,
                                dropout=dropout,
                                with_attn=False)

        ])

        ups = []
         # Due to skip connection from last layer of downs
        for ind in reversed(range(num_mults)):
            is_last_layer = (ind == 0)
            use_attention = (current_K in attention_resolutions)
            out_channel_num = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1): # +1 for skip connection from the downsample layer too # No downsample layer for the first layer
                skip_channel_num = feat_channels.pop()
                # twice of in_channel)_num due to the skip connection
                ups.append(ResNetBlockWithAttn(in_channel_num+skip_channel_num, out_channel_num,
                                               noise_level_emb_dim = noise_embed_dim,
                                               norm_groups=norm_groups,
                                               dropout=dropout,
                                               with_attn=use_attention))
                in_channel_num = out_channel_num
            if not is_last_layer:
                ups.append(Upsample(out_channel_num))
                current_K = current_K * 2
        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(out_channel_num, default(out_channel, in_channel), groups=norm_groups)


    def forward(self, x, noise_level):
        t = self.noise_level_mlp(noise_level) if exists(self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResNetBlockWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)
        
        for layer in self.mid:
            if isinstance(layer, ResNetBlockWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
        for layer in self.ups:
            if isinstance(layer, ResNetBlockWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)



