import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial
#该变量的值是一个列表，存储的是当前模块中一些成员（变量、函数或者类）的名称。
#通过在模块文件中设置 __all__ 变量，当其它文件以“from 模块名 import *”的形式导入该模块时，该文件中只能使用 __all__ 列表中指定的成员
__all__ = ['spikformer']


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        #定义各层参数，详细见forward
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        #全连接相当于卷积的卷积核为1
        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T,B,N,C = x.shape
        # torch.Size([4, 64, 64, 384])
        x_ = x.flatten(0, 1)
        # torch.Size([256, 64, 384])
        x = self.fc1_linear(x_)
        # 1536是隐藏维度为 4*C
        # torch.Size([256, 64, 1536])
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous()
        # torch.Size([4, 64, 64, 1536])
        x = self.fc1_lif(x)
        # torch.Size([4, 64, 64, 1536])

        x = self.fc2_linear(x.flatten(0,1))
        # torch.Size([256, 64, 384])
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        # torch.Size([4, 64, 64, 384])
        x = self.fc2_lif(x)
        # torch.Size([4, 64, 64, 384])
        return x


class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        #定义各层参数，详细见forward
        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        #这个C就是特征维度D
        T,B,N,C = x.shape

        #以Cifar10的B=64,T=4,D=384,num_heads=12运算为例
        x_for_qkv = x.flatten(0, 1)  # TB, N, C
        # torch.Size([256, 64, 384])

        #计算Query矩阵
        q_linear_out = self.q_linear(x_for_qkv)
        # [TB, N, C]
        # torch.Size([256, 64, 384])
        q_linear_out = self.q_bn(q_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        # torch.Size([4, 64, 64, 384])
        q_linear_out = self.q_lif(q_linear_out)
        # torch.Size([4, 64, 64, 384])
        # permute函数的作用是对tensor进行转置，这里多一个维度是采用多头自注意力机制
        q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        # torch.Size([4, 64, 12, 64, 64])

        # 计算Key矩阵
        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        # 计算Value矩阵
        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        #attn为Q*K^T
        #q为torch.Size([4, 64, 12, 64, 64])，k.transpose(-2, -1)为torch.Size([4, 64, 12, 64, 64])
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # torch.Size([4, 64, 12, 64, 64])
        x = attn @ v
        # torch.Size([4, 64, 12, 64, 32])
        # 多头合并
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        # torch.Size([4, 64, 64, 384])
        x = self.attn_lif(x)
        # torch.Size([4, 64, 64, 384])
        x = x.flatten(0, 1)
        # torch.Size([256, 64, 384])
        x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C))
        # torch.Size([4, 64, 64, 384])
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        #对单个batch归一化,似乎没有用上
        self.norm1 = norm_layer(dim)
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        #imagenet中在这里还有一个DropPath层，是为了防止过拟合
        #drop_path 将深度学习模型中的多分支结构随机 “失效”，而 drop_out 是对神经元随机 “失效”。
        # 换句话说，drop_out 是随机的点对点路径的关闭，drop_path 是随机的点对层之间的关闭
        self.norm2 = norm_layer(dim)
        #计算mlp隐藏维度,按照论文中为4*D
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        #单个Encoder块，经过SSA自注意力和MLP多功能感知两个过程
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        #后续用于将图像分割为patch_size*patch_size的小块
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        # //是向下取整的运算
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        #四个SPS块，4块的卷积层的输出通道数分别是D/8、D/4、D/2、D，cifar10测试时我们采用的embed_dims为384
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims//4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims//2)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        #相对位置嵌入
        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        # cifar10测试时为:torch.Size([4, 64, 3, 32, 32])
        # 这里的H,W是一个图片的宽高
        T, B, C, H, W = x.shape
        #flatten(0, 1)在第一维和第二维之间平坦化
        #x.flatten(0, 1)就是若干个C*H*W的矩阵,cifar10下为:torch.Size([256, 3, 32, 32])

        #注：这里前两个SPS没有池化层,但是imagenet的模型里每个SPS都有池化层
        #池化应该就是为了使最终的N不会太大，因为imagenet本身图片比较大所以才需要多池化几次
        #以cifar为例记录维度变化过程
        # torch.Size([4, 64, 3, 32, 32])
        x = self.proj_conv(x.flatten(0, 1))
        # torch.Size([256, 48, 32, 32])
        # 这个48是384/8
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        # torch.Size([4, 64, 48, 32, 32])
        # .reshape(T, B, -1, H, W)是为了传递给神经元
        x = self.proj_lif(x).flatten(0, 1).contiguous()
        # torch.Size([256, 48, 32, 32])

        x = self.proj_conv1(x)
        # torch.Size([256, 96, 32, 32])
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
        # torch.Size([4, 64, 96, 32, 32])
        x = self.proj_lif1(x).flatten(0, 1).contiguous()
        # torch.Size([256, 96, 32, 32])

        x = self.proj_conv2(x)
        # torch.Size([256, 192, 32, 32])
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous()
        # torch.Size([4, 64, 192, 32, 32])
        x = self.proj_lif2(x).flatten(0, 1).contiguous()
        # torch.Size([256, 192, 32, 32])
        x = self.maxpool2(x)
        # torch.Size([256, 192, 16, 16])

        x = self.proj_conv3(x)
        # torch.Size([256, 384, 16, 16])
        x = self.proj_bn3(x).reshape(T, B, -1, H//2, W//2).contiguous()
        # torch.Size([4, 64, 384, 16, 16])
        x = self.proj_lif3(x).flatten(0, 1).contiguous()
        # torch.Size([256, 384, 16, 16])
        x = self.maxpool3(x)
        # torch.Size([256, 384, 8, 8])

        x_feat = x.reshape(T, B, -1, H//4, W//4).contiguous()
        # torch.Size([256, 384, 8, 8])
        x = self.rpe_conv(x)
        # torch.Size([256, 384, 8, 8])
        x = self.rpe_bn(x).reshape(T, B, -1, H//4, W//4).contiguous()
        # torch.Size([4, 64, 384, 8, 8])
        x = self.rpe_lif(x)
        # torch.Size([4, 64, 384, 8, 8])
        x = x + x_feat
        # torch.Size([4, 64, 384, 8, 8])
        # T,B,N,C(这里的C就是特征维度D)
        x = x.flatten(-2).transpose(-1, -2)
        # 8,8合并为64，然后64混合384交换位置
        # torch.Size([4, 64, 64, 384])
        # imagenet中就没有这一行，保留了H和W最后两个分量，imagenet中在SSA中再合并
        return x


class Spikformer(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T = 4
                 ):

        #调用nn.Module的初始化
        super().__init__()
        self.T = T  # time step
        self.num_classes = num_classes
        self.depths = depths

        # 采用随机深度衰减法则 输出一个在0-drop_path_rate间有depths个数均匀分布的数组
        # 在训练时使用较浅的深度(例如随机在resnet的基础上pass掉一些层)，在测试时使用较深的深度，较少训练时间，提高训练性能
        # 优点：1、成功地解决了深度网络的训练时间难题。2、它大大减少了训练时间，并显着改善了几乎所有数据集的测试错误 。3、可以使得网络更深
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        #patch_embed为一个Spiking Patch Spliting模块
        patch_embed = SPS(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims)
        #block为L个Spikformer Encoder Block模块的串联
        block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        #为模块赋值patch_embed和block属性
        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        # nn.Identity()是恒等变换函数
        # 定义最后分类的全连接层
        # embed_dims是特征的维度
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        #模型赋值初始参数
        self.apply(self._init_weights)

    #这个装饰器向编译器表明，一个函数或方法应该被忽略，并保留为Python函数。这允许您在模型中保留尚未与TorchScript兼容的代码。
    #如果需要在nn.Module中排除某些方法，因为它们使用了TorchScript尚不支持的Python功能，则可以使用@torch.jit.ignore对其进行注释
    #问：试了一下在我电脑上把下面这一整段注释掉也可以用，所以不知道是做什么用的
    @torch.jit.ignore
    #获取位置嵌入
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            #permute(dims)如permute(0,1,2)，permute(1,0,2)进行维度交换
            return F.interpolate(pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            #timm.models.layers.trunc_normal_
            # 将参数初始化为截断正态分布
            trunc_normal_(m.weight, std=.02)
            #初始化全连接层偏置为0
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            #初始化偏置为1
            nn.init.constant_(m.bias, 0)
            #初始化连接权重为1
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        #运行SPS层
        x = patch_embed(x)
        #运行L个Spikformer Encoder Block层
        for blk in block:
            x = blk(x)
        # T*B*N*D
        # torch.Size([4, 64, 64, 384])
        #按照第三个维度取平均值
        return x.mean(2)

    def forward(self, x):
        #按照第一个维度升维
        #unsqueeze()函数的功能是在tensor的某个维度上添加一个维数为1的维度
        # 而一个样本的维度为[b, c, w, h]，此时用unsqueeze(0)增加一个维度变为[1, b, c, w, h]就很方便了。
        # repeat()函数可以对张量进行重复扩充
        # cifar10测试时为: torch.Size([64, 3, 32, 32])由变为torch.Size([4, 64, 3, 32, 32])
        # 64是batch_size,3是channel,32*32是图像的宽高
        # torch.Size([64, 3, 32, 32])
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        # torch.Size([4, 64, 3, 32, 32])
        #经过SPS、多个SSA和MLP提取特征
        x = self.forward_features(x)

        # torch.Size([4, 64, 384])
        #进行全连接分类
        # x.mean(0).shape:torch.Size([64, 384])
        x = self.head(x.mean(0))
        # torch.Size([64, 10])
        return x

#利用@register_model装饰器，注册这个新定义的模型，存储到_model_entrypoints这个字典中
#内部关键语句是_model_entrypoints[model_name] = fn
@register_model
def spikformer(pretrained=False, **kwargs):
    #创建上方的定义的Spikformer模型
    model = Spikformer(
        # img_size_h=224, img_size_w=224,
        # patch_size=16, embed_dims=768, num_heads=12, mlp_ratios=4,
        # in_channels=3, num_classes=1000, qkv_bias=False,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=12, sr_ratios=1,
        **kwargs
    )
    #定义模型参数包括num_classes、input_size、pool_size、crop_pct、interpolation等参数
    model.default_cfg = _cfg()
    return model


