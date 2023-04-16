#!/usr/bin/env python3
# This is a slightly modified version of timm's training script
""" Spikformer ImageNet Testing Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""


import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from spikingjelly.clock_driven import functional

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import create_dataset, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from loader import create_loader
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
import model #不是没用，在model.py中要注册spikformer类型,不然就会保存RuntimeError: Unknown model (spikformer)
try:
    #apex是一款基于 PyTorch 的混合精度训练加速神器
    #混合精度是指训练时在模型中同时使用 16位和 32位浮点类型，从而加快运行速度，减少内存使用的一种训练方法。
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    #torch.cuda.amp 给用户提供了较为方便的混合精度训练机制
    #autocast可以作为 Python上下文管理器和装饰器来使用，给算子自动安排按照FP16或者FP32的数值精度来操作
    #GradScaler的工作是在反向传播前给loss乘一个scale factor，所以之后反向传播得到的梯度都乘了相同的scale factor。并且为了不影响学习率，在梯度更新前将梯度unscale。
    #参考https://zhuanlan.zhihu.com/p/348554267
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True

except AttributeError:
    pass

try:
    #wandb用来帮助我们跟踪机器学习的项目，通过wandb可以记录模型训练过程中指标的变化情况以及超参的设置
    #还能够将输出的结果进行可视化的比对，帮助我们更好的分析模型在训练过程中的问题，同时我们还可以通过它来进行团队协作
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

#设置该值为True后，PyTorch中对模型里的卷积层进行预先的优化，也就是在每一个卷积层中测试cuDNN提供的所有卷积实现算法，然后选择最快的那个。
#这样在模型启动的时候，只要额外多花一点点预处理时间，就可以较大幅度地减少训练时间。
#对于网络的结构一般是不会动态变化的，图像一般都 resize 到固定的尺寸，batch size 也是固定的情况下，我们都可以在程序中加上这行神奇的代码，来减少运行时间
torch.backends.cudnn.benchmark = True

#logging模块是Python内置的标准模块，主要用于输出运行日志，可以设置输出日志的等级、日志保存路径、日志文件回滚等, 记录运行时的过程
_logger = logging.getLogger('train')#创建logger，默认为root

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below

#创建一个解析对象
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)

parser.add_argument('-c', '--config', default='cifar10.yml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
#parser与config_parser不同

# Model detail 模型细节
parser.add_argument('--model', default='spikformer', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('-T', '--time-step', type=int, default=4, metavar='time',
                    help='simulation time step of spiking neuron (default: 4)')
parser.add_argument('-L', '--layer', type=int, default=4, metavar='layer',
                    help='model layer (default: 4)')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--dim', type=int, default=None, metavar='N',
                    help='embedding dimsension of feature')
parser.add_argument('--num_heads', type=int, default=None, metavar='N',
                    help='attention head number')
parser.add_argument('--patch-size', type=int, default=None, metavar='N',
                    help='Image patch size')
parser.add_argument('--mlp-ratio', type=int, default=None, metavar='N',
                    help='expand ration of embedding dimension in MLP block')

# Dataset / Model parameters 数据集/模型参数
parser.add_argument('-data-dir', metavar='DIR',default="/home/zhou/Compact-Transformers-main/cifar-10-python/",
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='torch/cifar10',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')

parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')

parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--val-batch-size', type=int, default=16, metavar='N',
                    help='input val batch size for training (default: 32)')
# parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
#                     help='ratio of validation batch size to training batch size (default: 1)')

# Optimizer parameters 优化器参数
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Learning rate schedule parameters 学习率计划参数
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters 增强和正则化参数
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[1.0, 1.0], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently) 批标准化参数
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average 模型指数移动平均？
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc 其他
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')


def _parse_args():
    # Do we have a config file to parse?
    # parse_known_args方法的作用就是当仅获取到基本设置时，如果运行命令中传入了之后才会获取到的其他配置，不会报错；而是将多出来的部分保存起来，留到后面使用
    args_config, remaining = config_parser.parse_known_args()
    #如果有可用的文件，这里默认是”cifar10.yml“，就读入来用
    if args_config.config:
        with open(args_config.config, 'r') as f:
            #解析基本的yaml标记，得到名为cfg的一个字典
            cfg = yaml.safe_load(f)
            #把从文件得到的数据传入parser解析对象
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    # 如果上面的if里面的执行了就将文件参数替换默认的参数设置，否则采用前面设置的默认参数
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    # 将最新参数缓存为文本字符串，以便稍后将它们保存在输出目录中
    # args.__dict__字典形式的配置参数
    # 加入default_flow_style=False这个参数以后，重新写入后的格式跟源文件的格式就是同样的 yaml 风格了
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    #PyTorch Image Models，简称timm，是一个巨大的PyTorch代码集合，整合了常用的models、layers、utilities、optimizers、
    #schedulers、data-loaders/augmentations和reference training/validation scripts
    #参考https://zhuanlan.zhihu.com/p/404107277
    #timm.utils.setup_default_logging设置默认日志信息
    #"%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s"的日志输出形式
    setup_default_logging()

    args, args_text = _parse_args()

    # 是否使用wandb用来帮助我们跟踪机器学习的项目，通过wandb可以记录模型训练过程中指标的变化情况以及超参的设置
    # 还能够将输出的结果进行可视化的比对，帮助我们更好的分析模型在训练过程中的问题，同时我们还可以通过它来进行团队协作
    if args.log_wandb:
        #如果想要使用wandb而且有这个库，就进行初始化
        #使用wandb首先要在网站上创建team，然后在team下创建project，然后project下会记录每个实验的详细数据，这里的project参数是project名称
        #wandb.init(config=all_args,project=your_project_name,entity=your_team_name,notes=socket.gethostname(),name=your_experiment_namedir=run_dir,job_type="training",reinit=True)
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        #如果想要使用wandb但是没有这个库，就发出一个警报
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")
    #数据预取指在处理器访问该数据进行计算之前，提前将数据从主存储器加载到缓存存储器上，以降低处理器访问数据的停顿时间，以提高处理器的性能。数据预取分为软件预取和硬件预取。
    #数据预取可通过处理器内置的预取器(Prefetcher)以硬件预取方式实现，也可在程序中调用处理器的预取指令以软件预取方式实现。
    args.prefetcher = not args.no_prefetcher

    #根据WORLD_SIZE决定是否使用分布式训练
    args.distributed = False
    # 如果是多机多卡的机器，WORLD_SIZE代表使用的机器数，RANK对应第几台机器
    # 如果是单机多卡的机器，WORLD_SIZE代表有几块GPU，RANK和LOCAL_RANK代表第几块GPU
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    #此处设置的是args.distributed=False时候默认使用的gpu
    args.device = 'cuda:0'#设置使用cuda设备号,原来这里是1,我电脑只有一个GPU所以改成0
    args.world_size = 1
    args.rank = 0  # global rank
    # 要使用分布式时候添加执行的内容
    if args.distributed:
        #设置当前使用的cuda
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        #torch.distributed.init_process_group的参数：
        # backend(str): 后端选择，包括上面那几种 gloo,nccl,mpi;init_method(str，optional): 用来初始化包的URL, 一个用来做并发控制的共享方式;
        # world_size(int, optional): 参与这个工作的进程数;rank(int,optional): 当前进程的rankl;group_name(str,optional): 用来标记这组进程名的
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        #获取world size，在不同进程里都是一样的
        args.world_size = torch.distributed.get_world_size()
        #获取rank，每个进程都有自己的序号，各不相同
        args.rank = torch.distributed.get_rank()
        #打印日志
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    #保证进程号没有异常
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    # 混合精度 (Automatically Mixed Precision, AMP)
    use_amp = None
    #选择使用pytorch还是apex来实现混合精度，优先使用pytorch的torch.cuda.amp,下方的解释是apex的版本不会被积极维护
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    #这里看起来先判断apex,但是实际优先选择native,因为has_apex/has_native_amp在上方的if里决定
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:#两个都不能使用的时候发出一个警告
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")
    #使用timm.utils.random_seed来初始化args.seed+args.rank的结果作为随机种子
    #包括torch.manual_seed,np.random.seed以及python自带的random.seed
    random_seed(args.seed, args.rank)
    #原始定义是def create_model(model_name,pretrained=False,checkpoint_path='',scriptable=None,exportable=None,no_jit=None,**kwargs)
    #pretrained=Flase是从零开始训练
    #问1:drop_rate=0不知道什么用的，cifar10和imagemet模型中没用上
    #drop_path_rate就是随机梯度衰减的最大值,depths是编码器层数L
    #问2:drop_block_rate=None不知道是什么cifar10好imagenet模型中没有这个
    #img_size_h,img_size_w,in_channels就是输入的图像尺寸和通道数，numclass是有多少类，比如cifar10为10
    #qkv_bias就是是否对SSA中的qkv矩阵加偏置
    #问3:sr_ratios一开始以为是SSA中的s，实际上再cifar10模型中没用上，SSA中的s他直接默认赋值0.125，所以不知道这个是什么用的
    #patch_size就是分割图像为子图像的大小,embed_dims就是特征维度D,num_heads自注意力头的个数H
    #mlp_ratios一般为4用于乘以特征维度D计算mlp隐藏维度
    #create_model里面会运行is_model(model_name) 从而判断该model是否已经注册在timm的_model_entrypoints字典中
    #再create_fn = model_entrypoint(model_name)，model = create_fn(pretrained=pretrained, **kwargs)
    #model.py中的register_model是怎么修改 _model_entrypoints
    model = create_model(
        'spikformer',
        pretrained=False,
        drop_rate=0.,
        drop_path_rate=0.,
        drop_block_rate=None,
        img_size_h=args.img_size, img_size_w=args.img_size,
        patch_size=args.patch_size, embed_dims=args.dim, num_heads=args.num_heads, mlp_ratios=args.mlp_ratio,
        in_channels=3, num_classes=args.num_classes, qkv_bias=False,
        depths=args.layer, sr_ratios=1,
        T=args.time_step
    )
    print("Creating model")
    #统计要训练的参数的数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #例如某次实验输出：number of params: 9330874
    print(f"number of params: {n_parameters}")

    #如果没有args.num_classes从模型的num_classes获取
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly
    #仅在一台机子上打印创建成功的信息
    #例如输出INFO:train:Model spikformer created, param count:9330874
    if args.local_rank == 0:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    #timm.data.resolve_data_config
    #var()用于返回指定对象的 dict 属性，verbose判断是否输出日志信息
    #resolve_data_config主要包含解析输入/图像大小、解析插值法、解析数据集+模型均值进行归一化、解决数据集+模型STD偏差的归一化、解析默认裁剪百分比这5个过程
    #data_config是一个字典从模型里获取input_size,interpolation,mean,std,crop_pct这五个键值对，如果模型里没有会添加默认值
    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    #args.aug_splits含义: Number of augmentation splits (default: 0, valid: 0 or >=2)
    #设置扩增批分割，如果扩增批分割（augmentation batch splits）的数量大于1，将所有模型的BatchNormlayers转换为Split batch Normalization层
    #通常，当我们训练一个模型时，我们将数据增强【data augmentation】应用于完整批【batch】，然后从这个完整批定义批规范【batch norm】统计数据，如均值和方差。
    #但有时将数据分成组，并为每个组使用单独的批处理归一化层【Batch Normalization layers】来独立地归一化组是有益的。
    #这在一些论文中被称为辅助批规范【auxiliary batch norm】，在timm中被称为SplitBatchNorm2d
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits
    # enable split bn (separate bn stats per batch-portion)
    # args.split_bn含义: Enable separate BN layers per augmentation split.
    print("args.split_bn: ",args.split_bn)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        #timm.models.convert_splitbn_model，源码是递归的把所有的batchnorm变为SplitBatchNorm2d
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    #model.cuda将模型的所有参数和缓冲移动到GPU中
    #cuda在模型或者数据中都可以使用，都能实现从CPU到GPU的内存迁移，但是他们的作用效果有所不同
    #和nn.Module不同，调用tensor.cuda()只是返回这个tensor对象在GPU内存上的拷贝，而不会对自身进行改变，因此必须对tensor进行重新赋值，即tensor=tensor.cuda().
    model.cuda()
    #如果使用了channels_last内存布局
    if args.channels_last:
        #memory_format(torch.memory_format, optional)：期望返回的tensor的内存格式，默认为torch.preserve_format.
        #channels_last是对卷积网络内存格式的支持。此格式旨在与AMP结合使用，以进一步加速具有Tensor Cores的卷积神经网络
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    # 为分布式训练设置同步的BatchNorm，判定采用分布式且需要同步的时候才执行
    if args.distributed and args.sync_bn:
        assert not args.split_bn#该同步方法与SplitBatchNorm2d矛盾，所以不能同时使用
        if has_apex and use_amp != 'native':
            # Apex SyncBN preferred unless native amp is activated
            # 除非使用了"native amp"即torch.cuda.amp不然使用apex
            # 使用的是上方的from apex.parallel import convert_syncbn_model
            model = convert_syncbn_model(model)
        else:
            #使用torch.nn的同步方法
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        #仅在一台机子上打印已经转换的信息
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
    #判断是否使用torch.jit.script。torch.jit.script与apex.am和同步的BatchNorm都不兼容
    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        # torch.jit.script在编译function或 nn.Module 脚本将检查源代码，使用 TorchScript 编译器将其编译为 TorchScript 代码
        model = torch.jit.script(model)

    #完整名为timm.optim.create_optimizer_v2,创建优化器，具体参数与args中输入的配置相同
    #optimizer_kwargs将argparse args或cfg一样的对象中的优化器参数转换为需要的关键字参数，输出的字典包括opt,lr,weight_decay,momentum,eps,betas,opt_args这些参数
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    # 前面只是设置了可以使用AMP,这里具体设置AMP的使用
    # 全名为contextlib.suppress，返回一个上下文管理器，用于抑制指定异常
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    #使用apex的AMP混合精度
    if use_amp == 'apex':
        #全名为 apex.amp.initialize
        #直接使用 amp.initialize 包装模型和优化器，apex 就会自动帮助我们管理模型参数和优化器的精度了，根据精度需求不同可以传入其他配置参数
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        #timm.utils.ApexScaler
        #loss_scaler函数，它的作用本质上是loss.backward()和optimizer.step().
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    #使用torch.cuda.amp的AMP混合精度
    elif use_amp == 'native':
        #即使了混合精度训练，还是存在无法收敛的情况，原因是激活梯度的值太小，造成了溢出。
        #可以通过使用torch.cuda.amp.GradScaler，通过放大loss的值来防止梯度的underflow（只在BP时传递梯度信息使用，真正更新权重时还是要把放大的梯度再unscale回去）；
        #反向传播前，将损失变化手动增大2^k倍，因此反向传播时得到的中间变量（激活函数梯度）则不会溢出；
        #反向传播后，将权重梯度缩小2^k倍，恢复正常值
        amp_autocast = torch.cuda.amp.autocast
        #Loss Scale主要是为了解决fp16 underflow的问题。刚才提到，训练到了后期，梯度（特别是激活函数平滑段的梯度）会特别小，fp16 表示容易产生 underflow 现象。
        #loss_scaler函数，它的作用本质上是loss.backward()和optimizer.step().
        #全名为timm.utils.NativeScaler
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    #不使用AMP混合精度，仅采用32精度
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    #optionally resume from a checkpoint
    #args.resume:path类型
    #选择地从检查点恢复完整的模型和优化器状态，意思就是假设上次训练到了一半停下来了，我们可以从上次训练的地方开始继续训练
    resume_epoch = None
    if args.resume:
        #全名为timm.models.resume_checkpoint，这里返回的resume_epoch后面赋值给start_epoch
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    # 建立模型权重的指数移动平均，SWA也可以在这里使用
    # EMA指数移动平均：shadow权重是通过历史的模型权重指数加权平均数来累积的，每次shadow权重的更新都会受上一次shadow权重的影响，
    # 所以shadow权重的更新都会带有前几次模型权重的惯性，历史权重越久远，其重要性就越小，这样可以使得权重更新更加平滑。
    # SWA随机权重平均：在优化的末期取k个优化轨迹上的checkpoints，平均他们的权重，得到最终的网络权重，
    # 这样就会使得最终的权重位于flat曲面更中心的位置，缓解权重震荡问题，获得一个更加平滑的解，相比于传统训练有更泛化的解。
    # 1.EMA需要在每步训练时，同步更新shadow权重，但其计算量与模型的反向传播相比，成本很小，因此实际上并不会拖慢很对模型的训练进度；
    # 2.SWA可以在训练结束，进行手动加权，完全不增加额外的训练成本；
    # 3.实际使用两者可以配合使用，可以带来一点模型性能提升。
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        # 要在cuda()， DP包装器和AMP之后，但在SyncBN和DDP包装器之前创建EMA模型
        # timm.utils.ModelEmaV2 返回一份复制的加权移动平均累积模型
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            #加载入上一次结束的地方
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    # 设置分布式训练
    if args.distributed:
        if has_apex and use_amp != 'native':#使用apex库来分布式训练
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            #ApexDDP实际为apex.parallel.DistributedDataParallel
            model = ApexDDP(model, delay_allreduce=True)
        else:#使用pytorch来分布式训练
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            #NativeDDP实际为torch.nn.parallel.DistributedDataParallel,使用DistributedDataParallel进行单机多卡或者多机多卡分布式训练
            #DPP容器通过在批处理维度中分组，将输入分割到指定的设备上，从而并行化给定的模块。
            #模块被复制到每台机器和每台设备上，每个这样的副本处理输入的一部分。在反向传播时，每个节点的梯度被平均。批处理大小应该大于本地使用的GPU数量。
            model = NativeDDP(model, device_ids=[args.local_rank],find_unused_parameters=True)  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    # 设置学习率计划和开始epoch
    #全名为timm.scheduler.create_scheduler
    # 创建任务计划
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    #后面若没有修改就默认为0即从头开始
    start_epoch = 0
    #首先考虑作为参数输入的开始训练周期
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    #继承上次模型训练结束的周期
    elif resume_epoch is not None:
        start_epoch = resume_epoch

    #将对应的学习率就会按照策略调整start_epoch次
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    #打印总共训练周期数
    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # create the train and eval datasets
    # create_dataset全名为timm.data.create_dataset
    # 创建训练集
    # spilt:根据用户提供的参数将输入数据集拆分为两部分
    # dataset.shuffle作用是将数据进行打乱操作，传入参数为buffer_size，改参数为设置“打乱缓存区大小”，也就是说程序会维持一个buffer_size大小的缓存，每次都会随机在这个缓存区抽取一定数量的数据
    # dataset.batch作用是将数据打包成batch_size
    # dataset.repeat作用就是将数据重复使用多少epoch
    # repeats的知识:sample size不能整除 batch size的时候，最后一个batch的数据会不满用户自己设置的batch size，train的时候只会在不满 batch size的batch里training
    # 一种方法是drop每个epoch的最后一个batch就可以，另一种方案是使用repeat，repeat不设置次数之后，dataset变成一个无界的dataset了
    # batch size过小，花费时间多，同时梯度震荡严重，不利于收敛；batch size过大，不同batch的梯度方向没有任何变化，容易陷入局部极小值
    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir, split=args.train_split, is_training=True,
        batch_size=args.batch_size, repeats=args.epoch_repeats)
    #创建评估集
    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False, batch_size=args.batch_size)

    # setup mixup / cutmix
    # 设置mixup / cutmix
    # mixup,cutmix算法:二者都是图像混合增强手段，即在训练时，我们将两个样本按照某种方式进行混合，并相应地混合它们的标签。其中 Mixup和CutMix的区别就在于按照什么方式对图像进行混合。
    # 这种图像混合增强的目的是使图像经过神经网络映射后嵌入的低维流形变得平滑，从而提高网络的泛化能力。
    collate_fn = None
    mixup_fn = None
    #判断是否使用图像混合增强手段
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            #与扩增批分割方法矛盾不能同时用
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            # 有数据预取的话使用 timm.dataset.FastCollateMixup
            # 快速整理 Mixup/Cutmix 应用不同的参数到每个元素或整个批次
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            # Mixup/Cutmix 应用不同的参数到每个元素或整个批次
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    # 使用AugMix帮助器包装数据集
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    # 创建带有增强管道的数据加载器
    #args.train_interpolation是使用的插值的参数
    train_interpolation = args.train_interpolation
    #args.no_aug为真(禁用所有训练增强，覆盖其他训练aug参数)且没有设置插值方法，就设置一个默认插值方法
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']

    #可以将数据集看作是原始数据的集合，而数据加载器则是负责从数据集中读取数据并生成训练样本的工具。
    #使用数据集和数据加载器可以方便地读取和处理大量的数据，同时可以更好地控制数据的生成和使用方式
    #创建训练数据加载器
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader
    )

    #创建评估数据加载器
    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.val_batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    # setup loss function
    # 设置损失函数
    # 判断是否支持JS散度和CE loss要与aug-splits一起使用
    if args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        #采用JS散度 + 交叉熵损失
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
    elif mixup_active:
        # smoothing is handled with mixup target transform
        # 在使用图像混合增强时使用soft target
        train_loss_fn = SoftTargetCrossEntropy().cuda()
    elif args.smoothing:
        # 采用带标签平滑的负对数似然损失
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    else:
        #训练集默认采用交叉熵损失函数
        train_loss_fn = nn.CrossEntropyLoss().cuda()
    #评估集默认采用交叉熵损失函数
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    # 设置检查点保护程序和评估指标跟踪
    # eval_metric是评价函数，对模型的训练没有影响，而是在模型训练完成之后评估模型效果
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.rank == 0:
        # args.experiment是本次训练要建立的用于输出的子文件夹的名字
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        # 将几个输入路径连接，获取输出的完整路径
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        # 看了源码，感觉意思是根据decreasing决定某个数(或许是损失值)取最大值好还是最小值好
        decreasing = True if eval_metric == 'loss' else False
        #timm.utils.CheckpointSaver
        # 跟踪top-n个训练检查点，并在指定的时间间隔内维护恢复检查点。感觉就是用来保存中间模型结果用的。
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        #记录本次训练的配置文件
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    #开始训练
    try:
        for epoch in range(start_epoch, num_epochs):
            # 如果是分布是训练，各个进程互通自己的轮数
            # 设置此采样器的轮数。当'shuffle=True'时，这将确保所有副本对每一轮使用不同的随机顺序。否则，该采样器的下一次迭代将产生相同的排序。
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            #例如某次结果为OrderedDict([('loss', 2.14468679446417)])
            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)

            # 在每个轮之后在节点之间分发BatchNorm统计数据
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                # timm.utils.distribute_bn
                # 保证每个进程节点有相同的bn状态
                # reduce的话就average bn stats across whole group
                # broadcast的话就bn stats from rank 0 to whole group
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')
            #OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)


            if model_ema is not None and not args.model_ema_force_cpu:
                # 保证每个进程节点有相同的bn状态
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                #计算ema模型的结果
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast,
                    log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            #调整学习率
            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])
            #输出文件summary.csv加上每一轮的数据
            if output_dir is not None:
                #timm.utils.update_summary
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)
            #保存每一轮的last.pth模型文件
            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
                _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

    except KeyboardInterrupt:
        pass
    #训练结束，输出得到的最优秀的模型
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):
    # 如果mixup_off_epoch为1那么就在指定轮数之后关闭mixup混合增强方法
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False
    # first-order optimization和second- order optimization分别对应泰勒展开式的一次项式和泰勒的二次项式进行优化拟合
    # 利用second order方法去拟合我们data，优势就是没有一些超参数，例如learning rate；
    # 与此同时他在深度学习有个致命的弱点就是：如果我们有个neuron，那么Hessian(二阶导)的O(N^2)个元素，而Hessian的逆矩阵却有 O(N^3)个元素
    # 不过要注意我们的N = (Tens or Hundreds of) Millions，所占空间很大
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

    # timm.utils.AverageMeter：用于统计某个数据在一定次数内的平均值和总个数
    # 统计每批平均用时
    batch_time_m = AverageMeter()
    # 统计每次循环总用时
    data_time_m = AverageMeter()
    # 统计平均损失
    losses_m = AverageMeter()
    # 运行 model.train()之后，相当于告诉了 BN 层，对之后输入的每个 batch 独立计算其均值和方差，BN层的参数是在不断变化的
    # model.eval() ，相当于告诉 BN 层，我现在要测试了，你用刚刚统计的 μ和 σ来测试我，不要再变了。
    model.train()

    end = time.time()
    #最后一轮的id号
    last_idx = len(loader) - 1
    #对于1001个样本，batch_size的大小是10，train_loader的长度len(train_loader)=101，最后一个batch将仅含一个样本。
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        #判断是不是最后一批
        last_batch = batch_idx == last_idx
        #更新每轮的用时，不过为什么放在开头，感觉很奇怪，感觉应该放在后面
        data_time_m.update(time.time() - end)
        # 如果没有预取，就手动载入gpu
        # Prefetcher(预取)单元的作用是预先取得内存中的数据放到缓存中备用,以加快内存的潜伏期。
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            #如果有图像混合增强手段，就进行处理
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)

        if args.channels_last:
            #当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，使得两个tensor完全没有联系
            #channels_last是对卷积网络内存格式的支持。
            input = input.contiguous(memory_format=torch.channels_last)

        # with amp_autocast使操作自动混合精度转换
        # (在源码实现上是定义了查找表，不同的 OP 会依据查找表采用不同的精度计算)
        with amp_autocast():
            #计算模型对应的输出
            output = model(input)
            #计算模型对应的损失
            loss = loss_fn(output, target)

        #如果没有分布式训练，就更新平均损失
        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        #清空过往梯度
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            # loss.backward()
            # 反向传播，计算当前梯度
            loss.backward(create_graph=second_order)
            # 是否使用clip函数来修剪梯度，限制梯度的大小
            if args.clip_grad is not None:
                # timm.utils.dispatch_clip_grad
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            #根据梯度更新网络参数
            optimizer.step()
            #进来一个batch的数据，计算一次梯度，更新一次网络

        #重置脉冲神经网，例如每次都要把神经元初始电位恢复
        functional.reset_net(model)

        #根据需要对函数的参数进行指数平滑
        if model_ema is not None:
            model_ema.update(model)

        #等待当前设备上所有流中的所有核心完成
        #在pytorch里面，程序的执行都是异步的，因为是异步的原因，后台的cuda推理，可能没有结束
        torch.cuda.synchronize()

        #更新已经跑的批次
        num_updates += 1
        #更新每跑一批所用的平均时间
        batch_time_m.update(time.time() - end)
        # 如果是最后一批，或者想每隔几轮打印一些信息就进入这循环
        if last_batch or batch_idx % args.log_interval == 0:
            #获取所有学习率的列表
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            #分布式时候计算全部的平均损失
            if args.distributed:
                #timm.utils.reduce_tensor
                #在计算loss或者metric时,需要对每张显卡的参数信息进行回传、汇总,采用reduce_tensor实现
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                #更新平均损失
                losses_m.update(reduced_loss.item(), input.size(0))
            #如果是本机，则输出信息
            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))
                #如果需要则保存训练过程的图片
                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)
        #如果要保存并且满足设置的保存间隔或者是最后一轮就进行保存pth模型文件
        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)
        #如果需要，就更新学习率
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for
    #lookahead机制可以理解为使用了一个optmizer保留两个权重weights。分别是fast和slow weights，sync_period表示多少个step（batch）之后发生快慢的替换
    #当在高曲率方向上振荡时，快速权重更新会沿着低曲率方向快速更新。缓慢权重有助于通过ensemble的tricks消除振荡。
    #快速权重和慢速权重的组合改善了高曲率方向的学习，减少了方差，并使 Lookahead 在实践中快速收敛
    #LookAhead通过对权重进行平滑处理，从而具有对超参鲁棒的特点。
    #尤其是当内部优化器使用较大的学习率时，权重更新可以尽快通过曲率较小的方向（梯度中值较小的分方向）
    #而使用平滑方法可以缓解权重在高曲率方向（梯度中值较大的分方向）的振荡甚至发散，从而使得算法获得使用较大学习率带来的加速收敛的收益。
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
    #
    return OrderedDict([('loss', losses_m.avg)])

#该部分与train_one_epoch大多相似，只挑其中重要的进行解释
def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):

    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    #top1准确度的平均值
    top1_m = AverageMeter()
    #top5准确度的平均值
    top5_m = AverageMeter()

    # 运行 model.train()之后，相当于告诉了 BN 层，对之后输入的每个 batch 独立计算其均值和方差，BN层的参数是在不断变化的
    # model.eval() ，相当于告诉 BN 层，我现在要测试了，你用刚刚统计的 μ和 σ来测试我，不要再变了。
    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    #不进行梯度更新地计算，否则会自动进行计算图的生成（计算过程的构建，以便梯度反向传播等操作）
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):#我运行是tensor类型所以不会进入
                output = output[0]
            #eg: output
            #tensor([[-0.7295, -0.4751,  0.1173,  0.3149,  0.4602,  0.5024,  0.9082,  0.4968,  -0.9175, -0.3203],
            # [-0.6323, -0.1598,  0.2299,  0.5244,  0.2705,  0.4668,  0.4253,  0.3687,  -0.9180, -0.2720],
            # [ 2.7773, -0.4419,  0.3225, -0.7051, -1.0830, -0.8159, -1.1377, -1.0625,   2.7363, -0.4690],
            # [ 0.4519,  0.4844, -0.2534, -0.3621, -0.4231, -0.4626, -0.4023, -0.2703,   0.8896,  0.2920],
            # [ 1.5518,  1.9941, -0.9189, -1.5176, -1.5078, -1.5879, -1.5332, -0.7109,   2.4570,  2.0098],
            # [-0.9565, -0.9165,  1.2422, -0.0988,  1.6484,  0.2214,  1.9756,  0.9229,  -1.6396, -0.9570],
            # [-0.1326, -0.3242,  0.1531,  0.7144, -0.2527,  0.6245, -0.3323,  0.2063,  -0.4844, -0.3123],
            # [ 1.2021,  2.1035, -0.5825, -1.4150, -1.0752, -1.3428, -0.8691, -1.0625,   1.6924,  1.6924],
            # [-0.5483, -0.6631,  0.0748,  1.0527,  0.0586,  0.8340,  0.3140,  0.3101,  -0.9478, -0.5571],
            # [-0.8188, -0.9800,  0.1119,  1.4053, -0.1506,  1.8594, -0.6685,  0.9331,  -1.3652, -0.6616],
            # [-1.3760, -0.9492,  0.0748,  1.0342,  0.5059,  1.2695,  1.0674,  1.1357,  -1.7754, -0.4512],
            # [ 0.9849, -0.3730,  0.2812,  0.2358, -0.4993,  0.0937, -0.6382, -0.2478,   0.2578, -0.1273],
            # [-0.7915, -0.4832,  0.0029,  0.6313,  0.2820,  0.7915,  0.4646,  0.5645,  -1.0098, -0.4617],
            # [-1.0273, -0.9253,  0.2081,  1.3770,  0.2291,  1.5361, -0.1512,  0.8516,  -1.3369, -0.8491],
            # [-0.2952,  0.1652,  0.2788, -0.3330,  0.3264, -0.1331,  0.9038,  0.4365,  -0.7803,  0.1686],
            # [-0.4011,  0.1309, -0.2642, -0.0717, -0.0232,  0.0497, -0.0296,  0.9121,  -0.4534,  0.5654]]
            # , device='cuda:0', dtype=torch.float16)
            #eg: target
            #tensor([7, 5, 8, 0, 8, 2, 7, 0, 3, 5, 3, 8, 3, 5, 1, 7], device='cuda:0')
            # augmentation reduction
            # 测试/推断时间增强因子。默认值为0，默认为0。
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            functional.reset_net(model)

            #计算top1准确率，和top5准确率
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))
                    #例如输出：Test: [  78/78]  Time: 0.303 (0.615)  Loss:  1.7197 (1.7612)  Acc@1: 25.0000 (38.6500)  Acc@5: 100.0000 (88.0500)

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


if __name__ == '__main__':
    main()