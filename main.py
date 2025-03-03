import argparse
import warnings
from datetime import datetime
from glob import glob
from shutil import copyfile
from collections import OrderedDict

import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from models.generator import Generator as Generator
from models.discriminator import Discriminator as Discriminator
from models.guidingNet import GuidingNet
from models.blocks import LinearBlock, LinearBlockMultipleLayers

from train.train import trainGAN, train_fixed_content, train_fixed_content_with_style_attraction, trainGAN_with_CLIP

from validation.validation import validateUN, infer, infer_styles, infer_from_style, evaluate_style, top_average_evaluate, make_loss_dictionary

from tools.utils import *
from datasets.datasetgetter import get_dataset, get_dataset_for_inference, get_dataset_for_clip_embedded_image

from tensorboardX import SummaryWriter

from clipfont import clip

def clip_load_model(
    model,
    checkpoint_path=None,
    requires_grad=False,
    device='cuda',
    model_name='ViT-B/32',
):
    if checkpoint_path is None:
        model, _ = clip.load(model_name, device=device, jit=False)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    if not requires_grad:
        for param in model.parameters():
            param.requires_grad = False
    return model


def main():
    # Configuration
    parser = argparse.ArgumentParser(description='PyTorch GAN Training')
    parser.add_argument(
        '--data_path',
        type=str,
        default='../data',
        help='Dataset directory. Please refer Dataset in README.md')
    parser.add_argument(
        '--workers',
        default=4,
        type=int,
        help='the number of workers of data loader')
    parser.add_argument(
        '--model_name',
        type=str,
        default='GAN',
        help='Prefix of logs and results folders. '
             'ex) --model_name=ABC generates ABC_20191230-131145 in logs and results')
    parser.add_argument(
        '--epochs',
        default=250,
        type=int,
        help='Total number of epochs to run. Not actual epoch.')
    parser.add_argument(
        '--iters',
        default=1000,
        type=int,
        help='Total number of iterations per epoch')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument(
        '--font_num_per_batch',
        default=1,
        type=int,
        help='Number of fonts per batch')
    parser.add_argument('--val_num', default=190, type=int,
                        help='Number of test images for each style')
    parser.add_argument(
        '--val_batch',
        default=10,
        type=int,
        help='Batch size for validation. '
             'The result images are stored in the form of (val_batch, val_batch) grid.')
    parser.add_argument('--log_step', default=100, type=int)
    parser.add_argument(
        '--input_ch',
        default=3,
        type=int,
        help='The number of channels of input images=. 1 means gray scale. 3 means RGB')
    parser.add_argument(
        '--sty_dim',
        default=128,
        type=int,
        help='The size of style vector')
    parser.add_argument('--output_k', default=10, type=int,
                        help='Total number of classes to use')
    parser.add_argument(
        '--img_size',
        default=80,
        type=int,
        help='Input image size')
    parser.add_argument(
        '--cnt_img_size',
        default=20,
        type=int,
        help='Content size')
    parser.add_argument(
        '--cnt_ch',
        default=256,
        type=int,
        help='Content channels')
    parser.add_argument(
        '--cnt_num',
        default=2136,
        type=int,
        help='The number of kind of characters')
    parser.add_argument(
        '--dims',
        default=2048,
        type=int,
        help='Inception dims for FID')
    parser.add_argument(
        '--load_model',
        default=None,
        type=str,
        metavar='PATH',
        help='path to latest checkpoint (default: None)'
             'ex) --load_model GAN_20190101_101010'
             'It loads the latest .ckpt file specified in checkpoint.txt in GAN_20190101_101010')
    parser.add_argument(
        '--base_dir',
        default=None,
        type=str)
    parser.add_argument(
        '--check_point_path',
        default=None,
        type=str,
        metavar='PATH',
        help='path to checkpoint (default: None)'
             'ex) --check_point_path model_3.ckpt')
    parser.add_argument('--validation', dest='validation', action='store_true',
                        help='Call for valiation only mode')
    parser.add_argument('--use_clip_vision_encoder', action='store_true',
                        help='Use clip vision encoder instead of MOCO')
    parser.add_argument('--use_linear_block', action='store_true',
                        help='User linear block to map clip embedding to style vector')
    parser.add_argument('--clip_model_name', default='ViT-B/32', type=str,)
    parser.add_argument(
        '--image_base_path_for_clip_vision_encoder',
        default='../grayscale_images_200',
        type=str)
    parser.add_argument('--clip_model_checkpoint_path', default=None, type=str,)
    parser.add_argument('--fixed_content_font', action='store_true',
                        help='Call for inference only mode')
    parser.add_argument('--style_norm', action='store_true',
                        help='Deprecated. Call for inference only mode')
    parser.add_argument('--style_attraction', action='store_true',
                        help='Call for style attraction')
    parser.add_argument('--content_norm', action='store_true',
                        help='Call for inference only mode')
    parser.add_argument('--content_discriminator', action='store_true',
                        help='Add content discriminator')
    parser.add_argument('--last_kernel', action='store_false',
                        help='Does apply dynamic last_kernel to Discriminator')
    parser.add_argument('--without_normalization', action='store_true',
                        help='Does not use normalization')
    parser.add_argument(
        '--content_font_id',
        required=False,
        nargs='*',
        type=int,
        help='Content Font Id')
    parser.add_argument('--infer', action='store_true',
                        help='Call for inference only mode')
    parser.add_argument('--infer_styles', action='store_true',
                        help='Call for inference only mode')
    parser.add_argument('--infer_from_style', action='store_true',
                        help='Call for inference only mode')
    parser.add_argument('--evaluate_style', action='store_true',
                        help='Call for Style Evaluation only mode')
    parser.add_argument('--evaluate_average_style', action='store_true',
                        help='Call for Style Evaluation only mode')
    parser.add_argument('--make_loss_dictionary', action='store_true',
                        help='Call for Style Evaluation only mode')
    parser.add_argument('--gpu', default=None, type=str,
                        help='GPU id to use.')
    parser.add_argument(
        '--ddp',
        dest='ddp',
        action='store_true',
        help='Call if using DDP')
    parser.add_argument('--port', default='8993', type=str)
    parser.add_argument(
        '--iid_mode',
        default='iid+',
        type=str,
        choices=[
            'iid',
            'iid+'])
    parser.add_argument(
        '--w_gp',
        default=10.0,
        type=float,
        help='Coefficient of GP of D')
    parser.add_argument('--w_rec', default=0.1, type=float,
                        help='Coefficient of Rec. loss of G')
    parser.add_argument('--w_sty', default=0.1, type=float,
                        help='Coefficient of Rec. loss of G')
    parser.add_argument('--w_adv', default=1.0, type=float,
                        help='Coefficient of Adv. loss of G')
    parser.add_argument('--w_vec', default=0.01, type=float,
                        help='Coefficient of Style vector rec. loss of G')
    parser.add_argument('--w_sty_norm', default=1e-5, type=float,
                        help='Coefficient of Style vector rec. loss of G')
    parser.add_argument('--w_cnt_norm', default=0.1, type=float,
                        help='Coefficient of Style vector rec. loss of G')
    parser.add_argument('--w_sty_var', default=0.1, type=float,
                        help='Coefficient of Style vector rec. loss of G')
    parser.add_argument('--w_cd', default=0.1, type=float,
                        help='Weight of content discriminator')

    parser.add_argument(
        '--w_off',
        default=0.5,
        type=float,
        help='Coefficient of offset normalization. loss of G')
    parser.add_argument(
        '--check_point_step',
        default=20,
        type=int,
        help='check point step')
    parser.add_argument(
        '--style_img_paths',
        required=False,
        nargs='*',
        type=str,
        help='Image Paths for inference')
    parser.add_argument(
        '--content_img_paths',
        required=False,
        nargs='*',
        type=str,
        help='Image Paths for inference')
    parser.add_argument(
        '--lr',
        default=1e-4,
        type=float,
        help='Learning Rate')

    ####################
    # Default settings #
    ####################
    args = parser.parse_args()
    assert args.val_num >= args.val_batch

    # NAND
    assert not (args.validation and args.infer)

    if args.infer:
        assert args.style_img_paths is not None
        assert args.content_img_paths is not None
        assert len(args.style_img_paths) == len(args.content_img_paths)

    if args.fixed_content_font:
        assert args.content_font_id is not None

    print("______________________________________")
    print(f'Style Norm: {args.style_norm}')
    print(f'Content Norm: {args.content_norm}')
    print(f'Content Discriminator: {args.content_discriminator}')
    print("______________________________________")

    print("PYTORCH VERSION", torch.__version__)
    args.data_dir = args.data_path
    args.start_epoch = 0

    args.train_mode = 'GAN'

    # unsup_start : train networks with supervised data only before unsup_start
    # separated : train IIC only until epoch = args.separated
    # ema_start : Apply EMA to Generator after args.ema_start

    args.unsup_start = 0
    args.separated = 0
    args.ema_start = 1
    args.fid_start = 1

    # Cuda Set-up
    # if args.gpu is not None:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #     warnings.warn('You have chosen a specific GPU. This will completely '
    #                   'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()

    args.ngpus_per_node = ngpus_per_node

    # Logs / Results
    if args.load_model is None:
        args.model_name = '{}_{}'.format(
            args.model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        args.model_name = args.load_model

    makedirs('./logs')
    makedirs('./results')

    args.log_dir = os.path.join('./logs', args.model_name)
    args.event_dir = os.path.join(args.log_dir, 'events')
    args.res_dir = os.path.join('./results', args.model_name)

    makedirs(args.log_dir)
    dirs_to_make = next(os.walk('./'))[1]
    not_dirs = [
        '.idea',
        '.git',
        'logs',
        'results',
        '.gitignore',
        '.nsmlignore',
        'resrc']

    makedirs(os.path.join(args.log_dir, 'codes'))
    for to_make in dirs_to_make:
        if to_make in not_dirs:
            continue
        makedirs(os.path.join(args.log_dir, 'codes', to_make))
    makedirs(args.res_dir)

    if args.load_model is None:
        pyfiles = glob("./*.py")
        for py in pyfiles:
            copyfile(py, os.path.join(args.log_dir, 'codes') + "/" + py)

        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            tmp_files = glob(os.path.join('./', to_make, "*.py"))
            for py in tmp_files:
                copyfile(py, os.path.join(args.log_dir, 'codes', py[2:]))

    main_worker(args)


def main_worker(args):
    if args.gpu is None:
        args.device = 'cpu'
    else:
        args.device = 'cuda'

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.device))
    else:
        print("Use CPU: {} for training".format(args.device))

    # Load CLIP
    clip_model, clip_preprocess = clip.load(args.clip_model_name, device=args.device)
    clip_model = clip_load_model(clip_model, args.clip_model_checkpoint_path, requires_grad=False, device=args.device, model_name=args.clip_model_name)

    # # of GT-classes
    args.num_cls = args.output_k

    # Classes to use
    args.att_to_use = [i for i in range(0, args.num_cls)]

    # IIC statistics
    args.epoch_acc = []
    args.epoch_avg_subhead_acc = []
    args.epoch_stats = []

    # build model - return dict
    networks, opts = build_model(args)

    # load model if args.load_model is specified
    load_model(args, networks, opts)
    cudnn.benchmark = True

    # print all the argument
    print_args(args)

    if args.evaluate_style:
        print('EVALUATE STYLE')
        evaluate_style(networks, args)
        return
    if args.evaluate_average_style:
        print('EVALUATE AVERAGE STYLE')
        top_average_evaluate(networks, args)
        return
    if args.make_loss_dictionary:
        print('EVALUATE MAKE LOSS DICTIONARY')
        make_loss_dictionary(networks, args)
        return

    if args.infer:
        print('START INFERING IMAGES')
        style_dataset = get_dataset_for_inference(args, args.style_img_paths)
        content_dataset = get_dataset_for_inference(
            args, args.content_img_paths)
        infered_images = infer(style_dataset, content_dataset, networks, args)
        infered_images = to_pil_image(make_grid(infered_images.to('cpu')))
        infered_images.save('../sample.png')
        print('FINISH INFERING IMAGES')
        return infered_images

    # Logging
    logger = SummaryWriter(args.event_dir)

    # get dataset and data loader
    train_dataset, content_dataset = get_dataset(args)

    clip_vision_dataset = None
    clip_vision_loader = None
    clip_content_dataset = None
    clip_content_loader = None
    if args.use_clip_vision_encoder:
        clip_vision_dataset = get_dataset_for_clip_embedded_image(args, args.image_base_path_for_clip_vision_encoder, args.data_dir, clip_model, clip_preprocess)
        clip_vision_loader = torch.utils.data.DataLoader(
            clip_vision_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            sampler=None,
            drop_last=False)

    clip_model = clip_model.to('cpu')

    args.shuffle = True
    # if args.style_norm:
    #     args.shuffle = False

    train_loader = get_loader(
        args,
        train_dataset,
        shuffle=args.shuffle,
        is_style_attraction=args.style_attraction)

    content_loader = None
    if content_dataset is not None:
        content_loader = get_loader(
            args,
            content_dataset,
            shuffle=True,
            is_content=True,
            is_style_attraction=False)

    # For saving the model
    record_txt = open(os.path.join(args.log_dir, "record.txt"), "a+")
    for arg in vars(args):
        record_txt.write(
            '{:35}{:20}\n'.format(
                arg, str(
                    getattr(
                        args, arg))))
    record_txt.close()

    # Run

    for epoch in range(args.start_epoch, args.epochs):
        print("START EPOCH[{}]".format(epoch + 1))

        if epoch == args.ema_start and 'GAN' in args.train_mode:
            networks['G_EMA'].load_state_dict(networks['G'].state_dict())

        if args.use_clip_vision_encoder:
            trainGAN_with_CLIP(clip_vision_loader, train_loader, networks, opts, epoch, args, {'logger': logger})
        elif args.fixed_content_font:
            assert content_loader is not None
            if args.style_attraction:
                train_fixed_content_with_style_attraction(
                    train_loader, content_loader, networks, opts, epoch, args, {
                        'logger': logger})
            else:
                train_fixed_content(
                    train_loader, content_loader, networks, opts, epoch, args, {
                        'logger': logger})
        else:
            trainGAN(train_loader, networks, opts,
                     epoch, args, {'logger': logger})

        if (epoch + 1) % (args.check_point_step) == 0:
            save_model(args, epoch, networks, opts)

        print("\nFINISH EPOCH[{}]".format(epoch + 1))
    print("\nFINISH !!!!!!!!!!!!!!!!1")

#################
# Sub functions #
#################


def print_args(args):
    for arg in vars(args):
        print('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))


def build_model(args):
    args.to_train = 'CDG'
    if args.use_clip_vision_encoder:
        if args.use_linear_block:
            args.to_train = 'DGL'
        else:
            args.to_train = 'DG'

    networks = {}
    opts = {}
    if 'C' in args.to_train:
        networks['C'] = GuidingNet(
            args.img_size, {
                'cont': args.sty_dim, 'disc': args.output_k}, input_ch=args.input_ch)
        networks['C_EMA'] = GuidingNet(
            args.img_size, {
                'cont': args.sty_dim, 'disc': args.output_k}, input_ch=args.input_ch)
    if 'D' in args.to_train:
        networks['D'] = Discriminator(
            args.img_size,
            num_domains=args.output_k,
            input_ch=args.input_ch,
            last_kernel=args.last_kernel,
        )
        if args.content_discriminator:
            networks['CD'] = Discriminator(
                args.cnt_img_size,
                num_domains=args.cnt_num,
                input_ch=args.cnt_ch,
                last_kernel=args.last_kernel,
            )
    if 'G' in args.to_train:
        networks['G'] = Generator(
            args.img_size,
            args.sty_dim,
            use_sn=False,
            device=args.device,
            input_ch=args.input_ch,
            output_ch=args.input_ch)
        networks['G_EMA'] = Generator(
            args.img_size,
            args.sty_dim,
            use_sn=False,
            device=args.device,
            input_ch=args.input_ch,
            output_ch=args.input_ch)
    if 'L' in args.to_train:
        networks['L'] = LinearBlockMultipleLayers(
            512,
            args.sty_dim,
            norm='none',
            act='relu',
            use_sn=False)

    if args.device is not None:
        # torch.cuda.set_device(args.device)
        for name, net in networks.items():
            # networks[name] = net.cuda(args.gpu)
            # networks[name] = net.cuda(args.device)
            networks[name] = net.to(args.device)
    else:
        for name, net in networks.items():
            # networks[name] = torch.nn.DataParallel(net).cuda()
            networks[name] = torch.nn.DataParallel(net)

    if 'C' in args.to_train:
        opts['C'] = torch.optim.Adam(
            networks['C'].parameters(), args.lr, weight_decay=0.001)
        networks['C_EMA'].load_state_dict(networks['C'].state_dict())
    if 'D' in args.to_train:
        opts['D'] = torch.optim.RMSprop(
            networks['D'].parameters(), args.lr, weight_decay=0.0001)
        if args.content_discriminator:
            opts['CD'] = torch.optim.RMSprop(
                networks['CD'].parameters(), args.lr, weight_decay=0.0001)
    if 'G' in args.to_train:
        opts['G'] = torch.optim.RMSprop(
            networks['G'].parameters(), args.lr, weight_decay=0.0001)
    if 'L' in args.to_train:
        opts['L'] = torch.optim.Adam(
            networks['L'].parameters(), args.lr, weight_decay=0.0001)

    return networks, opts


def load_model(args, networks, opts):
    if args.check_point_path is not None:
        load_file = args.check_point_path
    elif args.load_model is not None:
        check_load = open(os.path.join(args.log_dir, "checkpoint.txt"), 'r')
        to_restore = check_load.readlines()[-1].strip()
        load_file = os.path.join(args.log_dir, to_restore)
    else:
        print("=> no checkpoint found at '{}'".format(args.log_dir))
        return

    if os.path.isfile(load_file):
        print("=> loading checkpoint '{}'".format(load_file))
        checkpoint = torch.load(load_file, map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        for name, net in networks.items():
            tmp_keys = next(
                iter(checkpoint[name + '_state_dict'].keys()))
            if 'module' in tmp_keys:
                tmp_new_dict = OrderedDict()
                for key, val in checkpoint[name +
                                           '_state_dict'].items():
                    tmp_new_dict[key[7:]] = val
                net.load_state_dict(tmp_new_dict)
                networks[name] = net
            else:
                net.load_state_dict(checkpoint[name + '_state_dict'])
                networks[name] = net

        for name, opt in opts.items():
            opt.load_state_dict(checkpoint[name.lower() + '_optimizer'])
            opts[name] = opt
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(load_file, checkpoint['epoch']))


def get_loader(
        args,
        dataset,
        shuffle=True,
        is_content=False,
        is_style_attraction=False):
    train_dataset = dataset
    train_dataset_ = train_dataset

    if args.content_norm and is_content:
        train_loader = torch.utils.data.DataLoader(
            train_dataset_,
            batch_size=1,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            sampler=None,
            drop_last=False)
    elif is_style_attraction:
        train_loader = torch.utils.data.DataLoader(
            train_dataset_,
            batch_size=args.font_num_per_batch,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            sampler=None,
            drop_last=False)

    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset_,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            sampler=None,
            drop_last=False)

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=args.val_batch,
    #     shuffle=shuffle,
    #     num_workers=0,
    #     pin_memory=True,
    #     drop_last=False)
    #
    # val_loader = {
    #     'VAL': val_loader,
    #     'VALSET': val_dataset,
    #     'TRAINSET': train_dataset['FULL']}

    return train_loader


def save_model(args, epoch, networks, opts):
    check_list = open(os.path.join(args.log_dir, "checkpoint.txt"), "a+")
    # if (epoch + 1) % (args.epochs//10) == 0:
    with torch.no_grad():
        save_dict = {}
        save_dict['epoch'] = epoch + 1
        for name, net in networks.items():
            save_dict[name + '_state_dict'] = net.state_dict()
            if name in ['G_EMA', 'C_EMA']:
                continue
            save_dict[name.lower() +
                      '_optimizer'] = opts[name].state_dict()
        print("SAVE CHECKPOINT[{}] DONE".format(epoch + 1))
        save_checkpoint(save_dict, check_list, args.log_dir, epoch + 1)
    check_list.close()


if __name__ == '__main__':
    main()
