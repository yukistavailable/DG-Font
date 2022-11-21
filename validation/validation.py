import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
from torchvision import transforms
import os


try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from scipy import linalg

from tools.utils import *


def infer_same(full_dataset, networks, args):
    # set nets
    D = networks['D']
    G = networks['G']
    C = networks['C']
    C_EMA = networks['C_EMA']
    G_EMA = networks['G_EMA']
    # switch to train mode
    D.eval()
    G.eval()
    C.eval()
    C_EMA.eval()
    G_EMA.eval()

    with torch.no_grad():
        dl = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False)
        it = iter(dl)
        result = None
        for imgs, _ in it:
            imgs = imgs.to(args.device)
            styles = C.moco(imgs)

            contents, skip1, skip2 = G.cnt_encoder(imgs)
            x_fake, _ = G.decode(contents, styles, skip1, skip2)
            if result is None:
                result = x_fake.clone()
            else:
                result = torch.cat((result, x_fake), 0)
    return result


def infer(
        style_dataset,
        content_dataset,
        networks,
        args):
    # set nets
    G = networks['G']
    C = networks['C']
    C_EMA = networks['C_EMA']
    G_EMA = networks['G_EMA']
    # switch to train mode
    G.eval()
    C.eval()
    C_EMA.eval()
    G_EMA.eval()

    assert len(style_dataset) == len(content_dataset)
    with torch.no_grad():
        style_dl = torch.utils.data.DataLoader(
            style_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False)
        style_it = iter(style_dl)

        content_dl = torch.utils.data.DataLoader(
            content_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False)
        content_it = iter(content_dl)
        result = None
        for (
                style_imgs, _), (content_imgs, _) in zip(
                style_it, content_it):
            style_imgs = style_imgs.to(args.device)
            content_imgs = content_imgs.to(args.device)
            styles = C.moco(style_imgs)

            contents, skip1, skip2 = G.cnt_encoder(content_imgs)
            x_fake, _ = G.decode(contents, styles, skip1, skip2)
            if result is None:
                result = x_fake.clone()
            else:
                result = torch.cat((result, x_fake), 0)
    return result


def infer_with_tensor(
        style_tensor,
        content_tensor,
        networks,
        args):
    # set nets
    G = networks['G']
    C = networks['C']
    C_EMA = networks['C_EMA']
    G_EMA = networks['G_EMA']
    # switch to train mode
    G.eval()
    C.eval()
    C_EMA.eval()
    G_EMA.eval()

    assert style_tensor.shape == content_tensor.shape
    with torch.no_grad():
        style_tensor = style_tensor.to(args.device)
        content_tensor = content_tensor.to(args.device)
        styles = C.moco(style_tensor)

        contents, skip1, skip2 = G.cnt_encoder(content_tensor)
        x_fake, _ = G.decode(contents, styles, skip1, skip2)
    return x_fake


def infer_styles(
        style_dataset,
        networks,
        args):
    # set nets
    C = networks['C']
    C_EMA = networks['C_EMA']

    # switch to eval mode
    C.eval()
    C_EMA.eval()

    with torch.no_grad():
        style_dl = torch.utils.data.DataLoader(
            style_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False)
        style_it = iter(style_dl)

        result = None
        for (style_imgs, _) in style_it:
            style_imgs = style_imgs.to(args.device)
            styles = C.moco(style_imgs)

            if result is None:
                result = styles.clone()
            else:
                result = torch.cat((result, styles), 0)
    return result


def infer_styles_with_tensor(
        character_tensor,
        networks,
        args):
    # set nets
    C = networks['C']
    C_EMA = networks['C_EMA']

    # switch to eval mode
    C.eval()
    C_EMA.eval()

    with torch.no_grad():
        character_tensor = character_tensor.to(args.device)
        styles = C.moco(character_tensor)
    return styles


def infer_from_style(
        content_dataset,
        style,
        networks,
        args):
    # set nets
    G = networks['G']
    G_EMA = networks['G_EMA']

    # switch to train mode
    G.eval()
    G_EMA.eval()

    with torch.no_grad():
        content_dl = torch.utils.data.DataLoader(
            content_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False)
        content_it = iter(content_dl)

        styles = None
        style = style.to(args.device)
        style = style.unsqueeze(0)
        for _ in range(args.batch_size):
            if styles is None:
                styles = style.clone()
            else:
                styles = torch.cat((styles, style), 0)

        result = None
        for (content_imgs, _) in content_it:
            content_imgs = content_imgs.to(args.device)

            contents, skip1, skip2 = G.cnt_encoder(content_imgs)
            x_fake, _ = G.decode(
                contents, styles[:len(content_imgs)], skip1, skip2)
            if result is None:
                result = x_fake.clone()
            else:
                result = torch.cat((result, x_fake), 0)
    return result


def infer_contents_with_tensor(
        character_tensor,
        networks,
        args):
    # set nets
    G = networks['G']

    # switch to train mode
    G.eval()

    with torch.no_grad():
        contents, _, _ = G.cnt_encoder(
            character_tensor.to(args.device))
    return contents


def infer_from_style_with_tensor(
        style_tensor,
        content_tensor,
        networks,
        args):
    # set nets
    G = networks['G']
    G_EMA = networks['G_EMA']

    # switch to train mode
    G.eval()
    G_EMA.eval()

    with torch.no_grad():
        styles = None
        style = style_tensor.to(args.device)
        style = style.unsqueeze(0)

        for _ in range(len(content_tensor)):
            if styles is None:
                styles = style.clone()
            else:
                styles = torch.cat((styles, style), 0)

        content_tensor = content_tensor.to(args.device)

        contents, skip1, skip2 = G.cnt_encoder(content_tensor)
        x_fake, _ = G.decode(
            contents, styles, skip1, skip2)
    return x_fake


def validateUN(full_dataset, networks, args, epoch=999):
    # set nets
    D = networks['D']
    G = networks['G']
    C = networks['C']
    C_EMA = networks['C_EMA']
    G_EMA = networks['G_EMA']
    # switch to train mode
    D.eval()
    G.eval()
    C.eval()
    C_EMA.eval()
    G_EMA.eval()
    # data loader
    val_dataset = full_dataset

    x_each_cls = []
    with torch.no_grad():
        val_tot_tars = torch.tensor(val_dataset.targets)
        for cls_idx in range(len(args.att_to_use)):
            tmp_cls_set = (val_tot_tars ==
                           args.att_to_use[cls_idx]).nonzero()[-args.val_num:]
            tmp_ds = torch.utils.data.Subset(val_dataset, tmp_cls_set)
            tmp_dl = torch.utils.data.DataLoader(
                tmp_ds,
                batch_size=args.val_num,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=False)
            tmp_iter = iter(tmp_dl)
            tmp_sample = None
            for sample_idx in range(len(tmp_iter)):
                imgs, _ = next(tmp_iter)
                x_ = imgs
                if tmp_sample is None:
                    tmp_sample = x_.clone()
                else:
                    tmp_sample = torch.cat((tmp_sample, x_), 0)
            x_each_cls.append(tmp_sample)

    if epoch >= args.fid_start:
        # Reference guided
        with torch.no_grad():
           # Just a buffer image ( to make a grid )
            ones = torch.ones(
                1,
                x_each_cls[0].size(1),
                x_each_cls[0].size(2),
                x_each_cls[0].size(3)).to(args.device)
            for src_idx in range(len(args.att_to_use)):
                x_src = x_each_cls[src_idx][:args.val_batch,
                                            :, :, :].to(args.device)
                rnd_idx = torch.randperm(x_each_cls[src_idx].size(0))[
                    :args.val_batch]
                x_src_rnd = x_each_cls[src_idx][rnd_idx].to(args.device)
                for ref_idx in range(len(args.att_to_use)):
                    x_res_ema = torch.cat((ones, x_src), 0)
                    x_rnd_ema = torch.cat((ones, x_src_rnd), 0)
                    x_ref = x_each_cls[ref_idx][:args.val_batch, :, :, :].to(
                        args.device)
                    rnd_idx = torch.randperm(x_each_cls[ref_idx].size(0))[
                        :args.val_batch]
                    x_ref_rnd = x_each_cls[ref_idx][rnd_idx].to(args.device)
                    for sample_idx in range(args.val_batch):
                        x_ref_tmp = x_ref[sample_idx: sample_idx +
                                          1].repeat((args.val_batch, 1, 1, 1))

                        c_src, skip1, skip2 = G_EMA.cnt_encoder(x_src)
                        s_ref = C_EMA(x_ref_tmp, sty=True)
                        x_res_ema_tmp, _ = G_EMA.decode(
                            c_src, s_ref, skip1, skip2)

                        x_ref_tmp = x_ref_rnd[sample_idx: sample_idx +
                                              1].repeat((args.val_batch, 1, 1, 1))

                        c_src, skip1, skip2 = G_EMA.cnt_encoder(x_src_rnd)
                        s_ref = C_EMA(x_ref_tmp, sty=True)
                        x_rnd_ema_tmp, _ = G_EMA.decode(
                            c_src, s_ref, skip1, skip2)

                        x_res_ema_tmp = torch.cat(
                            (x_ref[sample_idx: sample_idx + 1], x_res_ema_tmp), 0)
                        x_res_ema = torch.cat((x_res_ema, x_res_ema_tmp), 0)

                        x_rnd_ema_tmp = torch.cat(
                            (x_ref_rnd[sample_idx: sample_idx + 1], x_rnd_ema_tmp), 0)
                        x_rnd_ema = torch.cat((x_rnd_ema, x_rnd_ema_tmp), 0)

                    vutils.save_image(
                        x_res_ema,
                        os.path.join(
                            args.res_dir,
                            '{}_EMA_{}_{}{}.jpg'.format(
                                args.gpu,
                                epoch + 1,
                                src_idx,
                                ref_idx)),
                        normalize=True,
                        nrow=(
                            x_res_ema.size(0) // (
                                x_src.size(0) + 2) + 1))
                    vutils.save_image(
                        x_rnd_ema,
                        os.path.join(
                            args.res_dir,
                            '{}_RNDEMA_{}_{}{}.jpg'.format(
                                args.gpu,
                                epoch + 1,
                                src_idx,
                                ref_idx)),
                        normalize=True,
                        nrow=(
                            x_res_ema.size(0) // (
                                x_src.size(0) + 2) + 1))
