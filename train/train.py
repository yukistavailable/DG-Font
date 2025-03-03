from tqdm import trange
from PIL import Image
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from tools.utils import *
from tools.ops import compute_grad_gp, update_average, copy_norm_params, queue_data, dequeue_data, \
    average_gradients, calc_adv_loss, calc_contrastive_loss, calc_recon_loss, calc_style_norm, calc_variance, \
    calc_content_norm

def my_transform(n_px=224):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def preprocess_for_tensor(x):
    return my_transform()(x)



def trainGAN_with_CLIP(
   style_data_loader,
   content_data_loader,
    networks,
    opts,
    epoch, 
    args,
    additional):

    # avg meter
    d_losses = AverageMeter()
    d_advs = AverageMeter()
    d_gps = AverageMeter()

    g_losses = AverageMeter()
    g_advs = AverageMeter()
    g_imgrecs = AverageMeter()
    g_rec = AverageMeter()

    # set nets
    D = networks['D']
    G = networks['G']
    G_EMA = networks['G_EMA']

    # set opts
    d_opt = opts['D']
    g_opt = opts['G']

    # switch to train mode
    D.train()
    G.train()
    G_EMA.train()

    if args.use_linear_block:
        L = networks['L']
        l_opt = opts['L']
        L.train()

    logger = additional['logger']

    # summary writer
    style_train_it = iter(style_data_loader)
    # content_train_it = iter(content_data_loader)

    t_train = trange(0, args.iters, initial=0, total=args.iters)

    for i in t_train:
        try:
            embedded_image, style_for_discriminator, style_y = next(style_train_it)
            if len(embedded_image) < args.batch_size:
                style_train_it = iter(style_data_loader)
                embedded_image, style_for_discriminator, style_y = next(style_train_it)
        except BaseException:
            style_train_it = iter(style_data_loader)
            embedded_image, style_for_discriminator, style_y = next(style_train_it)

        # try:
        #     contents, content_y, cnt_cnt_idx = next(content_train_it)
        #     if len(contents) < args.batch_size:
        #         content_train_it = iter(content_data_loader)
        #         contents, content_y, cnt_cnt_idx = next(content_train_it)
        # except BaseException:
        #     content_train_it = iter(content_data_loader)
        #     contents, content_y, cnt_cnt_idx = next(content_train_it)

        try:
            content_embedded_image, contents, content_y = next(style_train_it)
            if len(content_embedded_image) < args.batch_size:
                style_train_it = iter(style_data_loader)
                content_embedded_image, contents, content_y = next(style_train_it)
        except BaseException:
            style_train_it = iter(style_data_loader)
            content_embedded_image, contents, content_y = next(style_train_it)

        # imgs.shape is [batch_size, input_ch, img_size, img_size]
        # y_org is [class_idx, class_idx, ..., class_idx] and the length is
        # batch_size

        x_org = contents
        y_org = content_y
        # x_org_image = [to_pil_image(x) for x in x_org]
        # x_org_image_tensor = torch.stack([clip_preprocess(x) for x in x_org_image]).to(args.device)

        x_org = x_org.to(args.device)
        y_org = y_org.to(args.device)

        embedded_image = embedded_image.to(args.device)
        if args.use_linear_block:
            x_ref = L(embedded_image)
        else:
            x_ref = embedded_image
        y_ref = style_y
        y_ref = y_ref.to(args.device)
        style_for_discriminator = style_for_discriminator.to(args.device)

        training_mode = 'GAN'

        ####################
        # BEGIN Train GANs #
        ####################
        with torch.no_grad():
            # style
            s_ref = x_ref

            # content
            c_src, skip1, skip2 = G.cnt_encoder(x_org)

            x_fake, _ = G.decode(c_src, s_ref, skip1, skip2)

        # - x: images of shape (batch, 3, image_size, image_size).
        # - y: domain indices of shape (batch).
        d_real_logit, _ = D(style_for_discriminator, y_ref)
        d_fake_logit, _ = D(x_fake.detach(), y_ref)

        d_adv_real = calc_adv_loss(d_real_logit, 'd_real')
        d_adv_fake = calc_adv_loss(d_fake_logit, 'd_fake')

        d_adv = d_adv_real + d_adv_fake

        d_gp = 0
        # d_gp = args.w_gp * compute_grad_gp(d_real_logit, x_ref, is_patch=False)

        d_loss = d_adv + d_gp

        d_opt.zero_grad()
        d_adv_real.backward(retain_graph=True)
        # d_gp.backward()
        d_adv_fake.backward()
        # if args.distributed:
        #     average_gradients(D)
        d_opt.step()

        # torch.cuda.empty_cache()

        # Train G
        # s_src = clip_model_visual(x_org_image_tensor.to(torch.float16))
        # s_src = s_src.to(torch.float32)
        content_embedded_image = content_embedded_image.to(args.device)
        if args.use_linear_block:
            s_src = L(content_embedded_image)
        else:
            s_src = content_embedded_image

        c_src, skip1, skip2 = G.cnt_encoder(x_org)
        x_fake, offset_loss = G.decode(c_src, s_ref, skip1, skip2)
        # x_fake_image = [to_pil_image(x) for x in x_fake]
        # x_fake_image_tensor = torch.stack([clip_preprocess(x) for x in x_fake_image]).to(args.device)
        x_rec, _ = G.decode(c_src, s_src, skip1, skip2)

        g_fake_logit, _ = D(x_fake, y_ref)
        g_rec_logit, _ = D(x_rec, y_org)

        g_adv_fake = calc_adv_loss(g_fake_logit, 'g')
        g_adv_rec = calc_adv_loss(g_rec_logit, 'g')

        g_adv = g_adv_fake + g_adv_rec

        g_imgrec = calc_recon_loss(x_rec, x_org)

        c_x_fake, _, _ = G.cnt_encoder(x_fake)
        g_conrec = calc_recon_loss(c_x_fake, c_src)

        # style_x_fake = clip_model_visual(x_fake_image_tensor.to(torch.float16))
        # style_x_fake = style_x_fake.to(torch.float32)
        # g_styrec = calc_recon_loss(style_x_fake, s_ref)

        g_loss = args.w_adv * g_adv + args.w_rec * g_imgrec + args.w_rec * \
            g_conrec + args.w_off * offset_loss

        g_opt.zero_grad()
        if args.use_linear_block:
            l_opt.zero_grad()
        g_loss.backward()

        g_opt.step()
        if args.use_linear_block:
            l_opt.step()

        ##################
        # END Train GANs #
        ##################

        if epoch >= args.ema_start:
            training_mode = training_mode + "_EMA"
            update_average(G_EMA, G)

        torch.cuda.synchronize()

        with torch.no_grad():
            if epoch >= args.separated:
                d_losses.update(d_loss.item(), x_org.size(0))
                d_advs.update(d_adv.item(), x_org.size(0))
                # d_gps.update(d_gp.item(), x_org.size(0))

                g_losses.update(g_loss.item(), x_org.size(0))
                g_advs.update(g_adv.item(), x_org.size(0))
                g_imgrecs.update(g_imgrec.item(), x_org.size(0))
                g_rec.update(g_conrec.item(), x_org.size(0))

            if (i + 1) % args.log_step == 0 and (args.gpu == 0 or args.gpu == '0'):
                summary_step = epoch * args.iters + i
                add_logs(args, logger, 'D/LOSS', d_losses.avg, summary_step)
                add_logs(args, logger, 'D/ADV', d_advs.avg, summary_step)
                # add_logs(args, logger, 'D/GP', d_gps.avg, summary_step)

                add_logs(args, logger, 'G/LOSS', g_losses.avg, summary_step)
                add_logs(args, logger, 'G/ADV', g_advs.avg, summary_step)
                add_logs(args, logger, 'G/IMGREC', g_imgrecs.avg, summary_step)
                add_logs(args, logger, 'G/conrec', g_rec.avg, summary_step)

                print('Epoch: [{}/{}] [{}/{}] MODE[{}] Avg Loss: D[{d_losses.avg:.2f}] G[{g_losses.avg:.2f}] '.format(
                    epoch + 1, args.epochs, i + 1, args.iters, training_mode, d_losses=d_losses, g_losses=g_losses))

    copy_norm_params(G_EMA, G)


def trainGAN(data_loader, networks, opts, epoch, args, additional):
    # avg meter
    d_losses = AverageMeter()
    d_advs = AverageMeter()
    d_gps = AverageMeter()

    g_losses = AverageMeter()
    g_advs = AverageMeter()
    g_imgrecs = AverageMeter()
    g_rec = AverageMeter()

    moco_losses = AverageMeter()

    # set nets
    D = networks['D']
    G = networks['G']
    C = networks['C']
    G_EMA = networks['G_EMA']
    C_EMA = networks['C_EMA']

    # set opts
    d_opt = opts['D']
    g_opt = opts['G']
    c_opt = opts['C']

    # switch to train mode
    D.train()
    G.train()
    C.train()
    C_EMA.train()
    G_EMA.train()

    is_cd = args.content_discriminator
    if is_cd:
        Cd = networks['CD']
        cd_opt = opts['CD']
        Cd.train()

    logger = additional['logger']

    # summary writer
    train_it = iter(data_loader)

    t_train = trange(0, args.iters, initial=0, total=args.iters)

    for i in t_train:
        try:
            imgs, y_org, cnt_idx = next(train_it)
        except BaseException:
            train_it = iter(data_loader)
            imgs, y_org, cnt_idx = next(train_it)

        # imgs.shape is [batch_size, 3, img_size, img_size]
        # y_org is [class_idx, class_idx, ..., class_idx] and the length is
        # batch_size

        x_org = imgs

        # [0, 1, 2, ..., batch_size]をランダムに入れ替えたもの
        # [6, 2, 7, 4, 1, 3, 5, 0]
        x_ref_idx = torch.randperm(x_org.size(0))

        x_org = x_org.to(args.device)
        y_org = y_org.to(args.device)

        # x_ref はx_orgをランダムに入れ替えたもの
        # x_ref is characters with the target font
        x_ref = x_org.clone()
        x_ref = x_ref[x_ref_idx]
        cnt_idx_ref = cnt_idx.clone()
        cnt_idx_ref = cnt_idx_ref[x_ref_idx]

        training_mode = 'GAN'

        ####################
        # BEGIN Train GANs #
        ####################
        with torch.no_grad():
            y_ref = y_org.clone()
            y_ref = y_ref[x_ref_idx]

            # style
            s_ref = C.moco(x_ref)

            # content
            c_src, skip1, skip2 = G.cnt_encoder(x_org)

            x_fake, _ = G.decode(c_src, s_ref, skip1, skip2)

        x_ref.requires_grad_()

        # - x: images of shape (batch, 3, image_size, image_size).
        # - y: domain indices of shape (batch).
        d_real_logit, _ = D(x_ref, y_ref)
        d_fake_logit, _ = D(x_fake.detach(), y_ref)

        d_adv_real = calc_adv_loss(d_real_logit, 'd_real')
        d_adv_fake = calc_adv_loss(d_fake_logit, 'd_fake')

        d_adv = d_adv_real + d_adv_fake

        d_gp = args.w_gp * compute_grad_gp(d_real_logit, x_ref, is_patch=False)

        d_loss = d_adv + d_gp

        d_opt.zero_grad()
        d_adv_real.backward(retain_graph=True)
        d_gp.backward()
        d_adv_fake.backward()
        # if args.distributed:
        #     average_gradients(D)
        d_opt.step()

        # Train G
        s_src = C.moco(x_org)
        s_ref = C.moco(x_ref)

        c_src, skip1, skip2 = G.cnt_encoder(x_org)
        x_fake, offset_loss = G.decode(c_src, s_ref, skip1, skip2)
        x_rec, _ = G.decode(c_src, s_src, skip1, skip2)

        g_fake_logit, _ = D(x_fake, y_ref)
        g_rec_logit, _ = D(x_rec, y_org)

        g_adv_fake = calc_adv_loss(g_fake_logit, 'g')
        g_adv_rec = calc_adv_loss(g_rec_logit, 'g')

        g_adv = g_adv_fake + g_adv_rec

        g_imgrec = calc_recon_loss(x_rec, x_org)

        c_x_fake, _, _ = G.cnt_encoder(x_fake)
        g_conrec = calc_recon_loss(c_x_fake, c_src)

        style_x_fake = C.moco(x_fake)
        g_styrec = calc_recon_loss(style_x_fake, s_ref)

        # Train Content Discriminator
        cd_loss = 0
        if is_cd:
            c_ref_src, _, _ = G.cnt_encoder(x_ref)
            _, c_sty_cnt_logit = Cd(c_ref_src, cnt_idx_ref)
            _, c_cnt_logit = Cd(c_src, cnt_idx)
            _, c_sty_logit = Cd(c_x_fake, cnt_idx)
            cd_loss = calc_adv_loss(c_sty_cnt_logit, 'g') + \
                calc_adv_loss(c_cnt_logit, 'g') + \
                calc_adv_loss(c_sty_logit, 'g')

        g_loss = args.w_adv * g_adv + args.w_rec * g_imgrec + args.w_rec * g_conrec + \
            args.w_off * offset_loss + args.w_sty * g_styrec + args.w_cd * cd_loss

        g_opt.zero_grad()
        c_opt.zero_grad()
        if is_cd:
            cd_opt.zero_grad()
        g_loss.backward()
        # if args.distributed:
        #     average_gradients(G)
        #     average_gradients(C)
        c_opt.step()
        g_opt.step()
        if is_cd:
            cd_opt.step()

        ##################
        # END Train GANs #
        ##################

        if epoch >= args.ema_start:
            training_mode = training_mode + "_EMA"
            update_average(G_EMA, G)
        update_average(C_EMA, C)

        torch.cuda.synchronize()

        with torch.no_grad():
            if epoch >= args.separated:
                d_losses.update(d_loss.item(), x_org.size(0))
                d_advs.update(d_adv.item(), x_org.size(0))
                d_gps.update(d_gp.item(), x_org.size(0))

                g_losses.update(g_loss.item(), x_org.size(0))
                g_advs.update(g_adv.item(), x_org.size(0))
                g_imgrecs.update(g_imgrec.item(), x_org.size(0))
                g_rec.update(g_conrec.item(), x_org.size(0))

                moco_losses.update(offset_loss.item(), x_org.size(0))

            if (i + 1) % args.log_step == 0 and (args.gpu == 0 or args.gpu == '0'):
                summary_step = epoch * args.iters + i
                add_logs(args, logger, 'D/LOSS', d_losses.avg, summary_step)
                add_logs(args, logger, 'D/ADV', d_advs.avg, summary_step)
                add_logs(args, logger, 'D/GP', d_gps.avg, summary_step)

                add_logs(args, logger, 'G/LOSS', g_losses.avg, summary_step)
                add_logs(args, logger, 'G/ADV', g_advs.avg, summary_step)
                add_logs(args, logger, 'G/IMGREC', g_imgrecs.avg, summary_step)
                add_logs(args, logger, 'G/conrec', g_rec.avg, summary_step)

                add_logs(
                    args,
                    logger,
                    'C/OFFSET',
                    moco_losses.avg,
                    summary_step)

                print('Epoch: [{}/{}] [{}/{}] MODE[{}] Avg Loss: D[{d_losses.avg:.2f}] G[{g_losses.avg:.2f}] '.format(
                    epoch + 1, args.epochs, i + 1, args.iters, training_mode, d_losses=d_losses, g_losses=g_losses))

    copy_norm_params(G_EMA, G)
    copy_norm_params(C_EMA, C)


def train_fixed_content(
        style_data_loader,
        content_data_loader,
        networks,
        opts,
        epoch,
        args,
        additional):
    # avg meter
    d_losses = AverageMeter()
    d_advs = AverageMeter()
    d_gps = AverageMeter()

    g_losses = AverageMeter()
    g_advs = AverageMeter()
    g_imgrecs = AverageMeter()
    g_rec = AverageMeter()

    moco_losses = AverageMeter()

    # set nets
    D = networks['D']
    G = networks['G']
    C = networks['C']
    G_EMA = networks['G_EMA']
    C_EMA = networks['C_EMA']

    # set opts
    d_opt = opts['D']
    g_opt = opts['G']
    c_opt = opts['C']

    # switch to train mode
    D.train()
    G.train()
    C.train()
    C_EMA.train()
    G_EMA.train()

    is_cd = args.content_discriminator
    if is_cd:
        Cd = networks['CD']
        cd_opt = opts['CD']
        Cd.train()

    logger = additional['logger']

    style_norm_count = 0
    style_norm_amount = 0
    style_variance_amount = 0
    content_norm_count = 0
    content_norm_amount = 0

    # summary writer
    style_train_it = iter(style_data_loader)
    content_train_it = iter(content_data_loader)

    t_train = trange(0, args.iters, initial=0, total=args.iters)

    for i in t_train:
        try:
            styles, style_y, sty_cnt_idx = next(style_train_it)
            if len(styles) < args.batch_size:
                style_train_it = iter(style_data_loader)
                styles, style_y, sty_cnt_idx = next(style_train_it)
        except BaseException:
            style_train_it = iter(style_data_loader)
            styles, style_y, sty_cnt_idx = next(style_train_it)

        if args.content_norm:
            try:
                contents, content_y, cnt_cnt_idx = next(content_train_it)
            except BaseException:
                content_train_it = iter(content_data_loader)
                contents, content_y, cnt_cnt_idx = next(content_train_it)
            assert len(contents) == 1
            contents = torch.cat([contents.clone()] * args.batch_size, dim=0)
            content_y = torch.cat([content_y.clone()] * args.batch_size, dim=0)
        else:
            try:
                contents, content_y, cnt_cnt_idx = next(content_train_it)
                if len(contents) < args.batch_size:
                    content_train_it = iter(content_data_loader)
                    contents, content_y, cnt_cnt_idx = next(content_train_it)
            except BaseException:
                content_train_it = iter(content_data_loader)
                contents, content_y, cnt_cnt_idx = next(content_train_it)

        # imgs.shape is [batch_size, input_ch, img_size, img_size]
        # y_org is [class_idx, class_idx, ..., class_idx] and the length is
        # batch_size

        x_org = contents
        y_org = content_y

        x_org = x_org.to(args.device)
        y_org = y_org.to(args.device)

        x_ref = styles
        y_ref = style_y
        x_ref = x_ref.to(args.device)
        y_ref = y_ref.to(args.device)

        training_mode = 'GAN'

        ####################
        # BEGIN Train GANs #
        ####################
        with torch.no_grad():
            # style
            s_ref = C.moco(x_ref)

            # content
            c_src, skip1, skip2 = G.cnt_encoder(x_org)

            x_fake, _ = G.decode(c_src, s_ref, skip1, skip2)

        x_ref.requires_grad_()

        # - x: images of shape (batch, 3, image_size, image_size).
        # - y: domain indices of shape (batch).
        d_real_logit, _ = D(x_ref, y_ref)
        d_fake_logit, _ = D(x_fake.detach(), y_ref)

        d_adv_real = calc_adv_loss(d_real_logit, 'd_real')
        d_adv_fake = calc_adv_loss(d_fake_logit, 'd_fake')

        d_adv = d_adv_real + d_adv_fake

        d_gp = args.w_gp * compute_grad_gp(d_real_logit, x_ref, is_patch=False)

        d_loss = d_adv + d_gp

        d_opt.zero_grad()
        d_adv_real.backward(retain_graph=True)
        d_gp.backward()
        d_adv_fake.backward()
        # if args.distributed:
        #     average_gradients(D)
        d_opt.step()

        # Train G
        s_src = C.moco(x_org)
        s_ref = C.moco(x_ref)

        c_src, skip1, skip2 = G.cnt_encoder(x_org)
        x_fake, offset_loss = G.decode(c_src, s_ref, skip1, skip2)
        x_rec, _ = G.decode(c_src, s_src, skip1, skip2)

        g_fake_logit, _ = D(x_fake, y_ref)
        g_rec_logit, _ = D(x_rec, y_org)

        g_adv_fake = calc_adv_loss(g_fake_logit, 'g')
        g_adv_rec = calc_adv_loss(g_rec_logit, 'g')

        g_adv = g_adv_fake + 0.01 * g_adv_rec

        g_imgrec = calc_recon_loss(x_rec, x_org)

        c_x_fake, _, _ = G.cnt_encoder(x_fake)
        g_conrec = calc_recon_loss(c_x_fake, c_src)

        style_x_fake = C.moco(x_fake)
        g_styrec = calc_recon_loss(style_x_fake, s_ref)

        g_style_norm = 0
        g_style_var = 0
        if args.style_norm:
            if style_y.eq(style_y[0]).all():
                g_style_norm = calc_style_norm(s_ref)
                style_norm_count += 1
                style_norm_amount += g_style_norm.item()
                g_style_var = calc_variance(s_ref)
                # print(
                # f'Style Norm Loss: {style_norm_amount / style_norm_count}')

        g_content_norm = 0
        if args.content_norm:
            if content_y.eq(content_y[0]).all():
                b, c, h, w = c_src.shape
                tmp_c_src = c_src.view(b, c * h * w)
                g_content_norm = calc_content_norm(tmp_c_src)
                content_norm_count += 1
                content_norm_amount += g_content_norm.item()

        # Train Content Discriminator
        cd_loss = 0
        if is_cd:
            c_ref_src, _, _ = G.cnt_encoder(x_ref)
            _, c_sty_cnt_logit = Cd(c_ref_src, sty_cnt_idx)
            _, c_cnt_logit = Cd(c_src, cnt_cnt_idx)
            _, c_sty_logit = Cd(c_x_fake, cnt_cnt_idx)
            cd_loss = calc_adv_loss(c_sty_cnt_logit, 'g') + \
                calc_adv_loss(c_cnt_logit, 'g') + \
                calc_adv_loss(c_sty_logit, 'g')
            # cd_loss = calc_adv_loss(c_sty_cnt_logit, 'g') + \
            #           calc_adv_loss(c_cnt_logit, 'g')

        g_loss = args.w_adv * g_adv + args.w_rec * g_imgrec + args.w_rec * \
            g_conrec + args.w_off * offset_loss + args.w_rec * g_styrec + \
            args.w_sty_norm * g_style_norm + args.w_cnt_norm * g_content_norm + \
            args.w_sty_var * g_style_var + args.w_cd * cd_loss

        g_opt.zero_grad()
        c_opt.zero_grad()
        if is_cd:
            cd_opt.zero_grad()
        g_loss.backward()
        # if args.distributed:
        #     average_gradients(G)
        #     average_gradients(C)
        c_opt.step()
        g_opt.step()
        if is_cd:
            cd_opt.step()

        ##################
        # END Train GANs #
        ##################

        if epoch >= args.ema_start:
            training_mode = training_mode + "_EMA"
            update_average(G_EMA, G)
        update_average(C_EMA, C)

        torch.cuda.synchronize()

        with torch.no_grad():
            if epoch >= args.separated:
                d_losses.update(d_loss.item(), x_org.size(0))
                d_advs.update(d_adv.item(), x_org.size(0))
                d_gps.update(d_gp.item(), x_org.size(0))

                g_losses.update(g_loss.item(), x_org.size(0))
                g_advs.update(g_adv.item(), x_org.size(0))
                g_imgrecs.update(g_imgrec.item(), x_org.size(0))
                g_rec.update(g_conrec.item(), x_org.size(0))

                moco_losses.update(offset_loss.item(), x_org.size(0))

            if (i + 1) % args.log_step == 0 and (args.gpu == 0 or args.gpu == '0'):
                summary_step = epoch * args.iters + i
                add_logs(args, logger, 'D/LOSS', d_losses.avg, summary_step)
                add_logs(args, logger, 'D/ADV', d_advs.avg, summary_step)
                add_logs(args, logger, 'D/GP', d_gps.avg, summary_step)

                add_logs(args, logger, 'G/LOSS', g_losses.avg, summary_step)
                add_logs(args, logger, 'G/ADV', g_advs.avg, summary_step)
                add_logs(args, logger, 'G/IMGREC', g_imgrecs.avg, summary_step)
                add_logs(args, logger, 'G/conrec', g_rec.avg, summary_step)

                add_logs(
                    args,
                    logger,
                    'C/OFFSET',
                    moco_losses.avg,
                    summary_step)

                print('Epoch: [{}/{}] [{}/{}] MODE[{}] Avg Loss: D[{d_losses.avg:.2f}] G[{g_losses.avg:.2f}] '.format(
                    epoch + 1, args.epochs, i + 1, args.iters, training_mode, d_losses=d_losses, g_losses=g_losses))

    if args.style_norm:
        print(f'Style Norm Loss: {style_norm_amount / style_norm_count}')
        print(f'Style Norm Loss: {style_variance_amount / style_norm_count}')
    if args.content_norm:
        print(f'Content Norm Loss: {content_norm_amount / content_norm_count}')
    copy_norm_params(G_EMA, G)
    copy_norm_params(C_EMA, C)


def train_fixed_content_with_style_attraction(
        style_data_loader,
        content_data_loader,
        networks,
        opts,
        epoch,
        args,
        additional):
    # avg meter
    d_losses = AverageMeter()
    d_advs = AverageMeter()
    d_gps = AverageMeter()

    g_losses = AverageMeter()
    g_advs = AverageMeter()
    g_imgrecs = AverageMeter()
    g_rec = AverageMeter()

    moco_losses = AverageMeter()

    # set nets
    D = networks['D']
    G = networks['G']
    C = networks['C']
    G_EMA = networks['G_EMA']
    C_EMA = networks['C_EMA']

    # set opts
    d_opt = opts['D']
    g_opt = opts['G']
    c_opt = opts['C']

    # switch to train mode
    D.train()
    G.train()
    C.train()
    C_EMA.train()
    G_EMA.train()

    is_cd = args.content_discriminator
    if is_cd:
        Cd = networks['CD']
        cd_opt = opts['CD']
        Cd.train()

    logger = additional['logger']

    style_norm_count = 0
    style_norm_amount = 0
    style_variance_amount = 0
    content_norm_count = 0
    content_norm_amount = 0

    # summary writer
    style_train_it = iter(style_data_loader)
    content_train_it = iter(content_data_loader)

    t_train = trange(0, args.iters, initial=0, total=args.iters)

    for i in t_train:
        try:
            styles, style_y, sty_cnt_idx = next(style_train_it)
            if styles[0].shape[0] < args.font_num_per_batch:
                style_train_it = iter(style_data_loader)
                styles, style_y, sty_cnt_idx = next(style_train_it)
        except BaseException:
            style_train_it = iter(style_data_loader)
            styles, style_y, sty_cnt_idx = next(style_train_it)
        styles = torch.cat(styles, dim=0)
        style_y = torch.cat(style_y, dim=0)
        sty_cnt_idx = torch.cat(sty_cnt_idx, dim=0)

        if args.content_norm:
            try:
                contents, content_y, cnt_cnt_idx = next(content_train_it)
            except BaseException:
                content_train_it = iter(content_data_loader)
                contents, content_y, cnt_cnt_idx = next(content_train_it)
            assert len(contents) == 1
            contents = torch.cat([contents.clone()] * args.batch_size, dim=0)
            content_y = torch.cat([content_y.clone()] * args.batch_size, dim=0)
        else:
            try:
                contents, content_y, cnt_cnt_idx = next(content_train_it)
                if len(contents) < args.batch_size:
                    content_train_it = iter(content_data_loader)
                    contents, content_y, cnt_cnt_idx = next(content_train_it)
            except BaseException:
                content_train_it = iter(content_data_loader)
                contents, content_y, cnt_cnt_idx = next(content_train_it)

        # imgs.shape is [batch_size, input_ch, img_size, img_size]
        # y_org is [class_idx, class_idx, ..., class_idx] and the length is
        # batch_size

        x_org = contents
        y_org = content_y

        x_org = x_org.to(args.device)
        y_org = y_org.to(args.device)

        x_ref = styles
        y_ref = style_y
        x_ref = x_ref.to(args.device)
        y_ref = y_ref.to(args.device)

        training_mode = 'GAN'

        ####################
        # BEGIN Train GANs #
        ####################
        with torch.no_grad():
            # style
            s_ref = C.moco(x_ref)

            # content
            c_src, skip1, skip2 = G.cnt_encoder(x_org)

            x_fake, _ = G.decode(c_src, s_ref, skip1, skip2)

        x_ref.requires_grad_()

        # - x: images of shape (batch, 3, image_size, image_size).
        # - y: domain indices of shape (batch).
        d_real_logit, _ = D(x_ref, y_ref)
        d_fake_logit, _ = D(x_fake.detach(), y_ref)

        d_adv_real = calc_adv_loss(d_real_logit, 'd_real')
        d_adv_fake = calc_adv_loss(d_fake_logit, 'd_fake')

        d_adv = d_adv_real + d_adv_fake

        d_gp = args.w_gp * compute_grad_gp(d_real_logit, x_ref, is_patch=False)

        d_loss = d_adv + d_gp

        d_opt.zero_grad()
        d_adv_real.backward(retain_graph=True)
        d_gp.backward()
        d_adv_fake.backward()
        # if args.distributed:
        #     average_gradients(D)
        d_opt.step()

        # Train G
        s_src = C.moco(x_org)
        s_ref = C.moco(x_ref)

        c_src, skip1, skip2 = G.cnt_encoder(x_org)
        x_fake, offset_loss = G.decode(c_src, s_ref, skip1, skip2)
        x_rec, _ = G.decode(c_src, s_src, skip1, skip2)

        g_fake_logit, _ = D(x_fake, y_ref)
        g_rec_logit, _ = D(x_rec, y_org)

        g_adv_fake = calc_adv_loss(g_fake_logit, 'g')
        g_adv_rec = calc_adv_loss(g_rec_logit, 'g')

        g_adv = g_adv_fake + 0.01 * g_adv_rec

        g_imgrec = calc_recon_loss(x_rec, x_org)

        c_x_fake, _, _ = G.cnt_encoder(x_fake)
        g_conrec = calc_recon_loss(c_x_fake, c_src)

        style_x_fake = C.moco(x_fake)
        g_styrec = calc_recon_loss(style_x_fake, s_ref)

        g_style_norm = 0
        g_style_var = 0
        if args.style_attraction:
            # if style_y.eq(style_y[0]).all():
            g_style_norm = calc_style_norm(s_ref)
            style_norm_count += 1
            style_norm_amount += g_style_norm.item()
            # g_style_var = calc_variance(s_ref)
            # print(
            # f'Style Norm Loss: {style_norm_amount / style_norm_count}')

        g_content_norm = 0
        if args.content_norm:
            if content_y.eq(content_y[0]).all():
                b, c, h, w = c_x_fake.shape
                tmp_c_fake = c_x_fake.view(b, c * h * w)
                g_content_norm = calc_content_norm(tmp_c_fake)
                content_norm_count += 1
                content_norm_amount += g_content_norm.item()

        # Train Content Discriminator
        cd_loss = 0
        if is_cd:
            c_ref_src, _, _ = G.cnt_encoder(x_ref)
            _, c_sty_cnt_logit = Cd(c_ref_src, sty_cnt_idx)
            _, c_cnt_logit = Cd(c_src, cnt_cnt_idx)
            _, c_sty_logit = Cd(c_x_fake, cnt_cnt_idx)
            cd_loss = calc_adv_loss(c_sty_cnt_logit, 'g') + \
                calc_adv_loss(c_cnt_logit, 'g') + \
                calc_adv_loss(c_sty_logit, 'g')
            # cd_loss = calc_adv_loss(c_sty_cnt_logit, 'g') + \
            #           calc_adv_loss(c_cnt_logit, 'g')

        g_loss = args.w_adv * g_adv + args.w_rec * g_imgrec + args.w_rec * \
            g_conrec + args.w_off * offset_loss + args.w_rec * g_styrec + \
            args.w_sty_norm * g_style_norm + args.w_cnt_norm * g_content_norm + \
            args.w_sty_var * g_style_var + args.w_cd * cd_loss

        g_opt.zero_grad()
        c_opt.zero_grad()
        if is_cd:
            cd_opt.zero_grad()
        g_loss.backward()
        # if args.distributed:
        #     average_gradients(G)
        #     average_gradients(C)
        c_opt.step()
        g_opt.step()
        if is_cd:
            cd_opt.step()

        ##################
        # END Train GANs #
        ##################

        if epoch >= args.ema_start:
            training_mode = training_mode + "_EMA"
            update_average(G_EMA, G)
        update_average(C_EMA, C)

        torch.cuda.synchronize()

        with torch.no_grad():
            if epoch >= args.separated:
                d_losses.update(d_loss.item(), x_org.size(0))
                d_advs.update(d_adv.item(), x_org.size(0))
                d_gps.update(d_gp.item(), x_org.size(0))

                g_losses.update(g_loss.item(), x_org.size(0))
                g_advs.update(g_adv.item(), x_org.size(0))
                g_imgrecs.update(g_imgrec.item(), x_org.size(0))
                g_rec.update(g_conrec.item(), x_org.size(0))

                moco_losses.update(offset_loss.item(), x_org.size(0))

            if (i + 1) % args.log_step == 0 and (args.gpu == 0 or args.gpu == '0'):
                summary_step = epoch * args.iters + i
                add_logs(args, logger, 'D/LOSS', d_losses.avg, summary_step)
                add_logs(args, logger, 'D/ADV', d_advs.avg, summary_step)
                add_logs(args, logger, 'D/GP', d_gps.avg, summary_step)

                add_logs(args, logger, 'G/LOSS', g_losses.avg, summary_step)
                add_logs(args, logger, 'G/ADV', g_advs.avg, summary_step)
                add_logs(args, logger, 'G/IMGREC', g_imgrecs.avg, summary_step)
                add_logs(args, logger, 'G/conrec', g_rec.avg, summary_step)

                add_logs(
                    args,
                    logger,
                    'C/OFFSET',
                    moco_losses.avg,
                    summary_step)

                print('Epoch: [{}/{}] [{}/{}] MODE[{}] Avg Loss: D[{d_losses.avg:.2f}] G[{g_losses.avg:.2f}] '.format(
                    epoch + 1, args.epochs, i + 1, args.iters, training_mode, d_losses=d_losses, g_losses=g_losses))

    if args.style_norm:
        print(f'Style Norm Loss: {style_norm_amount / style_norm_count}')
        print(f'Style Norm Loss: {style_variance_amount / style_norm_count}')
    if args.content_norm:
        print(f'Content Norm Loss: {content_norm_amount / content_norm_count}')
    copy_norm_params(G_EMA, G)
    copy_norm_params(C_EMA, C)
