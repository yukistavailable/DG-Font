import torch
from torchvision.datasets import ImageFolder
import os
import torchvision.transforms as transforms
from datasets.custom_dataset import ImageFolderRemap, CrossdomainFolder, DatasetImages, ImageFolderRemapStyleAttraction


class Compose(object):
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, img):
        for t in self.tf:
            img = t(img)
        return img


def get_dataset(args):

    if args.input_ch == 1:
        mean = [0.5]
        std = [0.5]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    normalize = transforms.Normalize(mean=mean, std=std)

    transform = Compose([transforms.Resize((args.img_size, args.img_size)),
                         transforms.ToTensor(),
                         normalize])

    class_to_use = args.att_to_use

    print('USE CLASSES', class_to_use)

    # remap labels
    remap_table = {}
    i = 0
    for k in class_to_use:
        remap_table[k] = i
        i += 1

    print("LABEL MAP:", remap_table)

    img_dir = args.data_dir

    if args.style_attraction:
        dataset = ImageFolderRemapStyleAttraction(
            img_dir,
            transform=transform,
            remap_table=remap_table,
            input_ch=args.input_ch,
            batch_size=args.batch_size / args.font_num_per_batch,

        )
    else:
        dataset = ImageFolderRemap(
            img_dir,
            transform=transform,
            remap_table=remap_table,
            input_ch=args.input_ch
        )

    tot_targets = torch.tensor(dataset.targets)

    # valdataset = ImageFolderRemap(
    #     img_dir,
    #     transform=transform_val,
    #     remap_table=remap_table,
    #     input_ch=args.input_ch
    # )
    # parse classes to use

    # min_data = 99999999
    # max_data = 0

    # train_idx = None
    # val_idx = None
    # for k in class_to_use:
    #     tmp_idx = (tot_targets == k).nonzero()
    #     train_tmp_idx = tmp_idx[:-args.val_num]
    #     val_tmp_idx = tmp_idx[-args.val_num:]
    #     if k == class_to_use[0]:
    #         train_idx = train_tmp_idx.clone()
    #         val_idx = val_tmp_idx.clone()
    #     else:
    #         train_idx = torch.cat((train_idx, train_tmp_idx))
    #         val_idx = torch.cat((val_idx, val_tmp_idx))
    #     if min_data > len(train_tmp_idx):
    #         min_data = len(train_tmp_idx)
    #     if max_data < len(train_tmp_idx):
    #         max_data = len(train_tmp_idx)
    #
    # train_dataset = torch.utils.data.Subset(dataset, train_idx)
    # val_dataset = torch.utils.data.Subset(valdataset, val_idx)
    #
    # args.min_data = min_data
    # args.max_data = max_data
    # print("MINIMUM TRAIN DATA FONT :", args.min_data)
    # print("MAXIMUM TAIN DATA FONT :", args.max_data)

    train_dataset = dataset

    content_dataset = None

    if args.fixed_content_font:
        idx = None
        for k in args.content_font_id:
            tmp_idx = (tot_targets == k).nonzero()
            tmp_idx = tmp_idx[:-args.val_num]
            if idx is None:
                idx = tmp_idx.clone()
            else:
                idx = torch.cat((idx, tmp_idx))

        tmp_dataset = ImageFolderRemap(
            img_dir,
            transform=transform,
            remap_table=remap_table,
            input_ch=args.input_ch
        )
        content_dataset = torch.utils.data.Subset(tmp_dataset, idx)
        content_dataset = content_dataset

    return train_dataset, content_dataset


def get_dataset_for_inference(args, img_paths):

    if args.input_ch == 1:
        mean = [0.5]
        std = [0.5]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    normalize = transforms.Normalize(mean=mean, std=std)

    transform = Compose([transforms.Resize((args.img_size, args.img_size)),
                         transforms.ToTensor(),
                         normalize])

    dataset = DatasetImages(
        img_paths,
        transform=transform,
        input_ch=args.input_ch
    )

    return dataset
