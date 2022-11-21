from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import numpy as np
import pathlib
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Obtaining characters from .ttf')
parser.add_argument(
    '--ttf_path',
    type=str,
    default='../ttf_folder',
    help='ttf directory')
parser.add_argument(
    '--chara',
    type=str,
    default='../chara.txt',
    help='characters')
parser.add_argument(
    '--save_path',
    type=str,
    default='../save_folder',
    help='images directory')
parser.add_argument(
    '--img_size',
    type=int,
    default=80,
    help='The size of generated images')
parser.add_argument(
    '--chara_size',
    type=int,
    default=70,
    help='The size of generated characters')
parser.add_argument(
    '--start_font',
    type=int,
    default=0,
)
args = parser.parse_args()

file_object = open(args.chara, encoding='utf-8')
try:
    characters = file_object.read()
finally:
    file_object.close()

print('Characters: ', characters[92:])


def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    draw.text((canvas_size // 2, canvas_size // 2), ch,
              (0, 0, 0), font=font, anchor='mm')
    return img


_img = Image.new("RGB", (args.img_size, args.img_size), (255, 255, 255))
white_space_hashes = [hash(_img.tobytes())]


def draw_example(ch, src_font, canvas_size, x_offset, y_offset):
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    dst_hash = hash(src_img.tobytes())
    if dst_hash in white_space_hashes:
        return None
    return src_img


data_dir = args.ttf_path
data_root = pathlib.Path(data_dir)
print(f'Font Data Root: {data_root}')

all_image_paths = list(data_root.glob('*.*tf*'))[args.start_font:]
all_image_paths = sorted([str(path) for path in all_image_paths])
print(f'{len(all_image_paths)} fonts are found.')
# for i in range(len(all_image_paths)):
#     # print(all_image_paths[i])
#     if 'JP_NotoSerifJP-Regular.otf' in all_image_paths[i]:
#         print(i)


print(len(all_image_paths))

seq = list()
for (label, item) in enumerate(tqdm(all_image_paths)):
    label += args.start_font
    src_font = ImageFont.truetype(item, size=args.chara_size)
    for (cnt, chara) in enumerate(characters[92:]):
        img = draw_example(
            chara,
            src_font,
            args.img_size,
            (args.img_size - args.chara_size) / 2,
            (args.img_size - args.chara_size) / 2)
        if img is not None:
            path_full = os.path.join(args.save_path, 'id_%d' % label)
            if not os.path.exists(path_full):
                os.mkdir(path_full)
            img.save(os.path.join(path_full, "%04d.png" % (cnt)))
