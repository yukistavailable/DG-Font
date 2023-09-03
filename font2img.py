from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import numpy as np
import pathlib
import argparse
from tqdm import tqdm
import shutil


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
    help='images directory for save')
parser.add_argument(
    '--image_base_path',
    type=str,
    default=None,
    help='images directory for fox text')
parser.add_argument(
    '--image_save_path',
    type=str,
    default=None,
    help='images directory for fox text for save')
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

kanji_chars = characters[92:]
print('Characters: ', kanji_chars)


def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    draw.text((canvas_size // 2, canvas_size // 2), ch,
              (0, 0, 0), font=font, anchor='mm')
    return img


def draw_example(
        ch,
        src_font,
        canvas_size,
        x_offset,
        y_offset,
        white_space_hashes):
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    dst_hash = hash(src_img.tobytes())
    if dst_hash in white_space_hashes:
        return None
    return src_img


def char_to_hash(ch, font, canvas_size, x_offset, y_offset):
    img = draw_single_char(ch, font, canvas_size, x_offset, y_offset)
    return hash(img.tobytes())


data_dir = args.ttf_path
data_root = pathlib.Path(data_dir)
print(f'Font Data Root: {data_root}')

all_font_paths = list(data_root.glob('*.*tf*'))[args.start_font:]
all_font_paths = sorted([str(path) for path in all_font_paths])
print(f'{len(all_font_paths)} fonts are found.')
for i in range(len(all_font_paths)):
    # print(all_image_paths[i])
    # if 'ipaexg.ttf' in all_font_paths[i]:
    #     print(i)
    
    if 'Roboto-Regular.ttf' in all_font_paths[i]:
        print(i)

all_hashes = []
_img = Image.new("RGB", (args.img_size, args.img_size), (255, 255, 255))
white_space_hashes = [hash(_img.tobytes())]


x_offset = (args.img_size - args.chara_size) / 2
y_offset = (args.img_size - args.chara_size) / 2

white_space_hash_sets = []
# 空白文字を取得
print(len(all_font_paths))
for font_path in tqdm(all_font_paths):
    tmp_white_space_hashes = set()
    tmp_all_hashes = set()
    font = ImageFont.truetype(font_path, size=args.chara_size)
    for ch in characters:
        tmp_hash = char_to_hash(ch, font, args.img_size, x_offset, y_offset)
        if tmp_hash in tmp_all_hashes:
            tmp_white_space_hashes.add(tmp_hash)
        else:
            tmp_all_hashes.add(tmp_hash)
    white_space_hash_sets.append(tmp_white_space_hashes)

print(len(white_space_hash_sets))
print('Save Images')
seq = list()
for (label, item) in enumerate(tqdm(all_font_paths)):
    label += args.start_font
    src_font = ImageFont.truetype(item, size=args.chara_size)
    tmp_white_space_hashes = white_space_hash_sets[label]
    for (cnt, chara) in enumerate(characters):
        img = draw_example(
            chara,
            src_font,
            args.img_size,
            x_offset,
            y_offset, tmp_white_space_hashes)
        if img is not None:
            path_full = os.path.join(args.save_path, 'id_%d' % label)
            if not os.path.exists(path_full):
                os.mkdir(path_full)
            img.save(os.path.join(path_full, "%04d.png" % (cnt)))

exit(0)

if args.image_base_path is not None:
    for (label, item) in enumerate(tqdm(all_font_paths)):
        label += args.start_font
        font_name = os.path.splitext(os.path.basename(item))[0]
        image_path = os.path.join(args.image_base_path, font_name+'.png')
        path_full = os.path.join(args.image_save_path, 'id_%d' % label + '.png')
        # copyfile(image_path, path_full)
        shutil.copyfile(image_path, path_full)
        