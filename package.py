import os
import pickle
import argparse


def package(
        font_img_dir='../font-images',
        save_dir='../font-dumped',
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    font_dir_list = sorted(
        [f.path for f in os.scandir(font_img_dir) if os.path.isdir(f)])
    save_path = os.path.join(save_dir, 'characters.obj')

    with open(save_path, 'wb') as fs:
        for font_dir_path in font_dir_list:

            # An example of font_dir_basename is 'id_123'
            font_dir_basename = os.path.basename(font_dir_path)
            font_id = int(font_dir_basename.replace('id_', ''))

            # character_imgs = [f for f in os.scandir(font_dir_path) if os.path.isfile(f)]
            character_imgs = [f for f in os.scandir(font_dir_path)]
            for img in character_imgs:
                with open(img, 'rb') as f:
                    img_bytes = f.read()
                    sample = (font_id, img_bytes)
                    pickle.dump(sample, fs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Package')
    parser.add_argument('--dir', type=str, default='../font-images')
    parser.add_argument('--save_dir', type=str, default='../font-dumped')
    args = parser.parse_args()
    package(font_img_dir=args.dir, save_dir=args.save_dir)
