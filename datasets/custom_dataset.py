import torch.utils.data as data

from PIL import Image

import os
import os.path
import sys
from io import BytesIO
import pickle
import torch
import numpy as np

def default_loader(path, input_ch=3):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path, input_ch=3)
    else:
        return pil_loader(path, input_ch)


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    """
    Parameters
    ----------
    dir : str
        path to the root of the dataset
    class_to_idx : dict[str, int]
        dir name and the value is index. e.g. {'id_0': 0, 'id_1': 1, ..., }
    extensions : list[str]
        list of allowed extensions

    Returns
    -------
    list[(str, int)]
        list of (image path, class index) tuples
    """
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

def find_classes(dir, is_dir=True):
    """
    Finds the class folders in a dataset.

    Args:
        dir (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    Ensures:
        No class is a subdirectory of another.
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        if is_dir:
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [os.path.splitext(d.name)[0] for d in os.scandir(dir)]
    else:
        classes = [
            d for d in os.listdir(dir) if os.path.isdir(
                os.path.join(
                    dir, d))]
    classes.sort()
    classes.sort(key=lambda x: int(x[3:])) # remove 'id_' and sort by number

    # the key is dir name and the value is index.
    # {'id_0': 0, 'id_1': 1, ..., }
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

class DatasetForCLIPEmbeddedImage(data.Dataset):
    def __init__(self, root, root_for_characters, clip_model, clip_preprocess, loader=default_loader, transform=None, input_ch=3, sampling_num=26):
        self.loader = loader
        self.transform = transform
        self.input_ch = input_ch
        self.sampling_num = sampling_num
        # class_to_idx: the key is dir name and the value is index.
        # {'id_0': 0, 'id_1': 1, ..., }
        self.classes, self.class_to_idx = find_classes(root, is_dir=False)
        self.embedded_images = {}

        with torch.no_grad():
            for class_name in self.classes:
                # class_name is 'id_0', 'id_1', ...
                image_file_path = os.path.join(root, class_name + '.png')
                image = Image.open(image_file_path)
                image = clip_preprocess(image).unsqueeze(0).to('cuda')
                embedded_image = clip_model.encode_image(image)
                embedded_image = embedded_image.cpu().detach().to(torch.float32)
                self.embedded_images[class_name] = embedded_image

        self.class_to_images = {}
        for class_name in self.classes:
            tmp_base_path = os.path.join(root_for_characters, class_name)
            file_paths = [os.path.join(tmp_base_path, file_name) for file_name in os.listdir(tmp_base_path)]
            # sample 10 files randomly
            # file_paths = np.random.choice(file_paths, self.sampling_num, replace=False)
            tmp_images = []
            for file_path in file_paths:
                image = self.loader(file_path, self.input_ch)
                if self.transform is not None:
                    image = self.transform(image)
                tmp_images.append(image)
            self.class_to_images[class_name] = tmp_images


        print('Finish embedding images')
        
    def __len__(self):
        return len(self.classes)
    
    def __getitem__(self, index):
        class_name = self.classes[index]
        embedded_image = self.embedded_images[class_name]
        sampling_idx = np.random.randint(self.sampling_num)
        try:
            return embedded_image, self.class_to_images[class_name][sampling_idx], self.class_to_idx[class_name]
        except:
            print('Error: ', class_name, sampling_idx)
            return embedded_image, self.class_to_images[class_name][0], self.class_to_idx[class_name]
        

class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root,
            loader,
            extensions,
            transform=None,
            target_transform=None):

        # class_to_idx: the key is dir name and the value is index.
        # {'id_0': 0, 'id_1': 1, ..., }
        classes, class_to_idx = find_classes(root)

        # The type of samples: [(img_path, class_idx), (img_path,
        # class_idx),...]
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(
                RuntimeError(
                    "Found 0 files in subfolders of: " +
                    root +
                    "\n"
                    "Supported extensions are: " +
                    ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [
                d for d in os.listdir(dir) if os.path.isdir(
                    os.path.join(
                        dir, d))]
        classes.sort()
        classes.sort(key=lambda x: int(x[3:]))

        # the key is dir name and the value is index.
        # {'id_0': 0, 'id_1': 1, ..., }
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        imgname = path.split('/')[-1].replace('.JPEG', '')
        return sample, target, imgname

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transform.__repr__().replace('\n',
                                                                       '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp,
                                   self.target_transform.__repr__().replace('\n',
                                                                            '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = [
    '.jpg',
    '.jpeg',
    '.png',
    '.ppm',
    '.bmp',
    '.pgm',
    '.tif',
    '.tiff',
    'webp']


def pil_loader(path, input_ch=3):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if input_ch == 1:
            return img.convert('L')
        return img.convert('RGB')


def accimage_loader(path, input_ch=3):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path, input_ch)




class PickledImageProvider(object):
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.examples = self.load_pickled_examples()

    def load_pickled_examples(self):
        with open(self.obj_path, "rb") as of:
            examples = list()
            while True:
                try:
                    e = pickle.load(of)
                    examples.append(e)
                    if len(examples) % 100000 == 0:
                        print("processed %d examples" % len(examples))
                except EOFError:
                    break
                except Exception:
                    pass
            print("unpickled total %d examples" % len(examples))
            return examples


class DatasetDumped(data.Dataset):
    def __init__(self, obj_path, transform=None, input_ch=3):
        self.image_provider = PickledImageProvider(obj_path)
        self.transform = transform
        self.input_ch = input_ch

    def __getitem__(self, index):
        target, sample = self.image_provider.examples[index]
        sample = BytesIO(sample)
        sample = Image.open(sample)
        if self.input_ch == 1:
            sample.convert('L')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.image_provider.examples)


class DatasetImages(data.Dataset):
    """
    Args:
        img_paths (list[string]): Image Paths
        loader (callable): A function to load a sample given its path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        samples (list): List of (sample path) tuples
    """

    def __init__(
            self,
            img_paths,
            loader=default_loader,
            transform=None,
            input_ch=3,
    ):

        # The type of samples: [(img_path, class_idx), (img_path,
        # class_idx),...]
        samples = img_paths

        self.loader = loader

        self.samples = samples

        self.transform = transform
        self.input_ch = input_ch

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        sample = self.loader(path, self.input_ch)
        if self.transform is not None:
            sample = self.transform(sample)
        img_name = path.split('/')[-1].replace('.JPEG', '')
        return sample, img_name

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Image paths: {}\n'.format(self.samples)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transform.__repr__().replace('\n',
                                                                       '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp,
                                   self.target_transform.__repr__().replace('\n',
                                                                            '\n' + ' ' * len(tmp)))
        return fmt_str


class ImageFolderRemap(DatasetFolder):
    def __init__(
            self,
            root,
            transform=None,
            target_transform=None,
            loader=default_loader,
            remap_table=None,
            with_idx=False,
            input_ch=3):
        super(
            ImageFolderRemap,
            self).__init__(
            root,
            loader,
            IMG_EXTENSIONS,
            transform=transform,
            target_transform=target_transform)

        self.imgs = self.samples
        self.class_table = remap_table
        self.with_idx = with_idx
        self.input_ch = input_ch

    def __getitem__(self, index):
        # The type of self.samples: [(img_path, class_idx), (img_path,
        # class_idx),...]
        path, target = self.samples[index]
        cnt_idx = int(os.path.splitext(os.path.basename(path))[0])
        sample = self.loader(path, self.input_ch)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = self.class_table[target]
        if self.with_idx:
            return sample, index, target

        # The type of sample is PIL.Image
        # The type of target is int
        # target is the target font id
        return sample, target, cnt_idx


class ImageFolderRemapStyleAttraction(DatasetFolder):
    def __init__(
            self,
            root,
            transform=None,
            target_transform=None,
            loader=default_loader,
            remap_table=None,
            input_ch=3,
            batch_size=32
    ):
        super(
            ImageFolderRemapStyleAttraction,
            self).__init__(
            root,
            loader,
            IMG_EXTENSIONS,
            transform=transform,
            target_transform=target_transform)

        # __getitem__では同じフォントの画像をbatch_size分取得する (style_attractionを計算するには同じフォントの画像が必要)

        # self.samplesはDatasetFolderのinit関数で定義される
        # self.samples = make_dataset(root, class_to_idx, extensions)
        # self.samplesは[(img_path, class_idx), (img_path, class_idx),...]の形式
        self.imgs = self.samples

        self.class_table = remap_table

        # 画像のinput_chを指定する
        self.input_ch = input_ch

        self.batch_size = batch_size

    def __getitem__(self, index):
        # The type of self.samples: [(img_path, class_idx), (img_path,
        # class_idx),...]

        samples = []
        targets = []
        cnt_idxs = []

        font_id = None
        for i in range(self.batch_size):
            # The type of sample is PIL.Image
            # The type of target is int
            # target is the target font id

            path, target = self.samples[index * self.batch_size + i]

            if font_id is None:
                font_id = target
            elif font_id != target:
                # batch_size分のデータを取得する前に、フォントが変わったら、残りの分を敷き詰めてループを抜ける
                for j in range(i, self.batch_size):
                    samples.append(sample)
                    targets.append(target)
                    cnt_idxs.append(cnt_idx)
                break
            cnt_idx = int(os.path.splitext(os.path.basename(path))[0])
            sample = self.loader(path, self.input_ch)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            target = self.class_table[target]

            samples.append(sample)
            targets.append(target)
            cnt_idxs.append(cnt_idx)

        return samples, targets, cnt_idxs

    def __len__(self):
        return int(len(self.samples) / self.batch_size)


class CrossdomainFolder(data.Dataset):
    def __init__(
            self,
            root,
            data_to_use=[
                'photo',
                'monet'],
            transform=None,
            loader=default_loader,
            extensions='jpg'):
        self.data_to_use = data_to_use
        classes, class_to_idx = self._find_classes(root)

        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(
                RuntimeError(
                    "Found 0 files in subfolders of: " +
                    root +
                    "\n"
                    "Supported extensions are: " +
                    ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(
                dir) if d.is_dir() and d.name in self.data_to_use]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(
                os.path.join(dir, d)) and d in self.data_to_use]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transform.__repr__().replace('\n',
                                                                       '\n' + ' ' * len(tmp)))
        return fmt_str
