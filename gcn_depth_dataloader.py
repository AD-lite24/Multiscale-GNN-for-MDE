from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import os
import random



def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


# preprocessing_transform = transforms.Compose([
#     transforms.Resize((320, 240), interpolation=Image.BILINEAR),
#     transforms.CenterCrop(304, 240),
#     transforms.ToTensor()
# ])

class GCNDepthDataLoader(Dataset):

    def __init__(self, mode, image_folder, depth_folder, transform = None) -> None:
        self.image_folder = image_folder
        self.depth_folder = depth_folder
        self.transform = transform

        self.image_files = sorted(os.listdir(image_folder))
        self.depth_files = sorted(os.listdir(depth_folder))

        # if mode == 'train':
            # self.training_samples = GCNDepthDataLoaderPreProcess(mode, transform=preprocesing_transform)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_index = self.extract_index(image_name)

        depth_name = self.obtain_depth_file(image_index)

        image_path = os.path.join(self.image_folder, image_name)
        depth_path = os.path.join(self.depth_folder, depth_name)

        rgb_image = self.load_image(image_path)
        depth_image = self.load_image(depth_path)

        return rgb_image, depth_image

    def extract_index(self, filename):
        index = filename.split("_")[-1]
        index = os.path.splitext(index)[0]  # Remove the file extension
        return index

    def obtain_depth_file(self, index):
        depth_file_name = f"sync_depth_{index}.png"
        return depth_file_name
    
    def load_image(self, path):

        image = Image.open(path)
        if self.transform is not None:
            image = self.transform(image)

        return image


class GCNDepthDataLoaderPreProcess():
    
    def __init__(self, args, mode, transform=None, is_for_online_eval=False) -> None:
        pass


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
