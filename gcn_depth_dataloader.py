from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import os
import random
import torch.nn as nn

class ExtractGraph(nn.Module):

    def __init__(self) -> None:
        super(ExtractGraph, self).__init__()

        self.maxpool = MaxPool(pool_size=2)
        self.noise = Noise(R_scale=0.4)  # From paper results
        self.dropout = GraphDropout(p=0.5)

    def forward(self, d_coarse, R_scale):

        d_pool = self.maxpool.forward(d_coarse)
        m = d_pool.shape[2]
        n = d_pool.shape[3]
        self.interval_threshold = IntervalThreshold(m, n)
        self.recon_graph = ReconGraph(m, n)

        print("pooled shape ", d_pool.shape)
        d_noise = self.noise.forward(d_pool)
        threshold = self.interval_threshold.forward(d_pool)
        adjacency_matrix = self.recon_graph.forward(d_noise, threshold)
        adjacency_matrix = self.dropout.forward(adjacency_matrix)

        return adjacency_matrix


class MaxPool(nn.Module):
    def __init__(self, pool_size):
        super(MaxPool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        return self.pool(x)


class Noise(nn.Module):
    def __init__(self, R_scale):
        super(Noise, self).__init__()
        self.mean = 0
        self.stdev = 1  # as defined in the paper

    def forward(self, d_coarse):
        noise = torch.rand_like(d_coarse)*self.stdev + self.mean
        d_noised = d_coarse + noise
        return d_noised


class IntervalThreshold(nn.Module):
    def __init__(self, m, n):
        super(IntervalThreshold, self).__init__()
        self.m = m
        self.n = n

    def forward(self, d_pool):
        threshold = (torch.max(d_pool) - torch.min(d_pool))/min(self.m, self.n)
        return threshold


class ReconGraph(nn.Module):
    def __init__(self, m, n):
        super(ReconGraph, self).__init__()
        self.m = m
        self.n = n

    def forward(self, d_noised, threshold):
        neighbours = set()
        labels = {}

        count = 0
        print(self.m, self.n)
        for i in range(self.m):
            for j in range(self.n):

                labels[(j, i)] = count  # Labeling each pixel in (x, y) form
                count += 1
                for dy in range(-1, 2):
                    for dx in range(-1, 2):

                        if dx != 0 and dy != 0 and i+dy >= 0 and i+dy < self.m and j+dx >= 0 and j+dx < self.n:
                            if abs(d_noised[0][0][i+dy][j+dx] - d_noised[0][0][i][j]) <= threshold:
                                # (x, y) format
                                neighbours.add(((j, i), (j+dx, i+dy)))
        adjacency_matrix = torch.zeros(
            (self.m*self.n, self.m*self.n), dtype=bool)
        print(adjacency_matrix.shape)

        for val in neighbours:
            N1, N2 = val  # in (x, y) form
            N1_x, N1_y = N1
            N2_x, N2_y = N2

            l1 = labels[(N1_x, N1_y)]
            l2 = labels[(N2_x, N2_y)]

            # Symmetric connections
            adjacency_matrix[l1, l2] = 1
            adjacency_matrix[l2, l1] = 1

        return adjacency_matrix


class GraphDropout(nn.Module):
    def __init__(self, p=0.5) -> None:
        super(GraphDropout, self).__init__()
        self.p = p

    def forward(self, adjacency_matrix):
        if self.train:
            mask = torch.empty_like(adjacency_matrix).bernoulli_(1 - self.p)
            output = adjacency_matrix * mask

        else:
            output = adjacency_matrix

        return output


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

    # def graph_extract(self, rgb, true_depth):
    #     extractor = ExtractGraph()
    #     encoder = Encoder()

    #     rgb = rgb.permute(1, 2, 0)*255
    #     rgb = rgb.numpy().astype(np.uint8)
    #     rgb = midas_transform(rgb).to(device)  # (1, C, H, W)

    #     depth_map = midas(rgb)  # (C, H, W) where C = 1
    #     down_rgb = encoder.forward(rgb)

    #     target_size = down_rgb.shape[2:]
    #     num_downsampled_channels = down_rgb.shape[1]
    #     # Maxpool will downsample by half further
    #     target_size = [x*2 for x in target_size]

    #     resize_transform = transforms.Resize(target_size)
    #     # Downsample midas output to (240, 320) using bilinear interpolation
    #     depth_maps = resize_transform(depth_maps)
    #     true_depth_map = resize_transform(true_depth)

    #     adjacency_matrix = extractor.forward(depth_maps, 0.4)
    #     true_adjacency_matrix = extractor.forward(true_depth_map, 0.4)
    #     print(adjacency_matrix.shape)
    #     print(true_adjacency_matrix.shape)

    #     # shape will be (64, 120*160)
    #     node_features = torch.reshape(down_rgb, (num_downsampled_channels, -1))

    #     return node_features, adjacency_matrix, true_adjacency_matrix



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
