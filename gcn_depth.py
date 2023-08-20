import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import cv2
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter


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
        self.stdev = 1 #as defined in the paper
        
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
        
        for i in range(self.m):
            for j in range(self.n):

                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if dx != 0 and dy != 0 and i+dx >= 0 and i+dy < self.m and j+dx >= 0 and j+dy < self.n:
                            if abs(d_noised[i+dy][j+dx] - d_noised[i][j]) <= threshold:
                                neighbours.add(((j, i), (j+dx, j+dy))) #(x, y) format

        adjacency_matrix = torch.zeros((self.m, self.n), dtype=bool)

        for val in neighbours:

            Nj, Ni = val
            adjacency_matrix[Nj][Ni] = 1

        return adjacency_matrix        

class GraphDropout(nn.Module):
    def __init__(self, p = 0.5) -> None:
        super(GraphDropout, self).__init__()
        self.p = p
    
    def forward(self, adjacency_matrix):
        if self.train:
            mask = torch.empty_like(adjacency_matrix).bernoulli_(1 - self.p)
            output = adjacency_matrix * mask

        else:
            output = adjacency_matrix

        return output

class ExtractGraph(nn.Module):

    def __init__(self, d_coarse, R_scale, m, n) -> None:
        super(ExtractGraph, self).__init__()
        self.d_coarse = d_coarse
        self.R_scale = R_scale

        self.maxpool = MaxPool(pool_size=2)
        self.noise = Noise(R_scale=0.4) # From paper results
        self.interval_threshold = IntervalThreshold(m, n)
        self.recon_graph = ReconGraph(m, n)
        self.dropout = GraphDropout(p=0.5)

    def forward(self):
        d_pool = self.maxpool.forward(self.d_coarse)
        d_noise = self.noise.forward(d_pool)
        threshold = self.interval_threshold.forward(d_pool)
        adjacency_matrix = self.recon_graph.forward(d_noise, threshold)
        adjacency_matrix = self.dropout.forward(adjacency_matrix)

        return adjacency_matrix

class Encoder(nn.Module):

    def __init__(self) -> None:
        super(Encoder, self).__init__()
        encoder = models.resnet.resnet50(
            weights=models.ResNet50_Weights.DEFAULT)
        encoder = nn.Sequential(*list(encoder.children()))[:4]
        self.resnet_encoder = encoder

    def forward(self, x):
        self.resnet_encoder.eval()
        return self.resnet_encoder(x)

class GNNModel(nn.Module):

    def __init__(self, adjacnecy_matrix) -> None:
        super(GNNModel, self).__init__()
        self.adjacency_matrix = adjacnecy_matrix

    def propogate(self, l, features):
        adjacency_matrix_looped = self.adjacency_matrix + torch.eye(self.adjacency_matrix.shape[0])
        h_l = features
        h_l = F.relu()


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    encoder = Encoder()
    input = torch.randn(1, 3, 1280, 1920)
    node_featueres = encoder.forward(input) # Downsamples 224x224 to 56x56 after usage of 4 training modules
    print('shape after encoder layer: ', node_featueres.shape)

    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    img = cv2.imread("../test_images/inputs/skyscraper_city.jpeg")
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    print('input batch shape ', input_batch.shape)


    #Interpolate to the downsampled shape
    with torch.no_grad():
        print(input_batch.shape)
        prediction = midas(input_batch)

        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=node_featueres.shape[2:4],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu()
    print("shape through midas ", depth_map.shape)
    plt.imshow(depth_map.numpy())


    # At this point, the node features are in node_features while the depth map is in depth_map

    # graph_extract = ExtractGraph(depth_map, 0.4, 320, 480)
    # adjacency_matrix = graph_extract.forward()

    


