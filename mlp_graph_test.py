from models.mlp import MLP
from graphs.mlp_graph import MLPGraph
from model_merger import ModelMerge
from matching_functions import match_tensors_zipit

from REPAIR_MTZ.models.mlp import MLP as REPAIR_MLP
from REPAIR_MTZ.REPAIR import eval
from REPAIR_MTZ.REPAIR.neural_align_diff import NeuralAlignDiff

import torchvision.transforms as transforms

from torch.utils.data import ConcatDataset

import torch
import torchvision

def save_model(model, i):
    sd = model.state_dict()
    torch.save(sd, i)

def load_model(model, i):
    sd = torch.load(i)
    model.load_state_dict(sd)


if __name__ == "__main__":
    model1 = MLP(h=128, layers=5).eval()
    model2 = MLP(h=128, layers=5).eval()

    load_model(model1, "REPAIR_MTZ/mlps2/fash_mnist_mlp_e50_l5_h128_v2_cuda.pt")
    load_model(model2, "REPAIR_MTZ/mlps2/mnist_mlp_e50_l5_h128_v1.pt")

    graph1 = MLPGraph(model1, 5).graphify()  
    graph2 = MLPGraph(model2, 5).graphify()  

    model3 = MLP(h=128, layers=5).eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    FashionMNISTTrainSet = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    FashionMNISTTrainLoader = torch.utils.data.DataLoader(
        FashionMNISTTrainSet, 
        batch_size=128,
        shuffle=True, 
        num_workers=8
    )
    MNISTTrainSet = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    MNISTTrainLoader = torch.utils.data.DataLoader(
        MNISTTrainSet, 
        batch_size=128,
        shuffle=True, 
        num_workers=8
    )
    ConcatTrainLoader = torch.utils.data.DataLoader(
        ConcatDataset((FashionMNISTTrainSet, MNISTTrainSet)), 
        batch_size=128,
        shuffle=True, 
        num_workers=8
    )

    merge = ModelMerge(graph1, graph2)
    merge.transform(model3, ConcatTrainLoader, transform_fn=match_tensors_zipit)
     
