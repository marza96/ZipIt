from models.mlp import MLP
from graphs.mlp_graph import MLPGraph
from model_merger import ModelMerge
from matching_functions import match_tensors_zipit
from utils import reset_bn_stats

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import tqdm

from torch.utils.data import ConcatDataset
from repair import repair_bnorm

import os
import copy
import torch
import argparse
import torchvision
import eval_tools


def save_model(model, i):
    sd = model.state_dict()
    torch.save(sd, i)


def load_model(model, i):
    sd = torch.load(i, map_location=torch.device('cpu'))
    model.load_state_dict(sd)


def get_datasets():
    path   = os.path.dirname(os.path.abspath(__file__))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    mnistTrainSet = torchvision.datasets.FashionMNIST(
        root=path + '/data', 
        train=True,
        download=True, 
        transform=transform
    )

    first_half = [
        idx for idx, target in enumerate(mnistTrainSet.targets) 
        if target in [0, 1, 2, 3, 4]
    ]

    second_half = [
        idx for idx, target in enumerate(mnistTrainSet.targets) 
        if target in [5, 6, 7, 8, 9]
    ]  

    FirstHalfLoader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(mnistTrainSet, first_half),
        batch_size=512,
        shuffle=True,
        num_workers=8)
    
    SecondHalfLoader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(mnistTrainSet, second_half),
        batch_size=512,
        shuffle=True,
        num_workers=8)
    
    ConcatLoader = torch.utils.data.DataLoader(
        ConcatDataset((torch.utils.data.Subset(mnistTrainSet, first_half), torch.utils.data.Subset(mnistTrainSet, second_half))), 
        batch_size=512,
        shuffle=True, 
        num_workers=8
    )
    
    return FirstHalfLoader, SecondHalfLoader, ConcatLoader


def main(device="mps"):
    model1 = MLP(channels=512, layers=5, bnorm=True).eval()
    model2 = MLP(channels=512, layers=5, bnorm=True).eval()

    load_model(model1, "/Users/harissikic/Downloads/mlp_first_fmnist_bnorm_2_good.pt")
    load_model(model2, "/Users/harissikic/Downloads/mlp_second_fmnist_bnorm_2_good.pt")

    graph1 = MLPGraph(model1, 5, bnorm=True).graphify()  
    graph2 = MLPGraph(model2, 5, bnorm=True).graphify() 

    model3 = MLP(channels=512, layers=5, bnorm=True).eval()
    merge  = ModelMerge(graph1, graph2, device="cpu")
    
    loader0, loader1, loaderc = get_datasets()

    merge.transform(model3, loaderc, transform_fn=match_tensors_zipit)

    merge = repair_bnorm(model1, model2, merge, MLP, None, None, None, loaderc, device="cpu")

    acc = eval_tools.evaluate_acc_single_head(merge, loader=loaderc, device="cpu")
    print("FUSED ACC:", acc)

    save_model(merge, "merged.pt")


if __name__ == "__main__":
    main()
     
