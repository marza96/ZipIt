from models.mlp import MLP
from graphs.mlp_graph import MLPGraph
from model_merger import ModelMerge
from matching_functions import match_tensors_zipit

import torchvision.transforms as transforms

from torch.utils.data import ConcatDataset

import os
import torch
import argparse
import torchvision
import eval_tools


def save_model(model, i):
    sd = model.state_dict()
    torch.save(sd, i)


def load_model(model, i):
    sd = torch.load(i)
    model.load_state_dict(sd)


def main(dataset0, dataset1, device="cuda"):
    h = 128
    layers = 5
    device = torch.device(device)
    path   = os.path.dirname(__file__)

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

    loader0 = None
    loader1 = None
    prefix0 = None
    prefix1 = None

    if dataset0 == "MNIST":
        prefix0 = "mnist_"
    elif dataset0 == "FashionMNIST":
        prefix0 = "fash_mnist_"

    if dataset1 == "MNIST":
        prefix1 = "mnist_"
    elif dataset1 == "FashionMNIST":
        prefix1 = "fash_mnist_"

    model1 = MLP(h=128, layers=5).eval()
    model2 = MLP(h=128, layers=5).eval()

    load_model(model1, path + '/pt_models/%smlp_e50_l%d_h%d_v1_%s.pt' % (prefix0, layers, h, device))
    load_model(model2, path + '/pt_models/%smlp_e50_l%d_h%d_v2_%s.pt' % (prefix1, layers, h, device))

    graph1 = MLPGraph(model1, 5).graphify()  
    graph2 = MLPGraph(model2, 5).graphify()  

    model3 = MLP(h=128, layers=5).eval()
    merge  = ModelMerge(graph1, graph2)

    merge.transform(model3, ConcatTrainLoader, transform_fn=match_tensors_zipit)
    
    # print(merge.merges.keys())
    # print(merge.merges[5][0][:20, :20])
    # print(merge.merges[5][0][:20, :].sum(dim=1))

    print("DBG", torch.unique(merge.merges[5][0][:20, :], dim=1))
    print("FUSED ACC:", eval_tools.evaluate_acc(merge.head_models[0], loader=FashionMNISTTrainLoader, device=device))

    save_model(merge.head_models[0], "merged.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default="cuda")
    parser.add_argument('-d0', '--dataset0', default="cuda")
    parser.add_argument('-d1', '--dataset1', default="cuda")
    args = parser.parse_args()

    main(args.dataset0, args.dataset1, device=args.device)
     
