from models.mlp import MLP
from graphs.mlp_graph import MLPGraph
from model_merger import ModelMerge

import torchvision.transforms as transforms

import torch
import torchvision

if __name__ == "__main__":

    model1 = MLP(h=128, layers=5).eval()
    graph1 = MLPGraph(model1, 5).graphify()  

    model2 = MLP(h=128, layers=5).eval()
    graph2 = MLPGraph(model2, 5).graphify()  

    model3 = MLP(h=128, layers=5).eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=128,
        shuffle=True, 
        num_workers=8
    )

    from matching_functions import match_tensors_identity, match_tensors_zipit
    merge = ModelMerge(graph1, graph2)
    merge.transform(model3, dataloader, transform_fn=match_tensors_zipit)