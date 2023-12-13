# import the pytorch library into environment and check its version
from typing import Tuple

import torch_geometric.transforms as T
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import Planetoid


def get_planetoid(root: str, name: str) -> Tuple[Data, int, int]:
    transform = T.Compose([T.RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=True, 
                                             add_negative_train_samples=False)])
    dataset = Planetoid(f'{root}/Planetoid', name, transform=transform)
    return dataset, dataset.num_features, dataset.num_classes

def get_data(root: str, name: str) -> Tuple[Data, int, int]:
    if name.lower() in ['cora', 'citeseer', 'pubmed']:
        return get_planetoid(root, name)
    else:
        raise NotImplementedError
