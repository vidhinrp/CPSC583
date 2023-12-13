import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import wandb
from tap import Tap
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from evalcopy import evaluate
from data import get_data
from gcn import GCN
from utils import (TensorMap, get_logger, get_neighborhoods,
                           sample_neighborhoods_from_probs, slice_adjacency, compute_similarity)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Arguments(Tap):
    dataset: str = 'citeseer'

    sampling_hops: int = 2
    num_samples: int = 1000
    use_indicators: bool = True
    lr_gf: float = 0.024000282947302637
    lr_gc: float = 0.08269212064443145
    loss_coef: float = 1000
    log_z_init: float = 0.
    reg_param: float = 0.
    dropout: float = 0.1

    model_type: str = 'gcn'
    hidden_dim: int = 512
    max_epochs: int = 5
    batch_size: int = 256
    eval_frequency: int = 5
    eval_on_cpu: bool = True

    runs: int = 10
    notes: str = None
    log_wandb: bool = False
    config_file: str = None


def train(args: Arguments):

    logger = get_logger()

    path = os.path.join(os.getcwd(), 'data', args.dataset)
    data, num_features, num_classes = get_data(root=path, name=args.dataset)
    node_map = TensorMap(size=data.x.size(0))

    if args.use_indicators:
        num_indicators = args.sampling_hops + 1
    else:
        num_indicators = 0

    if args.model_type == 'gcn':
        gcn_c = GCN(data.num_features, hidden_dims=[args.hidden_dim, num_classes], dropout=args.dropout).to(device)
        # GCN model for GFlotNet sampling
        gcn_gf = GCN(data.num_features + num_indicators,
                      hidden_dims=[args.hidden_dim, 1]).to(device)

    log_z = torch.tensor(args.log_z_init, requires_grad=True)
    optimizer_c = Adam(gcn_c.parameters(), lr=args.lr_gc)
    optimizer_gf = Adam(list(gcn_gf.parameters()) + [log_z], lr=args.lr_gf)

    loss_fn = nn.BCEWithLogitsLoss()
    
    train_idx = data[0][0].train_mask.nonzero().squeeze(1)
    train_loader = DataLoader(TensorDataset(train_idx), batch_size=args.batch_size)

    val_idx = data[0][1].val_mask.nonzero().squeeze(1)
    val_loader = DataLoader(TensorDataset(val_idx), batch_size=args.batch_size)

    test_idx = data[0][2].test_mask.nonzero().squeeze(1)
    test_loader = DataLoader(TensorDataset(test_idx), batch_size=args.batch_size)

    adjacency = sp.csr_matrix((np.ones(data.edge_index.shape[1], dtype=bool),
                               data.edge_index),
                              shape=(data.x.size(0), data.x.size(0)))

    prev_nodes_mask = torch.zeros(data.x.size(0), dtype=torch.bool)
    batch_nodes_mask = torch.zeros(data.x.size(0), dtype=torch.bool)
    indicator_features = torch.zeros((data.x.size(0), num_indicators))

    # This will collect memory allocations for all epochs
    all_mem_allocations_point1 = []
    all_mem_allocations_point2 = []
    all_mem_allocations_point3 = []

    logger.info('Training')
    
    for epoch in range(1, args.max_epochs + 1):
        
        acc_loss_gfn = 0
        acc_loss_c = 0
        # add a list to collect memory usage
        mem_allocations_point1 = []  # The first point of memory usage measurement after the GCNConv forward pass
        mem_allocations_point2 = []  # The second point of memory usage measurement after the GCNConv backward pass
        mem_allocations_point3 = []  # The third point of memory usage measurement after the GCNConv backward pass
                
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}') as bar:
            for batch_id, batch in enumerate(train_loader):
                #torch.cuda.empty_cache()
                #torch.cuda.reset_peak_memory_stats()
                
                target_nodes = batch[0]

                previous_nodes = target_nodes.clone()
                all_nodes_mask = torch.zeros_like(prev_nodes_mask)
                all_nodes_mask[target_nodes] = True

                indicator_features.zero_()
                indicator_features[target_nodes, -1] = 1.0

                global_edge_indices = []
                log_probs = []
                sampled_sizes = []
                neighborhood_sizes = []
                all_statistics = []
                # Sample neighborhoods with the GCN-GF model
                for hop in range(args.sampling_hops):
                    # Get neighborhoods of target nodes in batch
                    neighborhoods = get_neighborhoods(previous_nodes, adjacency)

                    # Identify batch nodes (nodes + neighbors) and neighbors
                    prev_nodes_mask.zero_()
                    batch_nodes_mask.zero_()
                    prev_nodes_mask[previous_nodes] = True
                    batch_nodes_mask[neighborhoods.view(-1)] = True
                    neighbor_nodes_mask = batch_nodes_mask & ~prev_nodes_mask

                    batch_nodes = node_map.values[batch_nodes_mask]
                    neighbor_nodes = node_map.values[neighbor_nodes_mask]
                    indicator_features[neighbor_nodes, hop] = 1.0

                    # Map neighborhoods to local node IDs
                    node_map.update(batch_nodes)
                    local_neighborhoods = node_map.map(neighborhoods).to(device)
                    # Select only the needed rows from the feature and
                    # indicator matrices
                    if args.use_indicators:
                        x = torch.cat([data.x[batch_nodes],
                                        indicator_features[batch_nodes]],
                                        dim=1
                                        ).to(device)
                    else:
                        x = data.x[batch_nodes].to(device)

                    # Get probabilities for sampling each node
                    node_logits, _ , m = gcn_gf(x, local_neighborhoods)
                    # Select logits for neighbor nodes only
                    node_logits = node_logits[node_map.map(neighbor_nodes)]

                    # Sample neighbors using the logits
                    sampled_neighboring_nodes, log_prob, statistics = sample_neighborhoods_from_probs(
                        node_logits,
                        neighbor_nodes,
                        args.num_samples
                    )
                    all_nodes_mask[sampled_neighboring_nodes] = True

                    log_probs.append(log_prob)
                    sampled_sizes.append(sampled_neighboring_nodes.shape[0])
                    neighborhood_sizes.append(neighborhoods.shape[-1])
                    all_statistics.append(statistics)

                    # Update batch nodes for next hop
                    batch_nodes = torch.cat([target_nodes,
                                                sampled_neighboring_nodes],
                                            dim=0)

                    # Retrieve the edge index that results after sampling
                    k_hop_edges = slice_adjacency(adjacency,
                                                rows=previous_nodes,
                                                cols=batch_nodes)
                    global_edge_indices.append(k_hop_edges)

                    # Update the previous_nodes
                    previous_nodes = batch_nodes.clone()
                                
                # Converting global indices to the local of final batch_nodes.
                # The final batch_nodes are the nodes sampled from the second
                # hop concatenated with the target nodes
                all_nodes = node_map.values[all_nodes_mask]
                node_map.update(all_nodes)
                edge_indices = [node_map.map(e).to(device) for e in global_edge_indices]
                x = data.x[all_nodes].to(device)
                logits, gcn_mem_alloc, preds = gcn_c(x, edge_indices)
                negative_edges = negative_sampling(
                                edge_index=torch.cat(edge_indices, dim=1),  # positive edges in the graph
                                num_nodes=len(all_nodes),  # number of nodes
                                num_neg_samples=10,  # number of negative examples
                                )
                
                negative_edges_list = [negative_edges] if isinstance(negative_edges, torch.Tensor) else negative_edges

                # Concatenate the edge indices tensors and negative samples along dimension 1
                all_edges = edge_indices + negative_edges_list

                # Concatenate the list of tensors along dimension 1
                all_edges_concatenated = torch.cat(all_edges, dim=1)
                labels = torch.ones(all_edges_concatenated.size(1))
                labels[-10:] = torch.zeros(10)

                sim = compute_similarity(preds, all_edges_concatenated)
                loss_c = loss_fn(sim, labels)+ args.reg_param * torch.sum(torch.var(preds, dim=0))
                
                optimizer_c.zero_grad()

                mem_allocations_point3.append(torch.cuda.memory_allocated() / (1024 * 1024))

                loss_c.backward()

                optimizer_c.step()

                optimizer_gf.zero_grad()
                cost_gfn = loss_c.detach()

                loss_gfn = (log_z + torch.sum(torch.cat(log_probs, dim=0)) + args.loss_coef*cost_gfn)**2

                mem_allocations_point1.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
                mem_allocations_point2.append(gcn_mem_alloc)

                loss_gfn.backward()

                optimizer_gf.step()

                batch_loss_gfn = loss_gfn.item()
                batch_loss_c = loss_c.item()

               
                log_dict = {}
                for i, statistics in enumerate(all_statistics):
                    for key, value in statistics.items():
                        log_dict[f"{key}_{i}"] = value


                acc_loss_gfn += batch_loss_gfn / len(train_loader)
                acc_loss_c += batch_loss_c / len(train_loader)

                bar.set_postfix({'batch_loss_gfn': batch_loss_gfn,
                                    'batch_loss_c': batch_loss_c,
                                    'log_z': log_z.item(),
                                    'log_probs': torch.sum(torch.cat(log_probs, dim=0)).item()})
                bar.update()

        bar.close()
        all_mem_allocations_point1.extend(mem_allocations_point1)
        all_mem_allocations_point2.extend(mem_allocations_point2)
        all_mem_allocations_point3.extend(mem_allocations_point3)
        
        if (epoch + 1) % args.eval_frequency == 0:
            roc = evaluate(gcn_c,
                           data,

                                        device,
                                        1,
                                        args.eval_on_cpu,
                                        )
            if args.eval_on_cpu:
                gcn_c.to(device)

            log_dict = {'epoch': epoch,
                        'loss_gfn': acc_loss_gfn,
                        'loss_c': acc_loss_c,
                        'valid_accuracy': roc}

            logger.info(f'loss_gfn={acc_loss_gfn:.6f}, '
                        f'loss_c={acc_loss_c:.6f}, '
                        f'valid_accuracy={roc:.3f} ')



    test_roc = evaluate(gcn_c,
                                      data,
                                      device,
                                      2,
                                      args.eval_on_cpu,
                                      )
    logger.info(f'test_roc={test_roc:.3f}')
    return test_roc, all_mem_allocations_point1, all_mem_allocations_point2, all_mem_allocations_point3


args = Arguments(explicit_bool=True).parse_args()

# If a config file is specified, load it, and parse again the CLI
# which takes precedence
if args.config_file is not None:
    args = Arguments(explicit_bool=True, config_files=[args.config_file])
    args = args.parse_args()

results = torch.empty(args.runs)
mem1 = []
mem2 = []
mem3 = []
for r in range(args.runs):
    test_f1, mean_mem1, mean_mem2, mean_mem3 = train(args)
    results[r] = test_f1
    mem1.extend(mean_mem1)
    mem2.extend(mean_mem2)
    mem3.extend(mean_mem3)


print(f'Memory point 1: {np.mean(mem1)} MB ± {np.std(mem1):.2f}')
print(f'Memory point 2: {np.mean(mem2)} MB ± {np.std(mem2):.2f}')
print(f'Memory point 2: {np.mean(mem3)} MB ± {np.std(mem3):.2f}')
print(f'Acc: {100 * results.mean():.2f} ± {100 * results.std():.2f}')