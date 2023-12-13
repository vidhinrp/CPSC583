import torch
import torch_geometric
from sklearn.metrics import roc_auc_score
from utils import get_logger, compute_similarity


@torch.inference_mode()
def evaluate(gcn_c: torch.nn.Module,
             data: torch_geometric.data.Data,
             device: torch.device,
             mask_int: int = None,
             eval_on_cpu: bool = True,
             ) -> tuple[float, float]:
    """
    Evaluate the model on the validation or test set. Use  full-batch message passing.
    """
    get_logger().info('Evaluating')

    x = data[0][mask_int].x
    edge_index = data[0][mask_int].edge_index

    if eval_on_cpu:
        # move data to CPU
        x = x.cpu()
        edge_index = edge_index.cpu()
        gcn_c = gcn_c.cpu()
    else:
        # move data to GPU
        x = x.to(device)
        edge_index = edge_index.to(device)
        gcn_c = gcn_c.to(device)
        
    logits_total, _, preds = gcn_c(x, edge_index)
    out = compute_similarity(preds, data[0][mask_int].edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data[0][mask_int].edge_label.cpu().numpy(), out.cpu().numpy()) 