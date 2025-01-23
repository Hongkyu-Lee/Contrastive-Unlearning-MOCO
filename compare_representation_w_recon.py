import os
import torch
import numpy as np
import wandb
import sys
sys.path.append("/home/hlee959/projects/2023_CSUL/CSUL/")
from core.args import Untrain_Parser
from core.data import Data_Loader
from core.model import Model_Loader
from core.model import Checkpoint_Loader


import os
import copy
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import wandb
import sys
sys.path.append("/home/hlee959/projects/2023_CSUL/CSUL/")
from torch.nn import functional as F
from core.args import Untrain_Parser
from core.data import Data_Loader
from core.model import Model_Loader
from core.model import Checkpoint_Loader

def contrastive_loss(u_feat,
                      rt_feat,
                      u_label,
                      rt_label,
                      temperature=0.7,
                      base_temperature=0.7):
    
    u_label = u_label.contiguous().view(-1, 1)
    rt_label = rt_label.contiguous().view(-1, 1)

    mask = torch.eq(u_label, rt_label.T)
    p_mask = (~mask).clone().float()
    p_count = torch.sum(mask).item()

    n_mask = (mask).clone().float()
    n_add_mask = (~(n_mask.sum(1).bool())).int().contiguous().view(-1, 1)
    
    orig_logits = torch.matmul(u_feat, rt_feat.T)
    
    logits = torch.div(orig_logits, temperature)
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits -= logits_max.detach()
    
    # print(n_mask, n_add_mask)

    exp_logits = torch.exp(logits) + 1e-20
    # print("exp_logits", exp_logits)
    p_logits = logits * p_mask
    n_logits = (exp_logits * n_mask).sum(1, keepdim=True)

    # print("n_mask_sum : ", n_mask.sum())
    # print("p_mask_sum : ", p_mask.sum())
    
    if (n_mask.sum(1) == 0).any():
        n_logits += n_add_mask
    
    log_prob = p_logits - torch.log(n_logits)
    
    # print(log_prob.shape)
    mean_log_prob = log_prob.sum(1) / p_mask.sum(1)
    # print("sum is zero: ", (p_mask.sum(1) == 0).any())

    loss = -(temperature / base_temperature) * mean_log_prob
    loss = loss.mean()

    return loss

def contrastive_unlearn(csul,
                        u_img,
                        u_label,
                        rt_img,
                        rt_label,
                        device):

    u_img = u_img.to(device)
    u_label = u_label.to(device)
    rt_img = rt_img.to(device)
    rt_label = rt_label.to(device)

    batch_size= u_img.shape[0]

    imgs = torch.cat([u_img, rt_img])
    pred, feat = csul.model(imgs)

    u_pred, rt_pred = torch.split(pred, [batch_size, batch_size], dim=0)
    u_feat, rt_feat = torch.split(feat, [batch_size, batch_size], dim=0)

    u_loss = contrastive_loss(u_feat, rt_feat, u_label, rt_label)
    rt_loss = nn.CrossEntropyLoss()(rt_pred, rt_label)

    total_loss = (0.5 * u_loss) + (0.5 * rt_loss)

    csul.optim.zero_grad()
    total_loss.backward()
    csul.optim.step()

    return u_loss.item(), rt_loss.item()

def momentum_update_key_encoder(encoder_q,
                                encoder_k,
                                m=0.999):
    """
    Momentum update of the key encoder
    """
    for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)
    
    return encoder_k

def dequeue_and_enqueue(queue, batch):
    batch_size = batch.shape[0]
    ptr = int(queue.queue_ptr)
    abs_ptr = int(queue.abs_queue_ptr)

    if queue.queue.ndim == 1:
        # Handle 1D queue (labels)
        if ptr + batch_size > queue.K:
            first_part_size = queue.K - ptr
            queue.queue[ptr:] = batch[:first_part_size].squeeze()
            queue.queue[:batch_size - first_part_size] = batch[first_part_size:].squeeze()
        else:
            queue.queue[ptr:ptr + batch_size] = batch.squeeze()
    else:
        # Handle 2D queue (features)
        if ptr + batch_size > queue.K:
            first_part_size = queue.K - ptr
            queue.queue[:, ptr:] = batch[:first_part_size].T
            queue.queue[:, :batch_size - first_part_size] = batch[first_part_size:].T
        else:
            queue.queue[:, ptr:ptr + batch_size] = batch.T
    
    queue.queue_ptr[0] = (ptr + batch_size) % queue.K
    if abs_ptr < queue.K:
        queue.abs_queue_ptr[0] = (abs_ptr + batch_size)

    return queue

def MoCo_loss(logits, pos_mask):
    log_softmax = logits - torch.log(torch.sum(torch.exp(logits), dim=1, keepdim=True))
    nll = -log_softmax[pos_mask]
    return torch.mean(nll)

def MoCo_unlearn(moco,
                 u_img,
                 u_label,
                 rt_img,
                 rt_label,
                 device):

    u_img = u_img.to(device)
    u_label = u_label.to(device)
    rt_img = rt_img.to(device)
    rt_label = rt_label.to(device)

    batch_size = u_img.shape[0]

    imgs = torch.cat([u_img, rt_img])
    logits_q, feats_q = moco.encoder_q(imgs)

    ul_logits, ul_feats = logits_q[:batch_size], feats_q[:batch_size]
    rt_logits, rt_feats = logits_q[batch_size:], feats_q[batch_size:]

    with torch.no_grad():
        moco.encoder_k = momentum_update_key_encoder(moco.encoder_q, moco.encoder_k)
        _, rt_feats_k = moco.encoder_k(rt_img.clone().detach().to(device))

    moco.queue = dequeue_and_enqueue(moco.queue, rt_feats_k)
    moco.label_queue = dequeue_and_enqueue(moco.label_queue, rt_label.clone().contiguous().view(-1, 1))

    if int(moco.queue.abs_queue_ptr) < moco.queue.K:
        _current_queue = int(moco.queue.abs_queue_ptr)
        logits = torch.einsum("nc,ck->nk", [ul_feats, moco.queue.queue.clone().detach().to(device)[:, :_current_queue]])
        queue_labels = moco.label_queue.queue[:_current_queue].clone().detach().to(device)
        pos_mask = torch.eq(u_label.unsqueeze(1), queue_labels.view(1, -1))
    else:
        logits = torch.einsum("nc,ck->nk", [ul_feats, moco.queue.queue.clone().detach().to(device)])
        queue_labels = moco.label_queue.queue.clone().detach().to(device)
        pos_mask = ~torch.eq(u_label.unsqueeze(1), queue_labels.view(1, -1))

    ul_loss = MoCo_loss(logits, pos_mask)
    rt_loss = nn.CrossEntropyLoss()(rt_logits, rt_label)
    total_loss = (0.5 * ul_loss) + (0.5 * rt_loss)

    moco.optim.zero_grad()
    total_loss.backward()
    moco.optim.step()

    return ul_loss.item(), rt_loss.item()

class Queue:
    def __init__(self, dim, K, device):
        self.K = K
        if dim < 0:
            self.queue = torch.ones(K, device=device)*-1
        else:
            self.queue = torch.randn(dim, K, device=device)
        self.queue_ptr = torch.zeros(1, dtype=torch.long, device=device)
        self.abs_queue_ptr = torch.zeros(1, dtype=torch.long, device=device)

class CSUL_Wrapper:
    def __init__(self, args, model, device):
        self.model = copy.deepcopy(model).to(device)
        self.optim = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
        
class MoCo_Wrapper:
    def __init__(self, args, model, device):
        self.encoder_k = copy.deepcopy(model).to(device)
        self.encoder_q = copy.deepcopy(model).to(device)
        self.optim = torch.optim.SGD(self.encoder_q.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
        self.queue = Queue(args.feat_dim, args.k, device)
        self.label_queue = Queue(-1, args.k, device)

def evaluate(model, loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_conf = 0
    for idx, (data, label) in enumerate(loader):
        data = data.to(device)
        label = label.to(device)

        _pred, feat = model(data)
        total_samples += len(data)
        _, pred = torch.max(_pred, 1)
        total_correct += (torch.eq(pred, label)).sum().item()
        conf, _ = torch.max(F.softmax(_pred, 1), 1)
        total_conf += conf.sum().item() 

    return total_correct / total_samples, total_conf/total_samples

def measure_differences(orig_feat, target_feat, suffix: str):
    out_dict = dict()
    
    # Ensure inputs are flattened for certain metrics
    orig_flat = orig_feat.view(-1).unsqueeze(0)  # Shape: [1, N]
    target_flat = target_feat.view(-1).unsqueeze(0)  # Shape: [1, N]
    
    # Norm-based differences (these work on original 2D matrices)
    out_dict[suffix+"_l1"] = torch.norm(orig_feat - target_feat, p=1)
    out_dict[suffix+"_l2"] = torch.norm(orig_feat - target_feat, p=2)
    out_dict[suffix+"_frob"] = torch.norm(orig_feat - target_feat, p="fro")
    out_dict[suffix+"_max_diff"] = torch.max(torch.abs(orig_feat - target_feat))
    out_dict[suffix+"_mean_diff"] = torch.mean(torch.abs(orig_feat - target_feat))
    out_dict[suffix+"_std_diff"] = torch.std(torch.abs(orig_feat - target_feat))
    
    # Cosine similarity (needs flattened vectors)
    out_dict[suffix+"_cosine_sim"] = 1 - F.cosine_similarity(orig_flat, target_flat)
    
    # Distribution-based differences (need softmax and proper shapes)
    orig_prob = F.softmax(orig_flat, dim=1)
    target_prob = F.softmax(target_flat, dim=1)
    
    out_dict[suffix+"_kl_div"] = F.kl_div(orig_prob.log(), target_prob, reduction="sum")
    
    # MSE and MAE (work on flattened vectors)
    out_dict[suffix+"_mse"] = F.mse_loss(orig_flat, target_flat)
    
    return out_dict

def compare_representations(args,
                            model,
                            csul,
                            moco,
                            ul_img,
                            ul_label,
                            rt_img,
                            rt_label):
    
    csul_ul_loss, csul_rt_loss = contrastive_unlearn(csul,
                                    ul_img.clone().detach().to(args.device),
                                    ul_label.clone().detach().to(args.device),
                                    rt_img.clone().detach().to(args.device),
                                    rt_label.clone().detach().to(args.device),
                                    args.device)
    moco_ul_loss, moco_rt_loss = MoCo_unlearn(moco,
                            ul_img.clone().detach().to(args.device),
                            ul_label.clone().detach().to(args.device),
                            rt_img.clone().detach().to(args.device),
                            rt_label.clone().detach().to(args.device),
                            args.device)

    def _compare(img_, out_dict_, suffix):
        with torch.no_grad():
            model.eval()
            csul.model.eval()
            moco.encoder_q.eval()
            
            _, o_feat = model(img_.clone().detach().to(args.device))
            _, c_feat = csul.model(img_.clone().detach().to(args.device))
            _, m_feat = moco.encoder_q(img_.clone().detach().to(args.device))
            
            model.train()
            csul.model.train()
            moco.encoder_q.train()

        out_dict_.update(measure_differences(o_feat, c_feat, "csul_"+suffix))
        out_dict_.update(measure_differences(o_feat, m_feat, "moco_"+suffix))
    
    out_dict = dict()
    _compare(ul_img, out_dict, "ul")
    _compare(rt_img, out_dict, "rt")

    out_dict["csul_ul_loss"] = csul_ul_loss
    out_dict["csul_rt_loss"] = csul_rt_loss
    out_dict["moco_ul_loss"] = moco_ul_loss
    out_dict["moco_rt_loss"] = moco_rt_loss

    return out_dict

def main(args):
    wandb.init(project=args.wandb_project,
               config=args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model_Loader(args)(name=args.model,
                               num_classes=args.num_classes,
                               feat_dim=args.feat_dim,
                               )
    model = Checkpoint_Loader(args, model)
    
    trainloaders, testloaders = Data_Loader(args)

    csul = CSUL_Wrapper(args, model, device)
    moco = MoCo_Wrapper(args, model, device)
    model = model.to(device)

    for epoch in range(args.unlearn_epoch):
        rt_iter = iter(trainloaders[1])
        for ul_imgs, ul_labels in trainloaders[0]:
            try:
                rt_imgs, rt_labels = next(rt_iter)
            except StopIteration:
                rt_iter = iter(trainloaders[1])
                rt_imgs, rt_labels = next(rt_iter)

            if ul_imgs.shape[0] != rt_imgs.shape[0]:
                rt_imgs = rt_imgs[:ul_imgs.shape[0]]
                rt_labels = rt_labels[:ul_labels.shape[0]]
        
            out_dict = compare_representations(args, model, csul, moco, ul_imgs, ul_labels, rt_imgs, rt_labels)
            wandb.log(out_dict)
        
        csul_ul_acc, _ = evaluate(csul.model, trainloaders[0], device)
        csul_ts_acc, _ = evaluate(csul.model, testloaders[0], device)
        moco_ul_acc, _ = evaluate(moco.encoder_q, trainloaders[0], device)
        moco_ts_acc, _ = evaluate(moco.encoder_q, testloaders[0], device)

        wandb.log({"csul_ul_acc": csul_ul_acc,
                   "csul_ts_acc": csul_ts_acc,
                   "moco_ul_acc": moco_ul_acc,
                   "moco_ts_acc": moco_ts_acc})
 
    wandb.finish()


if __name__ == "__main__":
    parser = Untrain_Parser()
    parser.add_argument('--k', type=int, default=2000,
                       help='Number of samples in the queue (default: 2000)')
    args = parser.parse_args()
    main(args)
