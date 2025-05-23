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

class MOCO_Unlearn(nn.Module):
    def __init__(self, args, model, device):
        super(MOCO_Unlearn, self).__init__()
        self.args = args
        self.temperature = args.temp
        self.device = device
        self.retain_ratio = args.retain_sampling_freq
        self.dim = args.feat_dim
        self.K = args.k
        self.loss_ratio = args.loss_ratio
        self.batch_size = args.batch_size
        self.CT_ratio = args.CT_ratio
        self.CE_ratio = args.CE_ratio
        self.loss_type = args.loss_type
        self.loss_threshold = args.loss_threshold
        self.encoder_q = copy.deepcopy(model)
        self.encoder_k = copy.deepcopy(model)
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("abs_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("label_queue", torch.ones(self.K)*-1)
        self.register_buffer("label_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("abs_label_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def unlearn(self, model, unlearn_loader, retain_loader, optim, criterion):
        
        for ul_imgs, ul_labels in tqdm(unlearn_loader):
            retain_iter = iter(retain_loader)
            
            ul_imgs = ul_imgs.to(self.device)
            ul_labels = ul_labels.to(self.device)
            ul_labels = ul_labels.contiguous().view(-1, 1)
            batch_size = ul_imgs.shape[0]
            for _ in range(self.retain_ratio):
                rt_imgs, rt_labels = next(retain_iter)
                rt_imgs = rt_imgs.to(self.device)
                rt_labels = rt_labels.to(self.device)

                if rt_imgs.shape[0] != batch_size:
                    rt_imgs = rt_imgs[:batch_size]
                    rt_labels = rt_labels[:batch_size]

                imgs = torch.cat([ul_imgs, rt_imgs.clone().detach().to(self.device)])
                logits, feats = self.encoder_q(imgs)
                ul_logits, ul_feats = logits[:ul_imgs.shape[0]], feats[:ul_imgs.shape[0]]
                rt_logits, _ = logits[ul_imgs.shape[0]:], feats[ul_imgs.shape[0]:]
                # compute key features
                with torch.no_grad():
                    # self._momentum_update_key_encoder()
                    # rt_imgs, idx_unshuffle = self._batch_shuffle_ddp(rt_imgs)
                    _, rt_feats = self.encoder_q(rt_imgs.clone().detach().to(self.device))

                # compute logits
                # enqueue & dequeue first.
                self._dequeue_and_enqueue(rt_feats)
                self._dequeue_and_enqueue_labels(rt_labels.clone().contiguous().view(-1, 1))
                print(self.queue.shape, self.label_queue.shape)

                if int(self.abs_queue_ptr) < self.K:
                    _current_queue = int(self.abs_queue_ptr)
                    logits = torch.einsum("nc,ck->nk", [ul_feats, self.queue.clone().detach().to(self.device)[:, :_current_queue]])
                    pos_mask = torch.eq(ul_labels, self.label_queue.clone().detach().to(self.device)[:_current_queue])
                else:
                    logits = torch.einsum("nc,ck->nk", [ul_feats, self.queue.clone().detach().to(self.device)])
                    pos_mask = ~torch.eq(ul_labels, self.label_queue.clone().detach().to(self.device))
                
                logits /= self.temperature

                
                ul_loss = self.criterion(logits, pos_mask)
                rt_loss = criterion(rt_logits, rt_labels)
                if self.loss_type == "combined":
                    optim.zero_grad()
                    loss = self.CT_ratio * ul_loss + self.CE_ratio * rt_loss
                    loss.backward()
                    optim.step()
                elif self.loss_type == "orthogonal":
                    ul_loss, rt_loss = self.orthogonal_loss(ul_loss, rt_loss, model, optim)

                wandb.log({"ul_loss": ul_loss, "rt_loss": rt_loss, "queue_size": int(self.abs_queue_ptr)})

    def orthogonal_loss(self, ul_loss, rt_loss, model, optim):
        # Get gradients independently for each loss
        optim.zero_grad()
        ul_loss.backward(retain_graph=True)
        ul_grads = []
        for param in model.parameters():
            if param.grad is not None:
                ul_grads.append(param.grad.clone())
            else:
                ul_grads.append(None)
        
        optim.zero_grad()
        rt_loss.backward()
        rt_grads = []
        for param in model.parameters():
            if param.grad is not None:
                rt_grads.append(param.grad.clone())
            else:
                rt_grads.append(None)

        total_grads = list()
        for _ul_grad, _rt_grad in zip(ul_grads, rt_grads):
            if _ul_grad is None:
                total_grads.append(_rt_grad)
            else:
                mask = _ul_grad < self.loss_threshold
                total_grads.append((~mask) * _ul_grad + mask * _rt_grad)
        # Update model parameters with total gradients
        for param, grad in zip(model.parameters(), total_grads):
            param.grad = grad
        optim.step()

        # Calculate average gradients
        avg_ul_grad = 0
        avg_rt_grad = 0
        count = 0
        
        for ul_g, rt_g in zip(ul_grads, rt_grads):
            if ul_g is not None and rt_g is not None:
                avg_ul_grad += torch.mean(torch.abs(ul_g))
                avg_rt_grad += torch.mean(torch.abs(rt_g))
                count += 1
                
        if count > 0:
            avg_ul_grad = avg_ul_grad / count
            avg_rt_grad = avg_rt_grad / count
        
        return avg_ul_grad, avg_rt_grad

    def criterion(self, logits, mask):
        # Compute log softmax

        log_softmax = logits - torch.log(torch.sum(torch.exp(logits), dim=1, keepdim=True))
        
        nll = -log_softmax[mask]
        # Compute negative log likelihood loss
        
        # Return mean loss
        return torch.mean(nll)

        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, feats):
        batch_size = feats.shape[0]
        ptr = int(self.queue_ptr)
        abs_ptr = int(self.abs_queue_ptr)
        
        # Handle wraparound
        if ptr + batch_size > self.K:
            first_part_size = self.K - ptr
            self.queue[:, ptr:] = feats[:first_part_size].T
            self.queue[:, :batch_size - first_part_size] = feats[first_part_size:].T
        else:
            self.queue[:, ptr:ptr + batch_size] = feats.T
        
        self.queue_ptr[0] = (ptr + batch_size) % self.K
        if abs_ptr < self.K:
            self.abs_queue_ptr[0] = (abs_ptr + batch_size)

    @torch.no_grad()
    def _dequeue_and_enqueue_labels(self, labels):
        batch_size = labels.shape[0]
        ptr = int(self.label_queue_ptr)
        abs_ptr = int(self.abs_label_queue_ptr)

        # Handle wraparound
        if ptr + batch_size > self.K:
            first_part_size = self.K - ptr
            self.label_queue[ptr:] = labels[:first_part_size].squeeze()
            self.label_queue[:batch_size - first_part_size] = labels[first_part_size:].squeeze()
        else:
            self.label_queue[ptr:ptr + batch_size] = labels.squeeze()
        
        # Update pointer
        self.label_queue_ptr[0] = (ptr + batch_size) % self.K
        if abs_ptr < self.K:
            self.abs_label_queue_ptr[0] = (abs_ptr + batch_size)


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


def run(args):

    wandb.init(project=args.wandb_project,
               config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model_Loader(args)(name=args.model,
                               num_classes=args.num_classes,
                               feat_dim=args.feat_dim,
                               )
    model = Checkpoint_Loader(args, model)
    model = model.to(device)
    trainloaders, testloaders = Data_Loader(args)

    unlearner = MOCO_Unlearn(args, device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)

    for epoch in range(args.unlearn_epoch):
        unlearner.unlearn(model,
                          unlearn_loader=trainloaders[0],
                          retain_loader=trainloaders[1],
                          optim=optim,
                          criterion=criterion)
        
        acc, conf = evaluate(model, testloaders[0], device)
        ul_acc, ul_conf = evaluate(model, trainloaders[0], device)
        wandb.log({"test_acc": acc, "test_conf": conf, "ul_acc": ul_acc, "ul_conf": ul_conf})

    wandb.finish()

if __name__ == "__main__":
    parser = Untrain_Parser()
    parser.add_argument('--loss_ratio', type=float, default=0.5,
                       help='Ratio between unlearning loss and retain loss (default: 0.5)')
    parser.add_argument('--k', type=int, default=2000,
                       help='Number of samples in the queue (default: 2000)')
    parser.add_argument('--loss_type', type=str, default="combined",
                       help='Loss type (default: combined)')
    parser.add_argument('--loss_threshold', type=float, default=1e-3,
                       help='Loss threshold for orthogonal loss (default: 1e-3)')
    args = parser.parse_args()
    run(args)