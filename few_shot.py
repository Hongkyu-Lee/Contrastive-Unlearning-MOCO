import os
import torch
import wandb
import argparse
import sys
sys.path.append('.')
from utils import train_few_shot, evaluate_zero_shot
from model.vit import ViT
from datetime import datetime
from omegaconf import OmegaConf

def main(args):     
    # Load configuration
    cfg = OmegaConf.load(args.cfg)
    device = torch.device(cfg.device)

    wandb_name = f"few_shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _wandb = wandb.init(project=cfg.wandb.project,
                        name=wandb_name)
    
    # Load model
    model = ViT()
    model = model.to(device)

    test_acc = evaluate_zero_shot(cfg, model, device)
    print(f"Zero-shot test accuracy: {test_acc:.2f}%")

    # Train model
    model = train_few_shot(cfg, model, device)

    # save model
    save_path = os.path.join(cfg.save_path, wandb_name)
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, 'few_shot_model.pth'))

    _wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/few-shot.yaml')
    args = parser.parse_args()
    main(args)

