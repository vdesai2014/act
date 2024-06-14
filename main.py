import torch
import os
import pickle
import yaml

from utils import load_data, set_seed, upload_to_s3
from train_bc import train_bc  

import wandb
import torch.distributed as dist

def main():
    # Load config file
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # command line parameters
    is_ddp = config['is_ddp']
    ckpt_dir = config['ckpt_dir']
    batch_size_train = config['batch_size_train']
    dataset_dir = config['dataset_dir']

    if is_ddp:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        device = torch.device('cuda', rank)
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        rank = 0
        world_size = 1

    if rank == 0:
        # Initialize WandB
        wandb.init(project=config['wandb']['project_name'], name=config['wandb']['experiment_name'])

    # Set seed
    set_seed(config['seed'] * rank)

    num_episodes = config['num_episodes']
    camera_names = config['camera_names']
    
    policy_config = {
        'lr': config['lr'],
        'num_queries': config['num_queries'],
        'kl_weight': config['kl_weight'],
        'hidden_dim': config['hidden_dim'],
        'dim_feedforward': config['dim_feedforward'],
        'lr_backbone': config['lr_backbone'],
        'backbone': config['backbone'],
        'enc_layers': config['enc_layers'],
        'dec_layers': config['dec_layers'],
        'nheads': config['nheads'],
        'camera_names': camera_names,
        'weight_decay': config['weight_decay'],
        'state_dim': config['state_dim'],
        'position_embedding': config['position_embedding'],
        'dropout': config['dropout'],
        'pre_norm': config['pre_norm'],
        'masks': config['masks'],
        'clip_max_norm': config['clip_max_norm'],
        'lr_drop': config['lr_drop'],
        'dilation': config['dilation']
    }

    config.update({
        'policy_config': policy_config,
        'device': device,
        'rank': rank,
        'world_size': world_size,
    })

    train_dataloader, stats = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, rank, world_size, config['num_queries'])

    # Save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, config, is_ddp, rank)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # Save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch {best_epoch}')

    # Upload to S3
    upload_to_s3(ckpt_path, config['aws'])

if __name__ == '__main__':
    main()
