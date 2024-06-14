import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import wandb
from policy import ACTPolicy

from utils import (set_seed, make_optimizer, detach_dict, 
                   compute_dict_mean, forward_pass, plot_history, save_checkpoint)

import torch.distributed as dist

def train_bc(train_dataloader, config, is_ddp, rank):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']       
    policy_config = config['policy_config']

    set_seed(seed)

    policy = ACTPolicy(policy_config)
    if is_ddp:
        policy.cuda(rank)
        policy = torch.nn.parallel.DistributedDataParallel(policy, device_ids=[rank], find_unused_parameters=True)
    else:
        policy.cuda()
    optimizer = make_optimizer(policy, is_ddp)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    for epoch in tqdm(range(num_epochs)):
        if rank == 0:
            print(f'\nEpoch {epoch}')
            wandb.log({"epoch": epoch})

        # Training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # Backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
            
            if rank == 0:
                # Log training metrics to WandB
                wandb.log(forward_dict)

        epoch_summary = compute_dict_mean(train_history[(batch_idx + 1) * epoch: (batch_idx + 1) * (epoch + 1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if rank == 0 and epoch % 1000 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            save_checkpoint(ckpt_path, policy, optimizer, epoch, min_val_loss, is_ddp)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)  

    if rank == 0:
        ckpt_path = os.path.join(ckpt_dir, 'policy_last.ckpt')
        save_checkpoint(ckpt_path, policy, optimizer, epoch, min_val_loss, is_ddp)

        best_epoch, min_val_loss, best_state_dict = best_ckpt_info
        ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
        save_checkpoint(ckpt_path, policy, optimizer, epoch, min_val_loss, is_ddp)
        print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

        # Save training curves
        plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info