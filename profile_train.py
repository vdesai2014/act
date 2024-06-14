import torch
import os
import pickle
import yaml
import time
from tqdm import tqdm
from policy import ACTPolicy
from utils import load_data, set_seed, make_optimizer, detach_dict, compute_dict_mean, forward_pass

def train_bc(train_dataloader, config, is_ddp, rank):
    seed = config['seed']       
    policy_config = config['policy_config']

    set_seed(seed)

    policy = ACTPolicy(policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy, is_ddp)

    train_history = []

    warmup_epochs = 5

    for epoch in tqdm(range(warmup_epochs + 1)):
        if rank == 0:
            print(f'\nEpoch {epoch}')

        # Training
        policy.train()
        optimizer.zero_grad()
        forward_times = []
        
        for batch_idx, data in enumerate(train_dataloader):
            # Measure dataloading time
            #start_time = time.time()
            #data = next(iter(train_dataloader))
            #dataloading_time = time.time() - start_time
            #dataloading_times.append(dataloading_time)
            start_time = time.time()
            forward_dict = forward_pass(data, policy)
            # Backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
            forward_time = time.time() - start_time
            forward_times.append(forward_time)
            print('Mister Miyagi')

        epoch_summary = compute_dict_mean(train_history[(batch_idx + 1) * epoch: (batch_idx + 1) * (epoch + 1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        average_forward_time = sum(forward_times) / len(forward_times)
        print(f'Average forward time: {average_forward_time:.5f} seconds')

        if epoch == warmup_epochs:
            # Profile the backward pass after warmup
            with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
                for batch_idx, data in enumerate(train_dataloader):
                    forward_dict = forward_pass(data, policy)
                    loss = forward_dict['loss']
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    if batch_idx == 0:  # profile only the first batch after warmup
                        break
            print(prof.key_averages().table(sort_by="cuda_time_total"))

    return 

def main():
    # Load config file
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # command line parameters
    is_ddp = config['is_ddp']
    ckpt_dir = config['ckpt_dir']
    batch_size_train = config['batch_size_train']
    dataset_dir = config['dataset_dir']

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    rank = 0
    world_size = 1
    # Set seed
    set_seed(config['seed'] * rank)

    num_episodes = config['num_episodes']
    camera_names = config['camera_names']
    
    policy_config = {
        'lr': config['lr'],
        'num_queries': config['chunk_size'],
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

    train_dataloader, stats = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, rank, world_size, config['chunk_size'])

    # Save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    train_bc(train_dataloader, config, is_ddp, rank)

if __name__ == '__main__':
    main()
