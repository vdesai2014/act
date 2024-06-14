import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from einops import rearrange
from copy import deepcopy
import boto3
import threading
import yaml
import h5py
from torch.utils.data import DataLoader, DistributedSampler
import subprocess

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, num_queries):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.num_queries = num_queries
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.h5')
        with h5py.File(dataset_path, 'r') as root:
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            action = root['/action'][start_ts:]
            action_len = episode_len - start_ts
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        action_data = action_data[:self.num_queries, :]
        is_pad = is_pad[:self.num_queries]
        image_data = image_data.float()
        action_data = action_data.float()
        qpos_data = qpos_data.float()
        return image_data, qpos_data, action_data, is_pad

def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []

    # Collect all qpos and action data without stacking them
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.h5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            action = root['/action'][()]
            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))

    # Concatenate all qpos and action data horizontally
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # Normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # Normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping
    print(f"action_mean: {action_mean}")
    print(f"action_std: {action_std}")
    print(f"qpos_mean: {qpos_mean}")
    print(f"qpos_std: {qpos_std}")
    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": all_qpos_data.numpy()
    }

    return stats

def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, rank, world_size, num_queries):
    print(f'\nData from: {dataset_dir}\n')
    train_indices = np.random.permutation(num_episodes)

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, num_queries)

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size_train)
    else:
        train_sampler = None
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)

    return train_dataloader, norm_stats

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def make_optimizer(policy, is_ddp):
    if is_ddp:
        optimizer = policy.module.configure_optimizers()
    else:
        optimizer = policy.configure_optimizers()
    return optimizer

def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None

def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # Save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')

def upload_to_s3(local_folder, aws_config):
    def _sync():
        bucket_name = aws_config['s3_bucket_name']
        s3_folder = aws_config['s3_folder']
        s3_path = f's3://{bucket_name}/{s3_folder}/'

        # Construct the aws s3 sync command
        command = [
            'aws', 's3', 'sync',
            local_folder,
            s3_path,
            '--exact-timestamps'
        ]

        # Set environment variables for AWS credentials
        env = os.environ.copy()
        env['AWS_ACCESS_KEY_ID'] = aws_config['access_key_id']
        env['AWS_SECRET_ACCESS_KEY'] = aws_config['secret_access_key']
        env['AWS_DEFAULT_REGION'] = aws_config.get('region', 'us-west-2')

        # Run the sync command
        result = subprocess.run(command, env=env, capture_output=True, text=True)

        if result.returncode == 0:
            print(f'Successfully synced {local_folder} to {s3_path}')
        else:
            print(f'Error syncing {local_folder} to {s3_path}: {result.stderr}')

    # Create a thread to run the _sync function
    thread = threading.Thread(target=_sync)
    thread.start()

def save_checkpoint(ckpt_path, policy, optimizer, epoch, min_val_loss, is_ddp):
    checkpoint = {
        'epoch': epoch,
        'policy_state_dict': policy.module.state_dict() if is_ddp else policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'min_val_loss': min_val_loss
    }
    torch.save(checkpoint, ckpt_path)