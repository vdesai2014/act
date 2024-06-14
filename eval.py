import os
import pickle
import time
import numpy as np
import torch
from utils import set_seed
from dynamixel import Dynamixel
from realsense import RealsenseCamera
from webcam import Webcam
import yaml
from policy import ACTPolicy    

def precise_sleep(duration):
    end_time = time.perf_counter() + duration
    while time.perf_counter() < end_time:
        pass

def eval_dyna(config):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    ckpt_name = config['ckpt_name']
    temporal_agg = False
    max_timesteps = 500
    # Initialize Dynamixel, RealsenseCamera, and Webcam
    dynamixel = Dynamixel()
    realsense_camera = RealsenseCamera(width=640, height=480, fps=30)
    webcam = Webcam(width=640, height=480, fps=30)  # shm parameter set to None

    # Load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = ACTPolicy(config)
    state_dict = torch.load(ckpt_path, map_location='cuda:0')['policy_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove the 'module.' prefix
        new_state_dict[new_key] = value
    state_dict = new_state_dict
    loading_status = policy.load_state_dict(state_dict)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    dynamixel.reset()
    time.sleep(2)
    query_frequency = 10  # policy_config['num_queries']

    while True:
        start_time = time.time()
        with torch.inference_mode():
            for t in range(max_timesteps):
                precise_sleep(0.092)
                if t % query_frequency == 0:
                    qpos_numpy = dynamixel.get_motor_positions()
                    realsense_image = realsense_camera.get_image()
                    webcam_image = webcam.get_image()

                    if realsense_image is None or webcam_image is None:
                        continue

                    # Stack images to form a tensor of shape [1, 2, 3, 480, 640]
                    images = torch.cat((realsense_image, webcam_image), dim=1).cuda()

                    qpos = pre_process(qpos_numpy)
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    print(f'Running inference at time {time.time() - start_time}')
                    all_actions = policy(qpos, images)
                    start_time = time.time()
                
                raw_action = all_actions[:, t % query_frequency]
                # Post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                # Set motor positions
                dynamixel.set_motor_positions(target_qpos)

def main():
    # Load config file
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(0)

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
        'weight_decay': config['weight_decay'],
        'state_dim': config['state_dim'],
        'position_embedding': config['position_embedding'],
        'dropout': config['dropout'],
        'pre_norm': config['pre_norm'],
        'masks': config['masks'],
        'clip_max_norm': config['clip_max_norm'],
        'lr_drop': config['lr_drop'],
        'dilation': config['dilation'],
        'camera_names' : config['camera_names'],
        'ckpt_dir' : config['ckpt_dir'],
        'ckpt_name' : config['ckpt_name']
    }
    eval_dyna(policy_config)

if __name__ == '__main__':
    main()
