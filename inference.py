import os
import yaml
import argparse
import numpy as np
from models import CDiT_models
from data_loader import EvalDatasetLoader
from diffusion import GaussianDiffusion
from diffusers.models import AutoencoderKL

import torch
from torch.utils.data import DataLoader
from utils import transform, save_image_rollouts, nwm_model_forward

def load_dataset(config, data_config):

    dataset_list = {}
    for dataset_name, dataset_param in data_config.items():

        dataset = EvalDatasetLoader(
            dataset_dir=dataset_param["data_folder"],
            data_split_folder=dataset_param["test"],
            dataset_name=dataset_name,
            # image_size=config["image_size"],
            min_distance_cat=config["eval_distance"]["eval_min_dist_cat"],
            max_distance_cat=config["eval_distance"]["eval_max_dist_cat"],
            len_traj_pred=config["eval_len_traj_pred"],
            context_size=config["context_size"],
            normalize=config["normalize"],
            goals_per_obs=4,
            transform=transform,
            traj_stride=config["traj_stride"]
        )

        dataloader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    pin_memory=True,
                    drop_last=False
                )

        dataset_list[dataset_name] = dataloader
    
    return dataset_list

def generate_rollout(args, output_dir, rollout_fps, idxs, all_models, obs_image, gt_image, delta, num_cond, device):
    rollout_stride = args.input_fps // rollout_fps
    gt_image = gt_image[:, rollout_stride-1::rollout_stride]
    delta = delta.unflatten(1, (-1, rollout_stride)).sum(2)
    curr_obs = obs_image.clone().to(device)
    
    for i in range(gt_image.shape[1]):
        curr_delta = delta[:, i:i+1].to(device)
        if args.create_gt:
            x_pred_pixels = gt_image[:, i].clone().to(device)
        else:
            x_pred_pixels = nwm_model_forward(all_models, curr_obs, curr_delta, rollout_stride, args.latent_size, num_cond=num_cond, num_goals=1, device=device)

        curr_obs = torch.cat((curr_obs, x_pred_pixels.unsqueeze(1)), dim=1) # append current prediction
        curr_obs = curr_obs[:, 1:] # remove first observation
        save_image_rollouts(output_dir, idxs, i, x_pred_pixels)

def generate_time(args, output_dir, idxs, all_models, obs_image, gt_output, delta, secs, num_cond, device):
    eval_timesteps = [sec*args.input_fps for sec in secs]
    for sec, timestep in zip(secs, eval_timesteps):
        curr_delta = delta[:, :timestep].sum(dim=1, keepdim=True)
        if args.create_gt:
            x_pred_pixels = gt_output[:, timestep-1].clone().to(device)
        else:
            x_pred_pixels = nwm_model_forward(all_models, obs_image, curr_delta, timestep, args.latent_size, num_cond=num_cond, num_goals=1, device=device)
        save_image_rollouts(output_dir, idxs, sec, x_pred_pixels)

def main(args):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.create_gt:
        args.save_dir = os.path.join("output", "gt")
    else:
        args.save_dir = os.path.join("output", args.exp_name)

    os.makedirs(args.save_dir, exist_ok=True)

    with open("config/eval_cfg.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config
    
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    data_loaders = load_dataset(config, config["eval_datasets"])

    if not args.create_gt:

        diffusion = GaussianDiffusion()
        tokenizer = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
        latent_size = config['image_size'] // 8
        args.latent_size = latent_size
        num_cond = config['context_size']
        model = CDiT_models[config['model']](context_size=num_cond, input_size=latent_size, in_channels=4).to(device)
        torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.eval()
        model = torch.compile(model)
    else:
        model = diffusion = tokenizer = None

    for dataset_name in data_loaders.keys():

        dataset_save_output_dir = os.path.join(args.save_dir, dataset_name)
        os.makedirs(dataset_save_output_dir, exist_ok=True)
        curr_data_loader = data_loaders[dataset_name]
        
        for (idxs, obs_image, gt_image, delta) in curr_data_loader:
            num_cond = config["context_size"]
            obs_image = obs_image[:, -num_cond:].to(device)
            gt_image = gt_image.to(device)
            
            if args.eval_type == 'rollout':
                for rollout_fps in args.rollout_fps_values:
                    curr_rollout_output_dir = os.path.join(dataset_save_output_dir, f'rollout_{rollout_fps}fps')
                    os.makedirs(curr_rollout_output_dir, exist_ok=True)
                    generate_rollout(args, curr_rollout_output_dir, rollout_fps, idxs, (model, tokenizer, diffusion), obs_image, gt_image, delta, num_cond, device)
            elif args.eval_type == 'time':
                secs = np.array([2**i for i in range(0, args.num_sec_eval)])
                curr_time_output_dir = os.path.join(dataset_save_output_dir, 'time')
                os.makedirs(curr_time_output_dir, exist_ok=True)
                generate_time(args, curr_time_output_dir, idxs, (model, tokenizer, diffusion), obs_image, gt_image, delta, secs, num_cond, device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="exp")
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--num_sec_eval", type=int, default=5)
    parser.add_argument("--input_fps", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--create_gt", action="store_true", help="Whether to load model or create ground trouth")
    parser.add_argument("--rollout_fps_values", type=str, default='1,4', help="")
    parser.add_argument("--eval_type", type=str, default=None, help="type of evaluation has to be either 'time' or 'rollout'")
    args = parser.parse_args()

    args.rollout_fps_values = [int(fps) for fps in args.rollout_fps_values.split(',')]

    main(args)