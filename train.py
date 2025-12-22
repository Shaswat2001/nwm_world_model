import os
import yaml
import time
import wandb
import argparse
from copy import deepcopy
from models import CDiT_models
from data_loader import TrainingDatasetLoader
from diffusers.models import AutoencoderKL

import torch
from torch.utils.data import DataLoader, ConcatDataset

from diffusion import GaussianDiffusion
from utils import requires_grad, update_target_networks, transform

def main(args):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = args.global_seed
    torch.manual_seed(seed)

    with open("config/eval_cfg.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config
    
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    run = None
    if args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or config.get("run_name", None),
            tags=args.wandb_tags,
            config={**config, "epochs": args.epochs, "global_seed": args.global_seed},
        )

    os.makedirs(config['results_dir'], exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_dir = f"{config['results_dir']}/{config['run_name']}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints

    tokenizer = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    latent_size = config['image_size'] // 8

    num_cond = config['context_size']
    model = CDiT_models[config['model']](context_size=num_cond, input_size=latent_size, in_channels=4).to(device)
    target_model = deepcopy(model).to(device)  

    lr = float(config.get('lr', 1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    diffusion = GaussianDiffusion()

    train_dataset = []
    test_dataset = []

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                    goals_per_obs = int(data_config["goals_per_obs"])
                    if data_split_type == 'test':
                        goals_per_obs = 4 # standardize testing
                    
                    if "distance" in data_config:
                        min_dist_cat=data_config["distance"]["min_dist_cat"]
                        max_dist_cat=data_config["distance"]["max_dist_cat"]
                    else:
                        min_dist_cat=config["distance"]["min_dist_cat"]
                        max_dist_cat=config["distance"]["max_dist_cat"]

                    if "len_traj_pred" in data_config:
                        len_traj_pred=data_config["len_traj_pred"]
                    else:
                        len_traj_pred=config["len_traj_pred"]

                    dataset = TrainingDatasetLoader(
                        dataset_dir=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        # image_size=config["image_size"],
                        min_distance_cat=min_dist_cat,
                        max_distance_cat=max_dist_cat,
                        len_traj_pred=len_traj_pred,
                        context_size=config["context_size"],
                        normalize=config["normalize"],
                        goals_per_obs=goals_per_obs,
                        transform=transform,
                        traj_stride=1,
                    )
                    if data_split_type == "train":
                        train_dataset.append(dataset)
                    else:
                        test_dataset.append(dataset)
                    print(f"Dataset: {dataset_name} ({data_split_type}), size: {len(dataset)}")

    # combine all the datasets from different robots
    print(f"Combining {len(train_dataset)} train datasets.")
    print(f"Combining {len(train_dataset)} test datasets.")

    train_dataset = ConcatDataset(train_dataset)
    test_dataset = ConcatDataset(test_dataset)

    loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    model.train()  # important! This enables embedding dropout for classifier-free guidance
    target_model.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    global_step = 0
    for epoch in range(args.epochs):
        print(f"Running epoch : {epoch}")
        for x, y, rel_t in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            rel_t = rel_t.to(device, non_blocking=True)
            
            with torch.no_grad():
                step_start = time.time()
                # Map input images to latent space + normalize latents:
                B, T = x.shape[:2]
                x = x.flatten(0,1)
                x = tokenizer.encode(x).latent_dist.sample().mul_(0.18215)
                x = x.unflatten(0, (B, T))
            
            num_goals = T - num_cond
            x_start = x[:, num_cond:].flatten(0, 1)
            x_cond = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
            y = y.flatten(0, 1)
            rel_t = rel_t.flatten(0, 1)
            
            t = torch.randint(0, diffusion.diffusion_steps, (x_start.shape[0],), device=device)
            model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)
            loss_dict = diffusion.diffusion_loss(model, x_start, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            
            update_target_networks(target_model, model)

            global_step += 1
            running_loss += loss.item()
            log_steps += 1

            if (global_step % args.log_every) == 0:
                avg_loss = running_loss / max(log_steps, 1)
                lr_val = opt.param_groups[0]["lr"]
                step_time = time.time() - step_start

                print(f"[epoch {epoch:03d} | step {global_step:07d}] loss={avg_loss:.6f} lr={lr_val:.2e} t={step_time:.3f}s")

                if args.wandb:
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/lr": lr_val,
                            "perf/step_time_sec": step_time,
                            "train/epoch": epoch,
                            "train/step": global_step,
                        },
                        step=global_step,
                    )

                running_loss = 0
                log_steps = 0

    if args.wandb:
        wandb.finish()


@torch.no_grad
def evaluate():

    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=5000)

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="cdit-diffusion")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)


    args = parser.parse_args()

    main(args)