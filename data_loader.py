import os
import yaml
import pickle
from utils import *

import torch
from torch.utils.data import Dataset

class BaseDatasetLoader(Dataset):

    def __init__(self, 
                 dataset_dir: str, 
                 data_split_folder: str,
                 context_size: int,
                 dataset_name: str,
                 min_distance_cat: int, 
                 max_distance_cat: int,
                 len_traj_pred: int, 
                 transform,
                 traj_stride: int,
                 goals_per_obs: int,
                 normalize: bool = True):
        
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name

        traj_path = os.path.join(data_split_folder, "traj_names.txt")
        
        with open(traj_path, "r") as f:
            traj_names = f.read().splitlines()
        
        self.traj_names = traj_names

        with open("config/dataset_cfg.yaml", "r") as f:
            all_data_config = yaml.safe_load(f)

        self.transform = transform
        self.normalize = normalize

        self.context_size = context_size
        self.min_distance_cat = min_distance_cat
        self.max_distance_cat = max_distance_cat
        self.len_traj_pred = len_traj_pred
        self.traj_stride = traj_stride
        self.goals_per_obs = goals_per_obs
        self.data_config = all_data_config[self.dataset_name]

        self.build_dataset()

        self.ACTION_STATS = {}
        for key in all_data_config['action_stats']:
            self.ACTION_STATS[key] = np.expand_dims(all_data_config['action_stats'][key], axis=0)


    def load_trajectory(self,traj_name):

        pickle_path = os.path.join(
            self.dataset_dir,
            traj_name,
            "traj_data.pkl"
        )

        with open(pickle_path, "rb") as f:
            trajectory_data = pickle.load(f)

        for k, v in trajectory_data.items():
            trajectory_data[k] = v.astype('float')

        return trajectory_data
    
    def __len__(self):
        return len(self.dataset)
    
    def build_dataset(self):

        self.dataset = []
        self.goals_index = []

        for trajectory_name in self.traj_names:

            trajectory_data = self.load_trajectory(trajectory_name)
            trajectory_len = len(trajectory_data["position"])

            for t in range(0, trajectory_len):
                self.goals_index.append((trajectory_name, t))

            begin = self.context_size - 1
            end = trajectory_len - self.len_traj_pred

            for current_time in range(begin, end, self.traj_stride):

                min_distance = max(self.min_distance_cat, -current_time)
                max_distance = min(self.max_distance_cat, trajectory_len - current_time - 1)
                self.dataset.append((trajectory_name, current_time, min_distance, max_distance))

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred + 1
        yaw = traj_data["yaw"][start_index:end_index]
        positions = traj_data["position"][start_index:end_index]
        goal_pos = traj_data["position"][goal_time]
        goal_yaw = traj_data["yaw"][goal_time]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            raise ValueError("is used?")
            # const_len = self.len_traj_pred + 1 - yaw.shape[0]
            # yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            # positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

        waypoints_pos = to_local_coords(positions, positions[0], yaw[0])
        waypoints_yaw = angle_difference(yaw[0], yaw)
        actions = np.concatenate([waypoints_pos, waypoints_yaw.reshape(-1, 1)], axis=-1)
        actions = actions[1:]
        
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])
        goal_yaw = angle_difference(yaw[0], goal_yaw)
        
        if self.normalize:
            actions[:, :2] /= self.data_config["metric_waypoint_spacing"]
            goal_pos[:, :2] /= self.data_config["metric_waypoint_spacing"]
        
        goal_pos = np.concatenate([goal_pos, goal_yaw.reshape(-1, 1)], axis=-1)
        return actions, goal_pos    

class TrainingDatasetLoader(BaseDatasetLoader):

    def __getitem__(self, i):

        trajectory_name, current_time, min_distance, max_distance = self.dataset[i]
        goal_offsets = np.random.randint(min_distance, max_distance, size=(self.goals_per_obs))
        goal_times = current_time + goal_offsets
        relative_time = goal_offsets.astype("float") / 128.0

        context_times = list(range(current_time - self.context_size + 1, current_time + 1))
        context = [(trajectory_name, t) for t in context_times] + [(trajectory_name, t) for t in goal_times]

        obs_image = torch.stack([self.transform(Image.open(get_image_path(self.dataset_dir, f, t))) for f, t in context])

        trajectory_data = self.load_trajectory(trajectory_name)
        _, goal_pos = self._compute_actions(trajectory_data, current_time, goal_times)
        goal_pos[:, :2] = normalize_data(goal_pos[:, :2], self.ACTION_STATS)

        return (
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_pos, dtype=torch.float32),
                torch.as_tensor(relative_time, dtype=torch.float32),
            )
    
class EvalDatasetLoader(BaseDatasetLoader):

    def __getitem__(self, i):

        trajectory_name, current_time,  min_distance, max_distance = self.dataset[i]
        context_times = list(range(current_time - self.context_size + 1, current_time + 1))
        goal_times = list(range(current_time + 1, current_time + self.len_traj_pred + 1))

        context = [(trajectory_name, t) for t in context_times]
        goals = [(trajectory_name, t) for t in goal_times]

        obs_image = torch.stack([self.transform(Image.open(get_image_path(self.dataset_dir, f, t))) for f, t in context])
        goal_image = torch.stack([self.transform(Image.open(get_image_path(self.dataset_dir, f, t))) for f, t in goals])

        trajectory_data = self.load_trajectory(trajectory_name)

        actions, _ = self._compute_actions(trajectory_data, current_time, goal_times)
        actions[:, :2] = normalize_data(actions[:, :2], self.ACTION_STATS)
        delta = get_delta_np(actions)

        return (
                torch.as_tensor([i], dtype=torch.float32),
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_image, dtype=torch.float32),
                torch.as_tensor(delta, dtype=torch.float32),
            )
