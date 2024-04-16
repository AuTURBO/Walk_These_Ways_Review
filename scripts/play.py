import isaacgym

assert isaacgym
import torch
import numpy as np
import os

from isaacgym import gymapi
import glob
import pickle as pkl
from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym.envs.base.base_task import BaseTask
import os
from tqdm import tqdm

def load_policy(full_path):

    body = torch.jit.load(full_path + '/checkpoints/body_latest.jit')
    adaptation_module = torch.jit.load(full_path + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy

def load_env(label, headless=False):

    # The script now expects label to be the directory name where parameters.pkl is located
    # If label is not the directory name, you'll need to adjust how you're setting label
    base_path = os.path.join(MINI_GYM_ROOT_DIR, "runs")
    full_path = os.path.join(base_path, label)

    # Print out the path to ensure it's correct
    parameters_file_path = os.path.join(full_path, "parameters.pkl")
    print(f"Looking for parameters.pkl at: {parameters_file_path}")

    # Check if the parameters file exists at the specified path
    if not os.path.isfile(parameters_file_path):
        raise FileNotFoundError(f"The file parameters.pkl does not exist at the path: {parameters_file_path}")

    # Load the parameters file
    with open(parameters_file_path, 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        # Update the Cfg object with the loaded configuration
        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)
    
    # load policy
    policy = load_policy(full_path)

    return env, policy

def play_go1(label, headless=True):
    # from ml_logger import logger
    # from pathlib import Path
    # from go1_gym import MINI_GYM_ROOT_DIR
    # import glob
    # import os
    #import keyboard
    #gym = BaseTask
    # label = "gait-conditioned-agility/pretrain-v0/train/025417.456545"

    env, policy = load_env(label, headless=headless)

    if hasattr(env, 'viewer') and env.viewer is not None:
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_W, "forward")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_S, "backward")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_A, "turn_left")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_D, "turn_right")
    
    num_eval_steps = 20000
    gaits = {"pronking": [0.0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    # pronk : all together
    # bound : front and hind
    # pace : right and left
    
    ####### 
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    joint_positions = np.zeros((num_eval_steps, 12))

    obs = env.reset()
    #state = env.acquire_actor_root_state_tensor(env)
    #print(state)

    for i in tqdm(range(num_eval_steps)):
        
        env.handle_keyboard_events()
        x_vel_cmd = env.commands[:, 0].item()
        yaw_vel_cmd = env.commands[:, 2].item()
        
        with torch.no_grad():
            actions = policy(obs)
            
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        #print(env.commands)
        obs, rew, done, info = env.step(actions)

        measured_x_vels[i] = env.base_lin_vel[0, 0]
        joint_positions[i] = env.dof_pos[0, :].cpu()

    # plot target and measured forward velocity
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    joint_names = [
        'FL_hip_joint', 'RL_hip_joint', 'FR_hip_joint', 'RR_hip_joint',
        'FL_thigh_joint', 'RL_thigh_joint', 'FR_thigh_joint', 'RR_thigh_joint',
        'FL_calf_joint', 'RL_calf_joint', 'FR_calf_joint', 'RR_calf_joint'
    ]


    for i, joint_name in enumerate(joint_names):
        axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions[:, i], label=joint_name)

    axs[1].legend()
    axs[1].set_title("Joint Positions")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Joint Position (rad)")
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    label = "gait-conditioned-agility/pretrain-v0/train/025417.456545"
    play_go1(label=label, headless=False)
