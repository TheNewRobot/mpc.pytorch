import logging
import math
import time
import logging
import math
import time
import os
import argparse
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*The reward returned by.*")
warnings.filterwarnings("ignore", message=".*The obs returned by the.*")

import gymnasium as gym  # Updated from gym to gymnasium
import numpy as np
import torch
import torch.autograd
# Remove the gymnasium logger import as we'll just use the standard Python logging
from mpc import mpc

# Set up argument parser
parser = argparse.ArgumentParser(description='MPC Pendulum')
parser.add_argument('--record', action='store_true', help='Enable video recording')
parser.add_argument('--video_dir', type=str, default='videos/', help='Directory to store videos')
parser.add_argument('--render', action='store_true', help='Enable rendering')
args = parser.parse_args()

# Create video directory if it doesn't exist
if args.record:
    os.makedirs(args.video_dir, exist_ok=True)
    print(f"Videos will be saved to: {args.video_dir}")

# Use standard Python logging instead of gymnasium logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

if __name__ == "__main__":
    ENV_NAME = "Pendulum-v1"  # Updated from v0 to v1
    TIMESTEPS = 10  # T
    N_BATCH = 1
    LQR_ITER = 5
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0

    class PendulumDynamics(torch.nn.Module):
        def forward(self, state, action):
            th = state[:, 0].view(-1, 1)
            thdot = state[:, 1].view(-1, 1)

            g = 10
            m = 1
            l = 1
            dt = 0.05

            u = action
            u = torch.clamp(u, -2, 2)

            newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
            newth = th + newthdot * dt
            newthdot = torch.clamp(newthdot, -8, 8)

            state = torch.cat((angle_normalize(newth), newthdot), dim=1)
            return state

    def angle_normalize(x):
        return (((x + math.pi) % (2 * math.pi)) - math.pi)

    # Create the environment with render_mode
    render_mode = "human" if args.render else None
    
    # In Gymnasium, we use kwargs instead of directly modifying env attributes
    env_kwargs = {"g": 10.0}  # Default value in Pendulum-v1
    
    # Creating the environment with specified render mode and disable the passive checker
    # which causes warnings about observation and reward types
    env = gym.make(ENV_NAME, render_mode=render_mode, disable_env_checker=True, **env_kwargs)
    
    # Initialize environment state
    downward_start = True
    observation, info = env.reset()
    
    # We need to set the state through the env's unwrapped state attribute
    if downward_start:
        env.unwrapped.state = np.array([np.pi, 1], dtype=np.float32)
        observation = env.unwrapped._get_obs()  # Update observation after changing state

    # Set up video recording if enabled
    if args.record:
        try:
            # Create a new environment with recording enabled
            env = gym.wrappers.RecordVideo(
                env,
                args.video_dir,
                disable_env_checker=True,  # Add this to disable warnings
                episode_trigger=lambda episode_id: True  # Record every episode
            )
            # Reset again after wrapping
            observation, info = env.reset()
            if downward_start:
                env.unwrapped.state = np.array([np.pi, 1], dtype=np.float32)
                observation = env.unwrapped._get_obs()
        except Exception as e:
            logger.error(f"Error setting up video recording: {e}")
            logger.warning("Continuing without video recording")

    nx = 2
    nu = 1

    u_init = None
    render = args.render
    retrain_after_iter = 50
    run_iter = 500

    # swingup goal (observe theta and theta_dt)
    goal_weights = torch.tensor((1., 0.1))  # nx
    goal_state = torch.tensor((0., 0.))  # nx
    ctrl_penalty = 0.001
    q = torch.cat((
        goal_weights,
        ctrl_penalty * torch.ones(nu)
    ))  # nx + nu
    px = -torch.sqrt(goal_weights) * goal_state
    p = torch.cat((px, torch.zeros(nu)))
    Q = torch.diag(q).repeat(TIMESTEPS, N_BATCH, 1, 1)  # T x B x nx+nu x nx+nu
    p = p.repeat(TIMESTEPS, N_BATCH, 1)
    cost = mpc.QuadCost(Q, p)  # T x B x nx+nu (linear component of cost)

    # Extract initial state from observation (in Gymnasium, observation is different from state)
    # Pendulum observation is [cos(theta), sin(theta), theta_dot]
    # We need to convert back to [theta, theta_dot] for our dynamics model
    def obs_to_state(obs):
        # Handle numpy array observations
        if isinstance(obs, np.ndarray):
            cos_th, sin_th, thdot = obs
            theta = math.atan2(sin_th, cos_th)
            return np.array([theta, thdot], dtype=np.float32)
        # Handle dictionary observations (in case the environment uses dict observation space)
        elif isinstance(obs, dict):
            # Extract the relevant values from the dictionary
            if 'cos_th' in obs and 'sin_th' in obs and 'thdot' in obs:
                theta = math.atan2(obs['sin_th'], obs['cos_th'])
                return np.array([theta, obs['thdot']], dtype=np.float32)
            else:
                # If the keys are different or unknown, just use the first two values
                values = list(obs.values())
                return np.array(values[:2], dtype=np.float32)

    # run MPC
    total_reward = 0
    for i in range(run_iter):
        # For debugging
        if i == 0:
            logger.debug(f"Observation type: {type(observation)}, shape: {np.shape(observation)}, value: {observation}")
            
        # Get current state directly from the environment's unwrapped state
        # This is more reliable than trying to convert from observation
        state = env.unwrapped.state
        state = torch.tensor(state, dtype=torch.float32).view(1, -1)
        
        command_start = time.perf_counter()
        # recreate controller using updated u_init
        ctrl = mpc.MPC(nx, nu, TIMESTEPS, u_lower=ACTION_LOW, u_upper=ACTION_HIGH, lqr_iter=LQR_ITER,
                       exit_unconverged=False, eps=1e-2,
                       n_batch=N_BATCH, backprop=False, verbose=0, u_init=u_init,
                       grad_method=mpc.GradMethods.AUTO_DIFF)

        # compute action based on current state, dynamics, and cost
        nominal_states, nominal_actions, nominal_objs = ctrl(state, cost, PendulumDynamics())
        action = nominal_actions[0]  # take first planned action
        u_init = torch.cat((nominal_actions[1:], torch.zeros(1, N_BATCH, nu)), dim=0)

        elapsed = time.perf_counter() - command_start
        
        # In Gymnasium step returns 5 values instead of 4
        observation, reward, terminated, truncated, info = env.step(action.detach().numpy())
        done = terminated or truncated
        
        # Make sure reward is a scalar
        if isinstance(reward, np.ndarray):
            reward = float(reward.item())
        
        total_reward += reward
        logger.debug("action taken: %.4f cost received: %.4f time taken: %.5fs", action, -reward, elapsed)

    logger.info("Total reward %f", total_reward)
    
    # Close the environment properly
    env.close()