"""
MPC Pendulum Controller with Known Dynamics and Monitoring
=========================================================

This script implements Model Predictive Control (MPC) for the pendulum swing-up
problem using an explicitly defined dynamics model based on physics equations.
Includes a monitoring system for plotting state variables and control inputs.

Purpose:
- Demonstrate MPC control of a pendulum using known system dynamics
- Swing up a pendulum from downward position to upright balanced position
- Optimize control actions over a prediction horizon to minimize costs
- Monitor and plot key variables during simulation

Features:
- Physics-based pendulum dynamics model
- Configurable rendering via command-line argument
- Real-time plotting of state variables, control inputs, and rewards
- Quadratic cost function penalizing deviation from goal and control effort
- Model predictive control with auto-differentiation gradient method

Usage:
    python pendulum_mpc_with_monitor.py [--render] [--plot]

Arguments:
    --render    Enable visual rendering of the pendulum
    --plot      Enable real-time plotting of variables

"""

import logging
import math
import time
import argparse
import warnings

# Suppress specific warnings
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*The reward returned by.*")
warnings.filterwarnings("ignore", message=".*The obs returned by the.*")

import gymnasium as gym
import numpy as np
import torch
from mpc import mpc

# Import our monitoring module
from monitor import Monitor

# Set up argument parser
parser = argparse.ArgumentParser(description='MPC Pendulum with Monitoring')
parser.add_argument('--render', action='store_true', help='Enable rendering')
parser.add_argument('--plot', action='store_true', help='Enable real-time plotting')
args = parser.parse_args()

# Use standard Python logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

if __name__ == "__main__":
    ENV_NAME = "Pendulum-v1"
    TIMESTEPS = 10
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

    # Choose render_mode based on args
    if args.render:
        render_mode = "human"      # Display on screen
    else:
        render_mode = None         # No rendering
    
    # Environment kwargs
    env_kwargs = {"g": 9.81}
    
    # Create base environment
    env = gym.make(ENV_NAME, render_mode=render_mode, disable_env_checker=True, **env_kwargs)
    
    # Initialize monitoring system
    monitor = Monitor(enabled=args.plot, update_freq=5)
    
    # Initialize environment state
    downward_start = True
    observation, info = env.reset()
    
    if downward_start:
        env.unwrapped.state = np.array([np.pi, 1], dtype=np.float32)
        observation = env.unwrapped._get_obs()

    nx = 2
    nu = 1

    u_init = None
    retrain_after_iter = 50
    run_iter = 500

    # swingup goal (observe theta and theta_dt)
    goal_weights = torch.tensor((1., 0.1))
    goal_state = torch.tensor((0., 0.))
    ctrl_penalty = 0.001
    q = torch.cat((
        goal_weights,
        ctrl_penalty * torch.ones(nu)
    ))
    px = -torch.sqrt(goal_weights) * goal_state
    p = torch.cat((px, torch.zeros(nu)))
    Q = torch.diag(q).repeat(TIMESTEPS, N_BATCH, 1, 1)
    p = p.repeat(TIMESTEPS, N_BATCH, 1)
    cost = mpc.QuadCost(Q, p)

    total_reward = 0
    computation_times = []

    for i in range(run_iter):
        state = env.unwrapped.state
        state_tensor = torch.tensor(state, dtype=torch.float32).view(1, -1)
        
        # Only render if render mode is set
        if render_mode == "human":
            env.render()

        command_start = time.perf_counter()
        ctrl = mpc.MPC(nx, nu, TIMESTEPS, u_lower=ACTION_LOW, u_upper=ACTION_HIGH, lqr_iter=LQR_ITER,
                       exit_unconverged=False, eps=1e-2,
                       n_batch=N_BATCH, backprop=False, verbose=0, u_init=u_init,
                       grad_method=mpc.GradMethods.AUTO_DIFF)

        nominal_states, nominal_actions, nominal_objs = ctrl(state_tensor, cost, PendulumDynamics())
        action = nominal_actions[0]
        u_init = torch.cat((nominal_actions[1:], torch.zeros(1, N_BATCH, nu)), dim=0)

        elapsed = time.perf_counter() - command_start
        computation_times.append(elapsed)
        
        observation, reward, terminated, truncated, info = env.step(action.detach().numpy())
        done = terminated or truncated
        
        if isinstance(reward, np.ndarray):
            reward = float(reward.item())
        total_reward += reward
        
        # Calculate angle from cos/sin components if needed
        angle = np.arctan2(observation[1], observation[0])  # If observation is [cos(θ), sin(θ), θ̇]
        
        # Update monitor with current variables
        monitor.update(
            theta=state[0],                  # Current angle (in radians)
            theta_dot=state[1],              # Angular velocity
            control=action.item(),           # Control input
            reward=reward,                   # Instantaneous reward
            cu_reward=total_reward,          # Cumulative rewar
        )
        
        logger.debug("action taken: %.4f cost received: %.4f time taken: %.5fs", action, -reward, elapsed)
        
        if done:
            print("Environment has reached its end state!")
            break

    logger.info("Total reward %f", total_reward)
    logger.info("Average computation time: %f seconds", np.mean(computation_times))
    
    # Save plots if monitoring was enabled
    if args.plot:
        monitor.save_plots(directory="pendulum_mpc_plots", filename_prefix="pendulum_run")
        print(f"Plots saved to 'pendulum_mpc_plots' directory")
    
    # Close the monitor and environment properly
    monitor.close()
    env.close()