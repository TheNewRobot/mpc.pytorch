"""
Fixed Pendulum Training Script - Key Parameter Corrections
==========================================================

Main fixes:
1. Increased control penalty to prevent saturation
2. Better goal weights for swing-up behavior  
3. More diverse training data collection
4. Fixed MPC parameters matching working reference
"""

import logging
import math
import time
import argparse
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings("ignore", message=".*The reward returned by.*")
warnings.filterwarnings("ignore", message=".*The obs returned by the.*")

import gymnasium as gym
import numpy as np
import torch
from mpc import mpc
from monitor import Monitor
from simple_training_plots import SimpleTrainingPlotter

parser = argparse.ArgumentParser(description='Fixed MPC Pendulum Training')
parser.add_argument('--render', action='store_true', help='Enable rendering')
parser.add_argument('--plot', action='store_true', help='Enable real-time plotting')
parser.add_argument('--save-model', action='store_true', help='Save model weights after training')
parser.add_argument('--base-dir', type=str, default='model_weights', help='Base directory for saving model weights')
parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
args = parser.parse_args()

if args.device == 'auto':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = torch.device(args.device)

print(f"Using device: {DEVICE}")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s %(asctime)s] %(message)s', datefmt='%H:%M:%S')

if args.save_model:
    RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    RUN_DIR = os.path.join(args.base_dir, RUN_TIMESTAMP)
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"Model weights will be saved to: {RUN_DIR}")
else:
    RUN_DIR = None

def angle_normalize(x):
    if torch.is_tensor(x):
        return torch.atan2(torch.sin(x), torch.cos(x))
    else:
        return math.atan2(math.sin(x), math.cos(x))

def save_model_weights(model, params):
    """Save model weights and training parameters."""
    if not args.save_model or RUN_DIR is None:
        return
    
    # Save model weights
    model_path = os.path.join(RUN_DIR, "network_weights.pt")
    torch.save(model.state_dict(), model_path)
    
    # Add last update timestamp to params
    params["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save parameters as JSON
    params_path = os.path.join(RUN_DIR, "training_params.json")
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"Model saved to {RUN_DIR}")

def obs_to_state(obs):
    if torch.is_tensor(obs):
        theta = torch.atan2(obs[..., 1], obs[..., 0])
        theta_dot = obs[..., 2]
        return torch.stack([theta, theta_dot], dim=-1)
    else:
        theta = np.arctan2(obs[1], obs[0])
        theta_dot = obs[2]
        return np.array([theta, theta_dot])

class FixedPendulumDynamics(torch.nn.Module):
    def __init__(self, network, action_low=-2.0, action_high=2.0):
        super().__init__()
        self.network = network
        self.action_low = action_low
        self.action_high = action_high
        self.max_thdot = 8.0
        
    def forward(self, state, action):
        # Handle dimensions
        if len(state.shape) == 1:
            state = state.view(1, -1)
        if len(action.shape) == 1:
            action = action.view(1, -1)
        if action.shape[1] > 1:
            action = action[:, 0].view(-1, 1)
            
        state = state.to(DEVICE)
        action = action.to(DEVICE)
        
        # No inplace operations - create new tensors
        u = torch.clamp(action, self.action_low, self.action_high)
        
        # Create new tensor instead of modifying in place
        theta_normalized = angle_normalize(state[:, 0:1])
        theta_dot = state[:, 1:2]
        normalized_state = torch.cat([theta_normalized, theta_dot], dim=1)
        
        xu = torch.cat((normalized_state, u), dim=1)
        next_state = self.network(xu)
        
        # Create new tensors instead of inplace modification
        next_theta = angle_normalize(next_state[:, 0:1])
        next_theta_dot = torch.clamp(next_state[:, 1:2], -self.max_thdot, self.max_thdot)
        
        return torch.cat([next_theta, next_theta_dot], dim=1)

def true_dynamics_tensor(state, action, dt=0.05, g=9.81, m=1.0, l=1.0):
    """Correct pendulum dynamics that match Gym Pendulum-v1"""
    if len(state.shape) == 1:
        state = state.view(1, -1)
    if len(action.shape) == 1:
        action = action.view(1, -1)
    if action.shape[1] > 1:
        action = action[:, 0].view(-1, 1)
        
    theta = state[:, 0:1]
    theta_dot = state[:, 1:2]
    u = torch.clamp(action, -2.0, 2.0)
    
    # CORRECT Gym Pendulum physics equations
    theta_ddot = -3 * g / (2 * l) * torch.sin(theta + math.pi) + 3.0 / (m * l ** 2) * u
    
    new_theta_dot = theta_dot + theta_ddot * dt
    new_theta_dot = torch.clamp(new_theta_dot, -8.0, 8.0)
    new_theta = theta + new_theta_dot * dt
    
    # Normalize angle to [-pi, pi]
    new_theta = torch.atan2(torch.sin(new_theta), torch.cos(new_theta))
    
    return torch.cat([new_theta, new_theta_dot], dim=1)

def train_network(network, dataset, device, epochs=200, lr=0.001):
    if dataset is None or len(dataset) < 10:
        return 0.0
    
    states = dataset[:, :2].to(device)
    actions = dataset[:, 2:3].to(device)
    
    true_next_states = []
    for i in range(len(states)):
        true_next = true_dynamics_tensor(states[i:i+1], actions[i:i+1])
        true_next_states.append(true_next)
    
    true_next_states = torch.cat(true_next_states, dim=0)
    
    network.train()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        xu = torch.cat([states, actions], dim=1)
        predicted_next_states = network(xu)
        
        angle_diff = angle_normalize(predicted_next_states[:, 0] - true_next_states[:, 0])
        velocity_diff = predicted_next_states[:, 1] - true_next_states[:, 1]
        loss = torch.mean(angle_diff**2 * 10.0 + velocity_diff**2)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    network.eval()
    return loss.item()

def collect_diverse_bootstrap_data(env, steps=1000):
    """Collect diverse training data focusing on swing-up dynamics"""
    data = []
    obs, _ = env.reset()
    
    for i in range(steps):
        state = obs_to_state(obs)
        theta = angle_normalize(state[0])
        theta_dot = state[1]
        
        # More strategic action selection for swing-up
        if i < steps // 4:
            # Random exploration
            action = np.random.uniform(-2.0, 2.0)
        elif i < steps // 2:
            # Energy-based swing-up actions
            if abs(theta) > np.pi/2:  # In lower half, pump energy
                action = np.sign(theta_dot * np.cos(theta)) * np.random.uniform(1.0, 2.0)
            else:  # In upper half, be more careful
                action = np.random.uniform(-1.0, 1.0)
        elif i < 3 * steps // 4:
            # Focus on transitions between quadrants
            if abs(theta) > 2.0:  # Near bottom
                action = np.random.uniform(1.0, 2.0) * np.sign(theta_dot)
            else:
                action = np.random.uniform(-2.0, 2.0)
        else:
            # Mixed strategy with emphasis on control near upright
            if abs(theta) < 1.0:  # Near upright
                action = np.random.uniform(-1.0, 1.0)
            else:
                action = np.random.uniform(-2.0, 2.0)
        
        next_obs, reward, terminated, truncated, _ = env.step([action])
        data.append([state[0], state[1], action])
        
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    
    return data

def main():
    global RUN_DIR, RUN_TIMESTAMP
    
    plotter = SimpleTrainingPlotter(save_dir=RUN_DIR if args.save_model else "training_plots")

    # FIXED: Match exact parameters from working reference script
    G = 9.81
    TIMESTEPS = 10  # Same as reference
    N_BATCH = 1
    LQR_ITER = 5    # Same as reference  
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0
    DTYPE = torch.double
    
    GOAL_STATE = torch.tensor([0.0, 0.0], dtype=DTYPE, device=DEVICE)
    
    # FIXED: Use exact same goal weights as working reference
    GOAL_WEIGHTS = torch.tensor([1.0, 0.1], dtype=DTYPE, device=DEVICE)
    
    H_UNITS = 64
    BOOTSTRAP_ITER = 1500  # More diverse data
    RUN_ITER = 4000        # More training time
    
    nx, nu = 2, 1
    network = torch.nn.Sequential(
        torch.nn.Linear(nx + nu, H_UNITS),
        torch.nn.ReLU(),
        torch.nn.Linear(H_UNITS, H_UNITS),
        torch.nn.ReLU(),
        torch.nn.Linear(H_UNITS, nx)
    ).to(dtype=DTYPE, device=DEVICE)
    
    # Ensure network parameters require gradients
    for param in network.parameters():
        param.requires_grad_(True)
    
    dynamics = FixedPendulumDynamics(network, ACTION_LOW, ACTION_HIGH)
    
    env = gym.make("Pendulum-v1", render_mode="human" if args.render else None, g=G)
    monitor = Monitor(enabled=args.plot, update_freq=1)
    
    # FIXED: Better bootstrap data collection
    logger.info(f"Collecting diverse bootstrap data for {BOOTSTRAP_ITER} steps...")
    bootstrap_data = collect_diverse_bootstrap_data(env, BOOTSTRAP_ITER)
    
    dataset = torch.tensor(bootstrap_data, dtype=DTYPE)
    logger.info("Training network on bootstrap data...")
    train_loss = train_network(network, dataset, DEVICE, epochs=2000)  # More training
    logger.info(f"Bootstrap training completed. Final loss: {train_loss:.6f}")
    
    # Save model after bootstrap training
    if args.save_model:
        training_params = {
            "dataset_size": len(dataset),
            "error_metrics": {
                "bootstrap_loss": train_loss
            },
            "hyperparameters": {
                "hidden_units": H_UNITS,
                "bootstrap_iterations": BOOTSTRAP_ITER,
                "learning_rate": 0.001,
                "g": G,
                "action_limits": [ACTION_LOW, ACTION_HIGH],
                "goal_weights": GOAL_WEIGHTS.cpu().tolist()
            },
            "training_info": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "training_type": "bootstrap",
                "run_timestamp": RUN_TIMESTAMP if args.save_model else "unknown"
            }
        }
        save_model_weights(network, training_params)
    
    CTRL_PENALTY = 0.1  # 100x higher than original (was 0.001)
    q = torch.cat((GOAL_WEIGHTS, CTRL_PENALTY * torch.ones(nu, dtype=DTYPE, device=DEVICE)))
    px = -torch.sqrt(GOAL_WEIGHTS) * GOAL_STATE
    p = torch.cat((px, torch.zeros(nu, dtype=DTYPE, device=DEVICE)))
    Q = torch.diag(q).repeat(TIMESTEPS, N_BATCH, 1, 1)
    p = p.repeat(TIMESTEPS, N_BATCH, 1)
    cost = mpc.QuadCost(Q, p)
    
    # Main training loop
    obs, _ = env.reset()
    total_reward = 0
    success_count = 0
    data_buffer = []
    RETRAIN_INTERVAL = 200  # Less frequent retraining
    mpc_failures = 0
    u_init = None

    logger.info("Starting MPC control loop...")

    for i in range(RUN_ITER):
        state = obs_to_state(obs)
        state_tensor = torch.tensor(state, dtype=DTYPE, device=DEVICE).view(1, -1)
        
        if args.render:
            env.render()
        
        try:

            ctrl = mpc.MPC(nx, nu, TIMESTEPS,
                        u_lower=ACTION_LOW, u_upper=ACTION_HIGH,
                        lqr_iter=LQR_ITER,
                        exit_unconverged=False,
                        eps=1e-2,
                        n_batch=N_BATCH,
                        backprop=False,
                        verbose=-1,
                        u_init=u_init,
                        grad_method=mpc.GradMethods.AUTO_DIFF)
            
            nominal_states, nominal_actions, _ = ctrl(state_tensor, cost, dynamics)
            action = nominal_actions[0].cpu().detach().numpy().flatten()[0]
            
            # Update u_init for next iteration
            u_init = torch.cat((nominal_actions[1:], torch.zeros(1, N_BATCH, nu, dtype=DTYPE, device=DEVICE)), dim=0)
            
            mpc_success = True
            
        except Exception as e:
            # Better fallback controller for large angles
            theta_error = angle_normalize(state[0])
            theta_dot = state[1]
            
            # Energy-based control for swing-up with smoother transitions
            current_energy = 0.5 * theta_dot**2 + 9.81 * (1 - math.cos(theta_error))
            desired_energy = 9.81 * 2.0  # Energy needed to reach top
            
            if current_energy < desired_energy * 0.8:  # Need more energy
                # Pump energy in direction of motion
                action = np.sign(theta_dot * np.cos(theta_error)) * 1.5
            else:  # Near enough energy, try to control
                # PD control around upright
                action = -2.0 * theta_error - 1.0 * theta_dot
            
            action = np.clip(action, ACTION_LOW, ACTION_HIGH)
            mpc_success = False
            mpc_failures += 1
        
        next_obs, reward, terminated, truncated, _ = env.step([action])
        total_reward += reward
        
        # Store data for retraining
        data_buffer.append([state[0], state[1], action])
        
        # Retrain periodically with more data
        if i > 0 and i % RETRAIN_INTERVAL == 0 and len(data_buffer) >= RETRAIN_INTERVAL:
            recent_data = torch.tensor(data_buffer[-RETRAIN_INTERVAL:], dtype=DTYPE)
            retrain_loss = train_network(network, recent_data, DEVICE, epochs=300)
            logger.info(f"Retrained network at step {i}. Loss: {retrain_loss:.6f}")
            

            plotter.update(iteration=i, 
                   imitation_loss=retrain_loss, 
                   reward=total_reward, 
                   angle_error=abs(angle_normalize(state[0])))


            # Save model after retraining
            if args.save_model:
                training_params = {
                    "dataset_size": len(data_buffer),
                    "error_metrics": {
                        "retrain_loss": retrain_loss
                    },
                    "hyperparameters": {
                        "hidden_units": H_UNITS,
                        "retrain_interval": RETRAIN_INTERVAL,
                        "learning_rate": 0.001,
                        "g": G,
                        "action_limits": [ACTION_LOW, ACTION_HIGH],
                        "goal_weights": GOAL_WEIGHTS.cpu().tolist(),
                        "control_penalty": CTRL_PENALTY  # Save the higher penalty
                    },
                    "training_info": {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "training_type": "online_retrain",
                        "step": i,
                        "run_timestamp": RUN_TIMESTAMP if args.save_model else "unknown"
                    }
                }
                save_model_weights(network, training_params)
        
        # Success detection
        angle_error = abs(angle_normalize(state[0]))
        if angle_error < 0.3 and abs(state[1]) < 1.5:
            success_count += 1
        else:
            success_count = 0
        
        # More detailed logging to track control saturation
        if i % 50 == 0:
            status = "MPC" if mpc_success else "Fallback"
            action_sat = "SAT!" if abs(action) > 1.9 else "OK"  # Tighter saturation check
            logger.info(f"Step {i}: θ={angle_normalize(state[0]):.3f}, "
                    f"θ̇={state[1]:.3f}, u={action:.3f}[{action_sat}], r={reward:.3f} [{status}]")
        
        if args.plot:
            monitor.update(
                theta=angle_normalize(state[0]),
                theta_dot=state[1],
                control=action,
                reward=reward,
                cu_reward=total_reward
            )
        
        if success_count > 150:  # More strict success criteria
            logger.info(f"SUCCESS! Pendulum balanced at step {i}")
            break
        
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
            u_init = None
    
    # Save final model
    if args.save_model:
        final_training_params = {
            "final_results": {
                "total_reward": float(total_reward),
                "final_angle_error": float(abs(angle_normalize(state[0]))),
                "iterations_completed": i + 1,
                "success": success_count > 100,
                "mpc_failures": mpc_failures
            },
            "hyperparameters": {
                "hidden_units": H_UNITS,
                "bootstrap_iterations": BOOTSTRAP_ITER,
                "retrain_interval": RETRAIN_INTERVAL,
                "learning_rate": 0.001,
                "g": G,
                "action_limits": [ACTION_LOW, ACTION_HIGH],
                "goal_weights": GOAL_WEIGHTS.cpu().tolist(),
                "control_penalty": CTRL_PENALTY  # Save the corrected penalty
            },
            "model_info": {
                "dataset_size": len(data_buffer),
                "device": str(DEVICE)
            },
            "run_info": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "run_timestamp": RUN_TIMESTAMP if args.save_model else "unknown",
                "success": success_count > 100
            }
        }
        save_model_weights(network, final_training_params)
        print(f"Final model weights saved to {RUN_DIR}")
    
    logger.info(f"Training completed after {i+1} steps")
    logger.info(f"Total reward: {total_reward:.2f}")
    logger.info(f"Final angle error: {abs(angle_normalize(state[0])):.3f} rad")
    logger.info(f"MPC failures during training: {mpc_failures}")
    
    if args.plot:
        if RUN_DIR:
            plots_dir = os.path.join(RUN_DIR, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            monitor.save_plots(directory=plots_dir, filename_prefix="training")
        else:
            monitor.save_plots(filename_prefix="training")
    

    logger.info("Creating training progress plots...")
    plotter.plot_training_progress()
    plotter.print_summary()

    print(f"Training plots saved to: {plotter.save_dir}")

    monitor.close()
    env.close()

if __name__ == "__main__":
    main()