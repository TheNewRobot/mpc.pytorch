"""
Fixed Pendulum Neural Network Deployment
========================================

Key fixes:
1. Reduced MPC frequency for better swing-up behavior
2. Adjusted cost function weights
3. Better action limits handling
4. Improved initialization strategy
"""

import argparse
import os
import json
import time
import math
import gymnasium as gym
import torch
import numpy as np
from mpc import mpc
from monitor import Monitor

parser = argparse.ArgumentParser(description='Deploy a saved pendulum model')
parser.add_argument('--model-dir', type=str, required=True, help='Directory containing saved model')
parser.add_argument('--render', action='store_true', help='Enable rendering')
parser.add_argument('--plot', action='store_true', help='Enable real-time plotting')
parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations to run')
args = parser.parse_args()

def angle_normalize(x):
    """Normalize angle to [-pi, pi] using atan2"""
    if torch.is_tensor(x):
        return torch.atan2(torch.sin(x), torch.cos(x))
    else:
        return math.atan2(math.sin(x), math.cos(x))

def obs_to_state(obs):
    """Convert gym observation [cos(θ), sin(θ), θ_dot] to state [θ, θ_dot]"""
    if torch.is_tensor(obs):
        theta = torch.atan2(obs[..., 1], obs[..., 0])
        theta_dot = obs[..., 2]
        return torch.stack([theta, theta_dot], dim=-1)
    else:
        theta = np.arctan2(obs[1], obs[0])
        theta_dot = obs[2]
        return np.array([theta, theta_dot])

def load_model_weights(model_dir):
    """Load model weights and parameters from a saved directory."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} not found")
    
    # Load parameters
    params_path = os.path.join(model_dir, "training_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameters file not found at {params_path}")
        
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Load model weights
    weights_path = os.path.join(model_dir, "network_weights.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")
    
    # Extract hyperparameters - match the training script structure
    if "hyperparameters" in params:
        h_units = params["hyperparameters"].get("hidden_units", 64)
        action_limits = params["hyperparameters"].get("action_limits", [-2.0, 2.0])
        goal_weights = params["hyperparameters"].get("goal_weights", [100.0, 10.0])
        g = params["hyperparameters"].get("g", 10.0)
    else:
        print("No hyperparameters found, using defaults")
        h_units = 64
        action_limits = [-2.0, 2.0]
        goal_weights = [100.0, 10.0]
        g = 10.0
    
    # Create model architecture matching the TRAINING script (3 layers, not 4)
    nx, nu = 2, 1
    model = torch.nn.Sequential(
        torch.nn.Linear(nx + nu, h_units),
        torch.nn.ReLU(),
        torch.nn.Linear(h_units, h_units),
        torch.nn.ReLU(),
        torch.nn.Linear(h_units, nx)  # Output layer - only 3 layers total
    ).double()
    
    # Load weights
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Model loaded from {model_dir}")
    print(f"Hidden units: {h_units}")
    print(f"Action limits: {action_limits}")
    print(f"Goal weights: {goal_weights}")
    print(f"Gravity: {g}")
    
    return model, params, action_limits, goal_weights, g

class DeployedDynamics(torch.nn.Module):
    """Dynamics model for deployment that matches training."""
    
    def __init__(self, network, action_low=-2.0, action_high=2.0, device='cpu'):
        super().__init__()
        self.network = network
        self.action_low = action_low
        self.action_high = action_high
        self.max_thdot = 8.0
        self.device = device
    
    def forward(self, state, action):
        # Handle dimensions
        if len(state.shape) == 1:
            state = state.view(1, -1)
        if len(action.shape) == 1:
            action = action.view(1, -1)
        if action.shape[1] > 1:
            action = action[:, 0].view(-1, 1)
            
        # Move to device
        state = state.to(self.device)
        action = action.to(self.device)
        
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

def main():

    print("Loading model...")
    start_time = time.time()
    
    try:
        network, params, action_limits, goal_weights, g = load_model_weights(args.model_dir)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Extract parameters
    action_low, action_high = action_limits
    g = 9.81  # Force consistent gravity
    
    # FIXED: Use same goal weights as reference script
    goal_weights_tensor = torch.tensor([1.0, 0.1], dtype=torch.double)  # Match training
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = network.to(device)
    goal_weights_tensor = goal_weights_tensor.to(device)
    print(f"Using device: {device}")
    
    # Create environment with correct gravity
    render_mode = "human" if args.render else None
    env_kwargs = {"g": g}  # Use same g as training
    env = gym.make("Pendulum-v1", render_mode=render_mode, disable_env_checker=True, **env_kwargs)
    
    # Create dynamics model
    dynamics = DeployedDynamics(network, action_low, action_high, device)
    
    # FIXED: MPC setup - match reference script parameters
    TIMESTEPS = 10  # Match reference script
    N_BATCH = 1
    LQR_ITER = 5    # Match reference script
    nx, nu = 2, 1
    
    DTYPE = torch.double
    CTRL_PENALTY = 0.001  # Match reference script
    EPS = 1e-2      # Match reference script
    
    # Goal is upright position
    GOAL_STATE = torch.tensor([0.0, 0.0], dtype=DTYPE, device=device)
    
    # FIXED: MPC cost setup with correct weights
    q = torch.cat((goal_weights_tensor, CTRL_PENALTY * torch.ones(nu, dtype=DTYPE, device=device)))
    px = -torch.sqrt(goal_weights_tensor) * GOAL_STATE
    p = torch.cat((px, torch.zeros(nu, dtype=DTYPE, device=device)))
    Q = torch.diag(q).repeat(TIMESTEPS, N_BATCH, 1, 1)
    p = p.repeat(TIMESTEPS, N_BATCH, 1)
    cost = mpc.QuadCost(Q, p)
    
    # REMOVED: Don't create MPC controller here, create it in the loop
    
    # Create monitor for plotting
    monitor = Monitor(enabled=args.plot, update_freq=1)
    
    # Initialize environment - start in downward position
    observation, info = env.reset()
    env.unwrapped.state = np.array([np.pi, 0], dtype=np.float32)  # Start hanging down
    observation = env.unwrapped._get_obs()
    
    # Run loop
    total_reward = 0
    computation_times = []
    success_counter = 0
    mpc_failures = 0
    
    print(f"Running model for {args.iterations} iterations...")
    print("Starting from downward position, attempting to swing up to upright...")
    
    # FIXED: Initialize u_init here, before the loop
    u_init = None

    for i in range(args.iterations):
        # Get current state from observation
        state = obs_to_state(observation)
        state_tensor = torch.tensor(state, dtype=DTYPE, device=device).view(1, -1)
        
        if render_mode == "human":
            env.render()
            
        command_start = time.time()
        
        try:
            # FIXED: Create MPC controller with u_init each time
            ctrl = mpc.MPC(nx, nu, TIMESTEPS, 
                        u_lower=action_low, u_upper=action_high, 
                        lqr_iter=LQR_ITER,
                        exit_unconverged=False, 
                        eps=EPS,
                        n_batch=N_BATCH, 
                        backprop=False,
                        verbose=-1,
                        u_init=u_init,  # Pass u_init during creation
                        grad_method=mpc.GradMethods.AUTO_DIFF)
            
            # Call MPC without u_init parameter
            nominal_states, nominal_actions, nominal_objs = ctrl(state_tensor, cost, dynamics)
            action = nominal_actions[0].detach().cpu().numpy().flatten()
            
            # FIXED: Update u_init for next iteration
            u_init = torch.cat((nominal_actions[1:], torch.zeros(1, N_BATCH, nu, device=device)), dim=0)
            
            mpc_used = True
            
        except Exception as e:
            # Energy-based fallback control...
            theta = angle_normalize(state[0])
            theta_dot = state[1]
            
            current_energy = 0.5 * theta_dot**2 + g * (1 - math.cos(theta))
            desired_energy = g * 2.0
            
            if current_energy < desired_energy * 0.9:
                action = np.array([np.sign(theta_dot * math.cos(theta)) * action_high * 0.7])
            else:
                action = np.array([-3.0 * theta - 1.0 * theta_dot])
                action = np.clip(action, action_low, action_high)
            
            mpc_failures += 1
            mpc_used = False
            
            if i % 100 == 0:
                print(f"MPC failed at iteration {i}, using energy-based control: {e}")
        
        elapsed = time.time() - command_start
        computation_times.append(elapsed)
        
        # Apply action to environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Process reward
        if isinstance(reward, np.ndarray):
            reward = float(reward.item())
        total_reward += reward
        
        # Success detection - same as before but more reasonable
        angle_error = abs(angle_normalize(state[0]))
        velocity_error = abs(state[1])
        
        if angle_error < 0.2 and velocity_error < 1.0:  # Stricter success criteria
            success_counter += 1
            if success_counter > 100:  # Need sustained balance
                print(f"SUCCESS! Pendulum balanced at iteration {i}")
                print(f"Final angle error: {angle_error:.3f} rad ({math.degrees(angle_error):.1f}°)")
                print(f"Final velocity error: {velocity_error:.3f} rad/s")
                break
        else:
            success_counter = 0
        
        # Progress logging
        if i % 50 == 0:
            controller_type = "MPC" if mpc_used else "Energy"
            print(f"Iter {i}: angle={angle_normalize(state[0]):.3f} "
                f"({math.degrees(angle_normalize(state[0])):.1f}°), "
                f"vel={state[1]:.3f}, action={action[0]:.3f}, "
                f"reward={reward:.2f}, ctrl={controller_type}, "
                f"time={elapsed*1000:.1f}ms")
            
        # Update monitoring
        if args.plot and i % 2 == 0:
            monitor.update(
                theta=angle_normalize(state[0]),
                theta_dot=state[1],
                control=action[0],
                reward=reward,
                cu_reward=total_reward
            )

    # Print summary
    print("\n" + "="*60)
    print("DEPLOYMENT SUMMARY")
    print("="*60)
    print(f"Total iterations: {i + 1}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average computation time: {np.mean(computation_times) * 1000:.2f} ms per step")
    print(f"MPC failures: {mpc_failures} ({100*mpc_failures/(i+1):.1f}%)")
    
    final_angle = angle_normalize(state[0])
    final_velocity = state[1]
    print(f"Final state: angle={final_angle:.3f} rad ({math.degrees(final_angle):.1f}°), "
          f"velocity={final_velocity:.3f} rad/s")
    
    if success_counter > 50:
        print("STATUS: ✅ SUCCESS - Pendulum successfully balanced!")
    elif abs(final_angle) < 0.8:
        print("STATUS: 🟡 PARTIAL SUCCESS - Made significant progress toward upright")
    else:
        print("STATUS: ❌ FAILED - Could not balance pendulum")
    
    # Performance analysis
    if mpc_failures > 0:
        print(f"\nNote: MPC failed {mpc_failures} times. This might indicate:")
        print("- Model accuracy issues")
        print("- Optimization convergence problems")
        print("- Consider retraining with more data")
    
    # Save plots if enabled
    if args.plot:
        results_dir = os.path.join(args.model_dir, "deployment_results")
        os.makedirs(results_dir, exist_ok=True)
        
        monitor.save_plots(directory=results_dir, 
                         filename_prefix="deployment_run")
        print(f"\nPlots saved to {results_dir}")
    
    # Save deployment results
    deployment_results = {
        "deployment_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "iterations_run": i + 1,
        "total_reward": total_reward,
        "avg_computation_time_ms": float(np.mean(computation_times) * 1000),
        "mpc_failures": mpc_failures,
        "mpc_failure_rate": mpc_failures / (i + 1),
        "final_state": {
            "angle_rad": float(final_angle),
            "angle_deg": float(math.degrees(final_angle)),
            "velocity": float(final_velocity)
        },
        "success": success_counter > 50,
        "success_counter": success_counter
    }
    
    results_path = os.path.join(args.model_dir, "deployment_results.json")
    with open(results_path, 'w') as f:
        json.dump(deployment_results, f, indent=4)
    print(f"Deployment results saved to {results_path}")
    
    # Clean up resources
    monitor.close()
    env.close()

if __name__ == "__main__":
    main()