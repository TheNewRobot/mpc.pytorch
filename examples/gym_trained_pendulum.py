"""
Fast Pendulum Neural Network Runner
==================================

This script loads a trained neural network model and runs it directly in the pendulum environment.
It performs simple inference without MPC, which is much faster.

Usage:
    python run_model_fast.py --model-dir model_weights/20250513-120000 --render
"""

import argparse
import os
import json
import time
import math
import gymnasium as gym
import torch
import numpy as np
from monitor import Monitor

# Set up argument parser
parser = argparse.ArgumentParser(description='Run a saved pendulum model (fast)')
parser.add_argument('--model-dir', type=str, required=True, help='Directory containing saved model')
parser.add_argument('--render', action='store_true', help='Enable rendering')
parser.add_argument('--plot', action='store_true', help='Enable real-time plotting')
parser.add_argument('--iterations', type=int, default=500, help='Number of iterations to run')
args = parser.parse_args()

# Pre-define angle normalization function
def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)

def load_model_weights(model_dir):
    """Load model weights and parameters from a saved directory."""
    # Check if directory exists
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
    
    # Extract hyperparameters
    if "hyperparameters" in params:
        h_units = params["hyperparameters"]["hidden_units"]
    else:
        print("No hyperparameters found, using default of 64 hidden units")
        h_units = 64
    
    # Create model 
    nx, nu = 2, 1  # State and action dimensions
    model = torch.nn.Sequential(
        torch.nn.Linear(nx + nu, h_units),
        torch.nn.ReLU(),
        torch.nn.Linear(h_units, h_units),
        torch.nn.ReLU(),
        torch.nn.Linear(h_units, h_units),
        torch.nn.ReLU(),
        torch.nn.Linear(h_units, nx)
    ).double()
    
    # Load weights
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded from {model_dir}")
    return model, params

class DirectController:
    """Simple controller that uses the neural network directly without MPC."""
    
    def __init__(self, network, action_low=-4.0, action_high=4.0):
        self.network = network
        self.action_low = action_low
        self.action_high = action_high
    
    def get_action(self, state):
        """
        Calculate action directly using the network.
        For a simple policy, we'll try multiple actions and pick the best one.
        """
        with torch.no_grad():  # No need for gradients in inference
            state_tensor = torch.tensor(state, dtype=torch.double).view(1, -1)
            
            # We'll try a few candidate actions and pick the best one
            num_candidates = 21
            actions = torch.linspace(self.action_low, self.action_high, num_candidates)
            actions = actions.view(-1, 1).double()
            
            # Clone state for each action
            states = state_tensor.repeat(num_candidates, 1)
            
            # Predict next states
            inputs = torch.cat((states, actions), dim=1)
            next_state_deltas = self.network(inputs)
            next_states = states + next_state_deltas
            
            # Compute costs: penalize distance from upright (0,0) and control effort
            angle_costs = (next_states[:, 0] ** 2) * 10.0  # Higher weight on angle
            velocity_costs = (next_states[:, 1] ** 2) * 1.0
            control_costs = (actions[:, 0] ** 2) * 0.1
            
            total_costs = angle_costs + velocity_costs + control_costs
            
            # Select action with lowest cost
            best_idx = torch.argmin(total_costs)
            best_action = actions[best_idx].item()
            
            return np.array([best_action])

def main():
    print("Loading model...")
    start_time = time.time()
    
    # Load the model and parameters
    try:
        network, params = load_model_weights(args.model_dir)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Extract parameters
    if "hyperparameters" in params:
        h_params = params["hyperparameters"]
        G = h_params.get("g", 10.0)
    else:
        G = 10.0
    
    # Create environment
    render_mode = "human" if args.render else None
    env_kwargs = {"g": G}
    env = gym.make("Pendulum-v1", render_mode=render_mode, disable_env_checker=True, **env_kwargs)
    
    # Create controller using the neural network
    controller = DirectController(network)
    
    # Create monitor for plotting
    monitor = Monitor(enabled=args.plot, update_freq=1)
    
    # Initialize environment - start in downward position
    observation, info = env.reset()
    env.unwrapped.state = np.array([np.pi, 0], dtype=np.float32)  # Start pendulum downward with no velocity
    observation = env.unwrapped._get_obs()
    
    # Run loop
    total_reward = 0
    computation_times = []
    
    print(f"Running model for {args.iterations} iterations...")
    for i in range(args.iterations):
        # Get current state
        state = env.unwrapped.state.flatten().copy()
        
        # Render if enabled
        if render_mode == "human":
            env.render()
            
        # Measure computation time
        command_start = time.time()
        
        # Get action from controller
        action = controller.get_action(state)
        
        # Record computation time
        elapsed = time.time() - command_start
        computation_times.append(elapsed)
        
        # Apply action to environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Process reward
        if isinstance(reward, np.ndarray):
            reward = float(reward.item())
            
        total_reward += reward
        
        # Log progress
        if i % 10 == 0:
            print(f"Iteration {i}, Angle: {angle_normalize(state[0]):.2f}, Action: {action[0]:.2f}, Reward: {reward:.2f}")
            
        # Update monitoring
        if args.plot:
            monitor.update(
                theta=angle_normalize(state[0]),
                theta_dot=state[1],
                control=action[0],
                reward=reward,
                cu_reward=total_reward
            )
        
        # Break if pendulum is balanced
        if abs(angle_normalize(state[0])) < 0.1 and abs(state[1]) < 0.5:
            success_counter = getattr(env, 'success_counter', 0) + 1
            setattr(env, 'success_counter', success_counter)
            
            if success_counter > 50:  # Stable for 50 steps
                print(f"Success! Pendulum balanced at iteration {i}.")
                break
        else:
            setattr(env, 'success_counter', 0)

    # Print summary
    print("\nRun Summary:")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average computation time: {np.mean(computation_times) * 1000:.2f} milliseconds per step")
    
    if 'success_counter' in dir() and success_counter > 50:
        print("Status: SUCCESS - Pendulum balanced")
    else:
        print("Status: FAILED - Could not balance pendulum")
    
    # Save plots if enabled
    if args.plot:
        # Create a results directory within the model directory
        results_dir = os.path.join(args.model_dir, "run_results")
        os.makedirs(results_dir, exist_ok=True)
        
        monitor.save_plots(directory=results_dir, 
                         filename_prefix="model_run_fast")
        print(f"Plots saved to {results_dir}")
    
    # Clean up resources
    monitor.close()
    env.close()

if __name__ == "__main__":
    main()