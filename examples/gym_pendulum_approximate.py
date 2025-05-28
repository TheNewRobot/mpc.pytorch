"""
Optimized Learning-Based MPC Pendulum Controller
================================================

This script implements Model Predictive Control (MPC) for the pendulum swing-up
problem using a neural network to learn and approximate the system dynamics
from collected state-action data, with optimizations for better performance.
"""

import logging
import math
import time
import argparse
import warnings
import os
import json
from datetime import datetime

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*The reward returned by.*")
warnings.filterwarnings("ignore", message=".*The obs returned by the.*")

import gymnasium as gym
import numpy as np
import torch
from mpc import mpc
from monitor import Monitor  # Import the monitoring module

# Set up argument parser
parser = argparse.ArgumentParser(description='MPC Pendulum with Approximated Dynamics')
parser.add_argument('--render', action='store_true', help='Enable rendering')
parser.add_argument('--plot', action='store_true', help='Enable real-time plotting')
parser.add_argument('--save-model', action='store_true', help='Save model weights after training')
parser.add_argument('--base-dir', type=str, default='model_weights', help='Base directory for saving model weights')
args = parser.parse_args()

# Configure logging - reduce verbosity for better performance
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

# Create a unique run directory based on timestamp when the script starts
if args.save_model:
    RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    RUN_DIR = os.path.join(args.base_dir, RUN_TIMESTAMP)
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"Model weights for this run will be saved to: {RUN_DIR}")
else:
    RUN_DIR = None

# Pre-define angle normalization function
def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)

def save_model_weights(model, params):
    """
    Save model weights and training parameters.
    Overwrites existing files within the run directory.
    
    Args:
        model: PyTorch model to save
        params: Dictionary of hyperparameters and training info
    """
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

def load_model_weights(run_dir, model=None, nx=2, nu=1):
    """
    Load model weights and parameters from a specified run directory.
    
    Args:
        run_dir: Run directory containing saved model
        model: Optional model to load weights into (will create new if None)
        nx: State dimension (needed if model is None)
        nu: Action dimension (needed if model is None)
    
    Returns:
        model: Loaded model
        params: Dictionary of saved parameters
    """
    # Check if directory exists
    if not os.path.exists(run_dir):
        print(f"Model directory {run_dir} not found")
        return None, None
    
    # Load parameters
    params_path = os.path.join(run_dir, "training_params.json")
    if not os.path.exists(params_path):
        print(f"Parameters file not found at {params_path}")
        return None, None
        
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Load model weights
    weights_path = os.path.join(run_dir, "network_weights.pt")
    if not os.path.exists(weights_path):
        print(f"Weights file not found at {weights_path}")
        return None, None
    
    # Create model if not provided
    if model is None:
        h_units = params["hyperparameters"]["hidden_units"]
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
    model.load_state_dict(torch.load(weights_path))
    
    print(f"Model loaded from {run_dir}")
    return model, params

if __name__ == "__main__":
    # Constants - predefined for efficiency
    ENV_NAME = "Pendulum-v1"
    TIMESTEPS = 20
    N_BATCH = 1
    LQR_ITER = 100
    ACTION_LOW = -4.0
    ACTION_HIGH = 4.0
    DEVICE = "cpu" #TODO not being used currently
    DTYPE = torch.double
    
    # System constants
    G = 10.0
    M = 1.0
    L = 1.0
    DT = 0.05
    MAX_THDOT = 8  # max angular velocity
    GOAL_STATE = torch.tensor((0., 0.), dtype=DTYPE)
    GOAL_WEIGHTS = torch.tensor((10., 1.0), dtype=DTYPE)
    
    # Neural network hyperparameters
    H_UNITS = 64  # Increased for better expressivity
    TRAIN_EPOCH = 200
    BOOT_STRAP_ITER = 300
    CTRL_PENALTY = 0.1
    LEARNING_RATE = 0.001

    # Training
    RETRAIN_AFTER_ITER = 100
    RUN_ITER = 1000
    
    # System dimensions
    nx = 2  # state dimension
    nu = 1  # action dimension

    # Initialize neural network for dynamics approximation with a deeper architecture
    network = torch.nn.Sequential(
        torch.nn.Linear(nx + nu, H_UNITS),
        torch.nn.ReLU(),
        torch.nn.Linear(H_UNITS, H_UNITS),
        torch.nn.ReLU(),
        torch.nn.Linear(H_UNITS, H_UNITS),  # Added another layer
        torch.nn.ReLU(),
        torch.nn.Linear(H_UNITS, nx)
    ).double()
    
    # Initialize weights for faster convergence
    for m in network.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    # Learned dynamics model
    class PendulumDynamics(torch.nn.Module):
        def forward(self, state, perturbed_action):
            # Apply control limits
            u = torch.clamp(perturbed_action, ACTION_LOW, ACTION_HIGH)
            
            # Handle dimensions efficiently
            if len(state.shape) == 1:
                state = state.view(1, -1)
            if len(u.shape) == 1:
                u = u.view(1, -1)
            if u.shape[1] > 1:
                u = u[:, 0].view(-1, 1)
                
            # Concatenate inputs for network
            xu = torch.cat((state, u), dim=1)
            
            # Predict state residual and compute next state
            state_residual = network(xu)
            next_state = state + state_residual
            
            # Normalize angle
            next_state[:, 0] = angle_normalize(next_state[:, 0])
            return next_state

    # True dynamics model for comparison and validation
    def true_dynamics(state, action):
        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)
        
        u = torch.clamp(action, ACTION_LOW, ACTION_HIGH)
        
        # Vectorized computation
        newthdot = thdot + (-3 * G / (2 * L) * torch.sin(th + np.pi) + 3. / (M * L ** 2) * u) * DT
        newth = th + newthdot * DT
        newthdot = torch.clamp(newthdot, -MAX_THDOT, MAX_THDOT)
        
        return torch.cat((angle_normalize(newth), newthdot), dim=1)

    # Initialize dataset and dynamics
    dataset = None
    dynamics = PendulumDynamics()
    current_model_error = 0.0

    # Create validation set for model evaluation
    Nv = 1000
    # More efficient validation set generation
    statev = torch.zeros((Nv, 2), dtype=DTYPE)
    statev[:, 0] = (torch.rand(Nv, dtype=DTYPE) - 0.5) * 2 * math.pi
    statev[:, 1] = (torch.rand(Nv, dtype=DTYPE) - 0.5) * 16
    actionv = (torch.rand(Nv, 1, dtype=DTYPE) - 0.5) * (ACTION_HIGH - ACTION_LOW)

    # Angle difference calculation (optimized with vectorization)
    def angular_diff_batch(a, b):
        d = a - b
        d = torch.remainder(d + math.pi, 2 * math.pi) - math.pi
        return d

    # Optimized training function with proper dataset management
    def train(new_data):
        global dataset
        global current_model_error

        # Normalize angles
        new_data[:, 0] = angle_normalize(new_data[:, 0])
        
        # Convert to tensor if needed
        if not torch.is_tensor(new_data):
            new_data = torch.from_numpy(new_data).to(dtype=DTYPE)
            
        # Clamp actions to valid range
        new_data[:, -1] = torch.clamp(new_data[:, -1], ACTION_LOW, ACTION_HIGH)
        
        # Append data efficiently - grow dataset over time
        if dataset is None:
            dataset = new_data.clone()
        else:
            dataset = torch.cat((dataset, new_data), dim=0)
            
            # Limit dataset size to prevent memory issues
            if dataset.shape[0] > 5000:
                # Keep most recent data points
                dataset = dataset[-5000:]

        # Prepare training data
        XU = dataset.detach()
        
        # Compute state differences for training targets
        dtheta = angular_diff_batch(XU[1:, 0], XU[:-1, 0])
        dtheta_dt = XU[1:, 1] - XU[:-1, 1]
        Y = torch.cat((dtheta.view(-1, 1), dtheta_dt.view(-1, 1)), dim=1)
        XU = XU[:-1]  # Make same size as Y

        # Enable gradients for training
        for param in network.parameters():
            param.requires_grad = True

        # Use efficient optimization with proper learning rate schedule
        optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        
        # Track losses for early stopping
        prev_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        # Training loop with early stopping
        for epoch in range(TRAIN_EPOCH):
            optimizer.zero_grad()
            
            # Mini-batch training for larger datasets
            if XU.shape[0] > 1000:
                # Simple mini-batch implementation
                batch_size = 1000
                idx = torch.randperm(XU.shape[0])[:batch_size]
                X_batch = XU[idx]
                Y_batch = Y[idx]
                
                Yhat = network(X_batch)
                loss = torch.mean((Y_batch - Yhat).norm(2, dim=1) ** 2)
            else:
                # Use full batch for smaller datasets
                Yhat = network(XU)
                loss = torch.mean((Y - Yhat).norm(2, dim=1) ** 2)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Log progress less frequently
            if epoch % 10 == 0:
                logger.debug("ds %d epoch %d loss %f", dataset.shape[0], epoch, loss.item())
            
            # Learning rate adjustment
            if epoch % 20 == 0:
                scheduler.step()
                
            # Early stopping check
            if abs(prev_loss - loss.item()) < 1e-4:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                patience_counter = 0
            prev_loss = loss.item()

        # Freeze network for inference
        for param in network.parameters():
            param.requires_grad = False

        # Evaluate network against true dynamics
        with torch.no_grad():
            yt = true_dynamics(statev, actionv)
            yp = dynamics(statev, actionv)
            
            # Compute errors
            dtheta = angular_diff_batch(yp[:, 0], yt[:, 0])
            dtheta_dt = yp[:, 1] - yt[:, 1]
            E = torch.cat((dtheta.view(-1, 1), dtheta_dt.view(-1, 1)), dim=1).norm(dim=1)
            current_model_error = E.mean().item()
            logger.info("Error with true dynamics theta %f theta_dt %f norm %f", 
                    dtheta.abs().mean(), dtheta_dt.abs().mean(), E.mean())
        
        # Save model weights after successful training (if enabled)
        if args.save_model:
            training_params = {
                "dataset_size": dataset.shape[0],
                "error_metrics": {
                    "mean_norm_error": current_model_error,
                    "mean_theta_error": dtheta.abs().mean().item(),
                    "mean_thetadt_error": dtheta_dt.abs().mean().item()
                },
                "hyperparameters": {
                    "hidden_units": H_UNITS,
                    "train_epochs": TRAIN_EPOCH,
                    "learning_rate": LEARNING_RATE,
                    "ctrl_penalty": CTRL_PENALTY,
                    "g": G,
                    "m": M,
                    "l": L,
                    "dt": DT
                },
                "training_info": {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "training_type": "bootstrap" if i < BOOT_STRAP_ITER else "online",
                    "run_timestamp": RUN_TIMESTAMP
                }
            }
            
            # Save to the run directory, overwriting existing files
            save_model_weights(network, training_params)

    # MPC and environment setup
    u_init = None
    render_mode = "human" if args.render else None
    env_kwargs = {"g": G}
    
    # Create environment and monitor - use proper update frequency
    env = gym.make(ENV_NAME, render_mode=render_mode, disable_env_checker=True, **env_kwargs)
    monitor = Monitor(enabled=args.plot, update_freq=1)  # Increased frequency for real-time plotting
    
    # Initialize environment
    observation, info = env.reset()
    env.unwrapped.state = np.array([np.pi, 1], dtype=np.float32)

    observation = env.unwrapped._get_obs()

    # Bootstrap with random actions 
    if BOOT_STRAP_ITER:
        logger.info("Bootstrapping with random actions for %d steps", BOOT_STRAP_ITER)
        new_data = np.zeros((BOOT_STRAP_ITER, nx + nu))
        
        for i in range(BOOT_STRAP_ITER):
            pre_action_state = env.unwrapped.state.copy()
            action = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH, size=(1,))
            observation, reward, terminated, truncated, info = env.step(action)
            
            new_data[i, :nx] = pre_action_state
            new_data[i, nx:] = action

        # Train on bootstrap data
        train(new_data)
        logger.info("Bootstrapping finished")
    
    # Reset environment after bootstrapping
    observation, info = env.reset()
    env.unwrapped.state = np.array([np.pi, 1], dtype=np.float32)
    observation = env.unwrapped._get_obs()
    
    q = torch.cat((GOAL_WEIGHTS, CTRL_PENALTY * torch.ones(nu, dtype=DTYPE)))
    px = -torch.sqrt(GOAL_WEIGHTS) * GOAL_STATE
    p = torch.cat((px, torch.zeros(nu, dtype=DTYPE)))
    Q = torch.diag(q).repeat(TIMESTEPS, N_BATCH, 1, 1)
    p = p.repeat(TIMESTEPS, N_BATCH, 1)
    cost = mpc.QuadCost(Q, p)
    
    # Create MPC controller (once, outside the loop)
    mpc_controller = mpc.MPC(nx, nu, TIMESTEPS, 
                            u_lower=ACTION_LOW, u_upper=ACTION_HIGH, 
                            lqr_iter=LQR_ITER,
                            exit_unconverged=False, 
                            eps=1e-2,
                            n_batch=N_BATCH, 
                            backprop=False,
                            verbose=-1,
                            u_init=u_init,
                            grad_method=mpc.GradMethods.AUTO_DIFF)

    # Run MPC with data collection
    collected_dataset = torch.zeros((RETRAIN_AFTER_ITER, nx + nu), dtype=DTYPE)
    u_zeros = torch.zeros(1, N_BATCH, nu, dtype=DTYPE)  # Preallocate
    total_reward = 0
    computation_times = []
    
    for i in range(RUN_ITER):
        # Get and prepare state
        state = env.unwrapped.state.flatten().copy()
        state_tensor = torch.tensor(state, dtype=DTYPE).view(1, -1)
        
        # Render if enabled
        if render_mode == "human":
            env.render()
            
        # Measure computation time
        command_start = time.perf_counter()
        
        # Compute control action using MPC
        nominal_states, nominal_actions, nominal_objs = mpc_controller(
            state_tensor, cost, dynamics)
        action = nominal_actions[0]
        
        # Update warm start solution
        u_init = torch.cat((nominal_actions[1:], u_zeros), dim=0)
        
        # Record computation time
        elapsed = time.perf_counter() - command_start
        computation_times.append(elapsed)
        
        # Apply action to environment
        observation, reward, terminated, truncated, info = env.step(action.detach().numpy())
        done = terminated or truncated
        
        # Process reward
        if isinstance(reward, np.ndarray):
            reward = float(reward.item())
            
        total_reward += reward
        
        # Log less frequently
        if i % 5 == 0:
            print(f"Iteration {i}, Current angle: {angle_normalize(state[0]):.2f}")
            logger.debug("action taken: %.4f cost received: %.4f time taken: %.5fs", 
                       action.item(), -reward, elapsed)

        # Store data for retraining
        di = i % RETRAIN_AFTER_ITER
        if di == 0 and i > 0:
            # Retrain the model periodically
            train(collected_dataset)
            
        # Store current state-action pair
        collected_dataset[di, :nx] = state_tensor
        collected_dataset[di, nx:] = action
        
        # Update monitoring
        if args.plot and i % 5 == 0:  # Update every 5 iterations to avoid overwhelming the plot
            monitor.update(
                theta=angle_normalize(state[0]),
                theta_dot=state[1],
                control=action.item(),
                reward=reward,
                model_error=current_model_error
            )
        
        # Break if problem is solved (pendulum is upright and stable)
        if abs(angle_normalize(state[0])) < 0.1 and abs(state[1]) < 0.5:
            success_counter = getattr(env, 'success_counter', 0) + 1
            setattr(env, 'success_counter', success_counter)
            
            if success_counter > 50:  # Stable for 50 steps
                print("Success! Pendulum balanced.")
                break
        else:
            setattr(env, 'success_counter', 0)

    # Print summary
    logger.info("Total reward: %f", total_reward)
    logger.info("Average computation time: %f seconds", np.mean(computation_times))
    
    # Save final model if enabled
    if args.save_model:
        final_training_params = {
            "final_results": {
                "total_reward": total_reward,
                "avg_computation_time": float(np.mean(computation_times)),
                "iterations_completed": i + 1
            },
            "hyperparameters": {
                "hidden_units": H_UNITS,
                "train_epochs": TRAIN_EPOCH,
                "learning_rate": LEARNING_RATE,
                "ctrl_penalty": CTRL_PENALTY,
                "g": G,
                "m": M,
                "l": L,
                "dt": DT
            },
            "model_info": {
                "dataset_size": dataset.shape[0] if dataset is not None else 0,
                "bootstrap_iterations": BOOT_STRAP_ITER,
                "retrain_interval": RETRAIN_AFTER_ITER
            },
            "run_info": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "run_timestamp": RUN_TIMESTAMP,
                "success": success_counter > 50 if 'success_counter' in dir() else False
            }
        }
        
        # Save the final model (overwriting previous saves in this run)
        save_model_weights(network, final_training_params)
        print(f"Final model weights saved to {RUN_DIR}")
    
    # Save plots if enabled
    if args.plot:
        # If saving models, also save the plots to the same run directory
        if args.save_model and RUN_DIR:
            plots_dir = os.path.join(RUN_DIR, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            monitor.save_plots(directory=plots_dir, 
                               filename_prefix="pendulum_learned_dynamics")
            print(f"Plots saved to {plots_dir}")
        else:
            monitor.save_plots(directory="pendulum_learning_plots", 
                               filename_prefix="pendulum_learned_dynamics")
            print(f"Plots saved to 'pendulum_learning_plots' directory")
    
    # Clean up resources
    monitor.close()
    env.close()