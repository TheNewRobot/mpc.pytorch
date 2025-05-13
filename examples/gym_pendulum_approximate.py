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
args = parser.parse_args()

# Configure logging - reduce verbosity for better performance
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

# Pre-define angle normalization function
def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)

if __name__ == "__main__":
    # Constants - predefined for efficiency
    ENV_NAME = "Pendulum-v1"
    TIMESTEPS = 10 # This was initially 10
    N_BATCH = 1
    LQR_ITER = 50
    ACTION_LOW = -4.0
    ACTION_HIGH = 4.0
    DEVICE = "cpu"
    DTYPE = torch.double
    
    # System constants
    G = 10.0
    M = 1.0
    L = 1.0
    DT = 0.05
    MAX_THDOT = 8  # max angular velocity
    
    # Neural network hyperparameters
    H_UNITS = 32 # This was initially 16
    TRAIN_EPOCH = 200
    BOOT_STRAP_ITER = 300 # Exploration During Bootstrapping
    CTRL_PENALTY = 0.5
    LEARNING_RATE = 0.001
    
    # System dimensions
    nx = 2  # state dimension
    nu = 1  # action dimension

    # Initialize neural network for dynamics approximation
    # Create a more efficient network architecture with proper initialization
    network = torch.nn.Sequential(
        torch.nn.Linear(nx + nu, H_UNITS),
        torch.nn.ReLU(),
        torch.nn.Linear(H_UNITS, H_UNITS),
        torch.nn.ReLU(),
        torch.nn.Linear(H_UNITS, nx)
    ).double()
    
    # Initialize weights with a better method for faster convergence
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

    # True dynamics model for comparison
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
    
    # Create validation set for model evaluation
    Nv = 1000
    # Use a more efficient approach for validation set generation
    statev = torch.zeros((Nv, 2), dtype=DTYPE)
    statev[:, 0] = (torch.rand(Nv, dtype=DTYPE) - 0.5) * 2 * math.pi
    statev[:, 1] = (torch.rand(Nv, dtype=DTYPE) - 0.5) * 16
    actionv = (torch.rand(Nv, 1, dtype=DTYPE) - 0.5) * (ACTION_HIGH - ACTION_LOW)

    # Angle difference calculation (optimized with vectorization)
    def angular_diff_batch(a, b):
        d = a - b
        mask_high = d > math.pi
        mask_low = d < -math.pi
        d[mask_high] -= 2 * math.pi
        d[mask_low] += 2 * math.pi
        return d

    # Optimized training function
    def train(new_data):
        global dataset
        
        # Normalize angles
        new_data[:, 0] = angle_normalize(new_data[:, 0])
        
        # Convert to tensor if needed
        if not torch.is_tensor(new_data):
            new_data = torch.from_numpy(new_data)
            
        # Clamp actions to valid range
        new_data[:, -1] = torch.clamp(new_data[:, -1], ACTION_LOW, ACTION_HIGH)
        
        # Append data efficiently
        if dataset is None:
            dataset = new_data.clone()
        else:
            dataset = torch.cat((dataset, new_data), dim=0)

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

        # Use more efficient optimization with learning rate schedule
        optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        
        # Track losses for early stopping
        prev_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        # Training loop with early stopping
        for epoch in range(TRAIN_EPOCH):
            optimizer.zero_grad()
            
            # Compute predictions and loss
            Yhat = network(XU)
            loss = torch.mean((Y - Yhat).norm(2, dim=1) ** 2)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Log progress less frequently
            if epoch % 10 == 0:
                logger.debug("ds %d epoch %d loss %f", dataset.shape[0], epoch, loss.item())
            
            # Learning rate adjustment
            # if epoch % 20 == 0:
                #scheduler.step()
                
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
            
            logger.info("Error with true dynamics theta %f theta_dt %f norm %f", 
                      dtheta.abs().mean(), dtheta_dt.abs().mean(), E.mean())

    # MPC and environment setup
    u_init = None
    retrain_after_iter = 100
    run_iter = 1000
    render_mode = "human" if args.render else None
    env_kwargs = {"g": G}
    
    # Create environment and monitor
    env = gym.make(ENV_NAME, render_mode=render_mode, disable_env_checker=True, **env_kwargs)
    monitor = Monitor(enabled=args.plot, update_freq=10)  # Less frequent updates for efficiency
    
    # Initialize environment
    observation, info = env.reset()
    env.unwrapped.state = np.array([np.pi, 1], dtype=np.float32)

    goal_state = torch.tensor((0., 0.), dtype=DTYPE)
    goal_weights = torch.tensor((3., 0.5), dtype=DTYPE)

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
    
    q = torch.cat((goal_weights, CTRL_PENALTY * torch.ones(nu, dtype=DTYPE)))
    px = -torch.sqrt(goal_weights) * goal_state
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
    collected_dataset = torch.zeros((retrain_after_iter, nx + nu), dtype=DTYPE)
    u_zeros = torch.zeros(1, N_BATCH, nu, dtype=DTYPE)  # Preallocate
    total_reward = 0
    computation_times = []
    
    for i in range(run_iter):
        # Get and prepare state
        state = env.unwrapped.state.copy()
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
        if i % 10 == 0:
            print(f"Iteration {i}, Current angle: {angle_normalize(state[0])}, Distance to upright: {angle_normalize(abs(state[0]))}")
            logger.debug("action taken: %.4f cost received: %.4f time taken: %.5fs", 
                       action.item(), -reward, elapsed)

        # Store data for retraining
        di = i % retrain_after_iter
        if di == 0 and i > 0:
            # Retrain the model periodically
            train(collected_dataset)
            
        # Store current state-action pair
        collected_dataset[di, :nx] = state_tensor
        collected_dataset[di, nx:] = action
        
        # Update monitoring (less frequently)
        if args.plot and i % 5 == 0:
            print("Updating the plot")
            monitor.update(
                theta=state[0],
                theta_dot=state[1],
                control=action.item(),
                reward=reward,
                cumulative_reward=total_reward,
                model_error=torch.tensor(computation_times).mean().item()
            )
        

    # Print summary
    logger.info("Total reward: %f", total_reward)
    logger.info("Average computation time: %f seconds", np.mean(computation_times))
    
    # Save plots if enabled
    if args.plot:
        monitor.save_plots(directory="pendulum_learning_plots", 
                          filename_prefix="pendulum_learned_dynamics")
        print(f"Plots saved to 'pendulum_learning_plots' directory")
    
    # Clean up resources
    monitor.close()
    env.close()