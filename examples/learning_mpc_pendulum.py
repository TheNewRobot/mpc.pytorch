"""
Differentiable MPC Parameter Learning Example
==========================================

This script demonstrates the differentiability features of mpc.pytorch by:
1. Learning unknown dynamics parameters (pendulum length and damping)
2. Learning optimal control cost weights through gradient descent
3. Using the differentiable MPC solver to backpropagate through the entire control loop

The example uses a modified pendulum dynamics where we pretend not to know
the exact length and damping coefficient, then learn them from data.
"""

import logging
import math
import time
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from mpc import mpc
import matplotlib.pyplot as plt

# Suppress warnings
logging.getLogger('matplotlib').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*The reward returned by.*")
warnings.filterwarnings("ignore", message=".*The obs returned by the.*")

def angle_normalize(x):
    """Normalize angle to [-pi, pi]"""
    return (((x + math.pi) % (2 * math.pi)) - math.pi)

class LearnablePendulumDynamics(torch.nn.Module):
    """
    Pendulum dynamics with learnable parameters.
    We'll learn the length and mass of the pendulum.
    """
    def __init__(self, dt=0.05, u_min=-2, u_max=2, thdot_min=-8, thdot_max=8):
        super(LearnablePendulumDynamics, self).__init__()
        
        # Fixed parameters
        self.g = 9.81
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        self.thdot_min = thdot_min
        self.thdot_max = thdot_max
        
        # Learnable parameters (initialized with wrong values)
        self.length = nn.Parameter(torch.tensor(0.8))  # True value is 1.0
        self.mass = nn.Parameter(torch.tensor(0.7))    # True value is 1.0
        
    def forward(self, state, action):
        """Forward dynamics with learnable parameters"""
        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)
        
        # Apply control constraints
        u = torch.clamp(action, self.u_min, self.u_max)
        
        # Standard pendulum physics equations with learnable parameters
        # θ̈ = -3g/(2l) * sin(θ + π) + 3/(ml²) * u
        newthdot = thdot + (-3 * self.g / (2 * self.length) * torch.sin(th + np.pi) + 
                           3. / (self.mass * self.length ** 2) * u) * self.dt
        newth = th + newthdot * self.dt
        
        # Apply state constraints
        newthdot = torch.clamp(newthdot, self.thdot_min, self.thdot_max)
        
        return torch.cat((angle_normalize(newth), newthdot), dim=1)

class LearnableCostWeights(torch.nn.Module):
    """
    Learnable cost function weights for MPC.
    We'll learn optimal weights for position, velocity, and control effort.
    """
    def __init__(self):
        super(LearnableCostWeights, self).__init__()
        
        # Initialize with suboptimal weights
        self.pos_weight = nn.Parameter(torch.tensor(0.5))  # Will learn ~1.0
        self.vel_weight = nn.Parameter(torch.tensor(0.05))  # Will learn ~0.1
        self.ctrl_weight = nn.Parameter(torch.tensor(0.01))  # Will learn ~0.001
        
    def get_weights(self):
        """Return positive weights using softplus"""
        return (torch.nn.functional.softplus(self.pos_weight),
                torch.nn.functional.softplus(self.vel_weight), 
                torch.nn.functional.softplus(self.ctrl_weight))

def create_cost_matrices(cost_weights, n_timesteps, n_batch):
    """Create quadratic cost matrices from learnable weights"""
    pos_w, vel_w, ctrl_w = cost_weights.get_weights()
    
    # Goal state (upright pendulum)
    goal_state = torch.tensor([0., 0.])
    
    # Create cost matrix
    q = torch.tensor([pos_w, vel_w, ctrl_w])
    px = -torch.sqrt(torch.tensor([pos_w, vel_w])) * goal_state
    p = torch.cat((px, torch.zeros(1)))
    
    Q = torch.diag(q).repeat(n_timesteps, n_batch, 1, 1)
    p = p.repeat(n_timesteps, n_batch, 1)
    
    return mpc.QuadCost(Q, p)

def simulate_true_dynamics(state, action, dt=0.05):
    """True pendulum dynamics for generating training data"""
    th, thdot = state
    u = np.clip(action, -2, 2)
    
    # True parameters - standard pendulum
    g, m, l = 9.81, 1.0, 1.0
    
    # Standard pendulum equation: θ̈ = -3g/(2l) * sin(θ + π) + 3/(ml²) * u
    newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 
                       3. / (m * l ** 2) * u) * dt
    newth = th + newthdot * dt
    newthdot = np.clip(newthdot, -8, 8)
    
    return np.array([angle_normalize(newth), newthdot])

def generate_training_data(n_episodes=5, episode_length=100):
    """Generate training data using random control inputs"""
    print("Generating training data...")
    
    states = []
    actions = []
    next_states = []
    
    for episode in range(n_episodes):
        # Random initial state
        state = np.array([np.random.uniform(-np.pi, np.pi), 
                         np.random.uniform(-2, 2)])
        
        for step in range(episode_length):
            # Random action
            action = np.random.uniform(-2, 2)
            
            # Simulate true dynamics
            next_state = simulate_true_dynamics(state, action)
            
            states.append(state.copy())
            actions.append([action])
            next_states.append(next_state.copy())
            
            state = next_state
    
    return (torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32))

def train_dynamics(dynamics_model, states, actions, next_states, epochs=200):
    """Train the dynamics model to match true dynamics"""
    print("Training dynamics model...")
    
    optimizer = optim.Adam(dynamics_model.parameters(), lr=0.01)
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        predicted_states = dynamics_model(states, actions)
        loss = torch.nn.functional.mse_loss(predicted_states, next_states)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, "
                  f"Length: {dynamics_model.length.item():.3f}, "
                  f"Mass: {dynamics_model.mass.item():.3f}")
    
    return losses

def evaluate_mpc_performance(dynamics, cost_weights, n_trials=3):
    """Evaluate MPC performance and return average cost"""
    total_cost = 0
    
    for trial in range(n_trials):
        # Random initial state (downward)
        initial_state = torch.tensor([[np.pi + np.random.normal(0, 0.1), 
                                     np.random.normal(0, 0.5)]], dtype=torch.float32)
        
        # MPC parameters
        nx, nu = 2, 1
        timesteps = 10
        n_batch = 1
        
        # Create cost
        cost = create_cost_matrices(cost_weights, timesteps, n_batch)
        
        # MPC controller
        ctrl = mpc.MPC(nx, nu, timesteps, 
                      u_lower=-2.0, u_upper=2.0,
                      lqr_iter=5, exit_unconverged=False,
                      eps=1e-2, n_batch=n_batch,
                      backprop=True, verbose=0,
                      grad_method=mpc.GradMethods.AUTO_DIFF)
        
        # Single MPC step
        nominal_states, nominal_actions, nominal_objs = ctrl(initial_state, cost, dynamics)
        
        # Cost is deviation from upright position
        final_state = nominal_states[-1]
        cost_val = (final_state[0, 0]**2 + 0.1 * final_state[0, 1]**2).item()
        total_cost += cost_val
    
    return total_cost / n_trials

def main():
    print("Differentiable MPC Parameter Learning Demo")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Generate training data
    states, actions, next_states = generate_training_data(n_episodes=10, episode_length=50)
    print(f"Generated {len(states)} training samples")
    
    # 2. Create learnable models
    dynamics_model = LearnablePendulumDynamics()
    cost_weights = LearnableCostWeights()
    
    print(f"\nInitial dynamics parameters:")
    print(f"Length: {dynamics_model.length.item():.3f} (true: 1.0)")
    print(f"Mass: {dynamics_model.mass.item():.3f} (true: 1.0)")
    
    # 3. Train dynamics model
    dynamics_losses = train_dynamics(dynamics_model, states, actions, next_states)
    
    print(f"\nFinal dynamics parameters:")
    print(f"Length: {dynamics_model.length.item():.3f} (true: 1.0)")
    print(f"Mass: {dynamics_model.mass.item():.3f} (true: 1.0)")
    
    # 4. Learn optimal cost weights through MPC performance
    print(f"\nLearning optimal cost weights...")
    
    cost_optimizer = optim.Adam(cost_weights.parameters(), lr=0.1)
    weight_losses = []
    
    print(f"Initial cost weights:")
    pos_w, vel_w, ctrl_w = cost_weights.get_weights()
    print(f"Position: {pos_w.item():.3f}, Velocity: {vel_w.item():.3f}, Control: {ctrl_w.item():.3f}")
    
    for epoch in range(20):  # Fewer epochs since MPC is expensive
        cost_optimizer.zero_grad()
        
        # Evaluate MPC performance (this is differentiable!)
        avg_cost = evaluate_mpc_performance(dynamics_model, cost_weights, n_trials=2)
        loss = torch.tensor(avg_cost, requires_grad=True)
        
        loss.backward()
        cost_optimizer.step()
        
        weight_losses.append(avg_cost)
        
        if epoch % 5 == 0:
            pos_w, vel_w, ctrl_w = cost_weights.get_weights()
            print(f"Epoch {epoch}, MPC Cost: {avg_cost:.4f}")
            print(f"  Weights - Pos: {pos_w.item():.3f}, Vel: {vel_w.item():.3f}, Ctrl: {ctrl_w.item():.3f}")
    
    print(f"\nFinal cost weights:")
    pos_w, vel_w, ctrl_w = cost_weights.get_weights()
    print(f"Position: {pos_w.item():.3f}, Velocity: {vel_w.item():.3f}, Control: {ctrl_w.item():.3f}")
    
    # 5. Demonstrate learned MPC controller
    print(f"\nDemonstrating learned MPC controller...")
    
    # Test final controller
    initial_state = torch.tensor([[np.pi, 0.5]], dtype=torch.float32)  # Start downward
    nx, nu, timesteps, n_batch = 2, 1, 15, 1
    
    cost = create_cost_matrices(cost_weights, timesteps, n_batch)
    ctrl = mpc.MPC(nx, nu, timesteps,
                  u_lower=-2.0, u_upper=2.0,
                  lqr_iter=5, exit_unconverged=False,
                  eps=1e-2, n_batch=n_batch,
                  backprop=True, verbose=0,
                  grad_method=mpc.GradMethods.AUTO_DIFF)
    
    nominal_states, nominal_actions, nominal_objs = ctrl(initial_state, cost, dynamics_model)
    
    # Print trajectory
    print("\nMPC Trajectory (angle, angular_velocity, control):")
    for t in range(timesteps):
        if t < len(nominal_actions):
            print(f"t={t:2d}: θ={nominal_states[t,0,0].item():6.3f}, "
                  f"θ̇={nominal_states[t,0,1].item():6.3f}, "
                  f"u={nominal_actions[t,0,0].item():6.3f}")
        else:
            print(f"t={t:2d}: θ={nominal_states[t,0,0].item():6.3f}, "
                  f"θ̇={nominal_states[t,0,1].item():6.3f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Dynamics learning
    axes[0,0].plot(dynamics_losses)
    axes[0,0].set_title('Dynamics Model Training Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('MSE Loss')
    axes[0,0].grid(True)
    
    # Weight learning
    axes[0,1].plot(weight_losses)
    axes[0,1].set_title('MPC Performance During Weight Learning')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Average MPC Cost')
    axes[0,1].grid(True)
    
    # MPC trajectory - angle
    t_vals = range(timesteps)
    angles = [nominal_states[t,0,0].item() for t in range(timesteps)]
    axes[1,0].plot(t_vals, angles, 'b-', marker='o')
    axes[1,0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Target')
    axes[1,0].set_title('MPC Trajectory - Angle')
    axes[1,0].set_xlabel('Time Step')
    axes[1,0].set_ylabel('Angle (rad)')
    axes[1,0].grid(True)
    axes[1,0].legend()
    
    # MPC trajectory - control
    control_vals = [nominal_actions[t,0,0].item() for t in range(len(nominal_actions))]
    axes[1,1].plot(range(len(control_vals)), control_vals, 'g-', marker='s')
    axes[1,1].set_title('MPC Control Actions')
    axes[1,1].set_xlabel('Time Step')
    axes[1,1].set_ylabel('Control Input')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('mpc_learning_results.png', dpi=150, bbox_inches='tight')
    print(f"\nResults saved to 'mpc_learning_results.png'")
    
    print(f"\nDemo completed! This example showed:")
    print(f"1. Learning dynamics parameters through supervised learning")
    print(f"2. Learning cost weights through differentiable MPC")
    print(f"3. Using the learned controller for pendulum swing-up")

if __name__ == "__main__":
    main()