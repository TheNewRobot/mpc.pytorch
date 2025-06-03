"""
Quick Deployment Test Script
===========================

A simplified version of the deployment script that can work with 
models trained from the learning pipeline.

Usage:
    python quick_deploy_test.py --model-path <path_to_model.pt> [--render]
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import math
from mpc import mpc

parser = argparse.ArgumentParser(description='Quick deployment test')
parser.add_argument('--model-path', type=str, help='Path to saved model weights (.pt file)')
parser.add_argument('--render', action='store_true', help='Enable rendering')
parser.add_argument('--steps', type=int, default=200, help='Number of steps to run')
args = parser.parse_args()

def angle_normalize(x):
    """Normalize angle to [-pi, pi]"""
    if torch.is_tensor(x):
        return torch.atan2(torch.sin(x), torch.cos(x))
    else:
        return math.atan2(math.sin(x), math.cos(x))

def obs_to_state(obs):
    """Convert gym observation to state"""
    if torch.is_tensor(obs):
        theta = torch.atan2(obs[..., 1], obs[..., 0])
        theta_dot = obs[..., 2]
        return torch.stack([theta, theta_dot], dim=-1)
    else:
        theta = np.arctan2(obs[1], obs[0])
        theta_dot = obs[2]
        return np.array([theta, theta_dot])

class DeploymentDynamics(torch.nn.Module):
    """Simple deployment dynamics model"""
    
    def __init__(self, hidden_units=64, u_min=-2, u_max=2, thdot_min=-8, thdot_max=8):
        super().__init__()
        self.u_min = u_min
        self.u_max = u_max
        self.thdot_min = thdot_min
        self.thdot_max = thdot_max
        
        self.net = nn.Sequential(
            nn.Linear(4, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units//2),
            nn.ReLU(),
            nn.Linear(hidden_units//2, 2)
        ).double()
        
    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if action.dim() == 3:
            action = action.squeeze(0)
        if state.dim() == 3:
            state = state.squeeze(0)
        if action.shape[1] != 1:
            action = action[:, :1]
            
        th = state[:, 0:1]
        thdot = state[:, 1:2]
        u = torch.clamp(action, self.u_min, self.u_max)
        th_normalized = angle_normalize(th)
        
        input_features = torch.cat([torch.sin(th_normalized), torch.cos(th_normalized), thdot, u], dim=1)
        delta_state = self.net(input_features)
        
        next_th = th + delta_state[:, 0:1]
        next_thdot = thdot + delta_state[:, 1:2]
        
        next_th = angle_normalize(next_th)
        next_thdot = torch.clamp(next_thdot, self.thdot_min, self.thdot_max)
        
        return torch.cat([next_th, next_thdot], dim=1)

def main():
    print("=== Quick Deployment Test ===")
    
    # Load model
    if args.model_path:
        print(f"Loading model from {args.model_path}")
        model = DeploymentDynamics()
        model.net.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        model.eval()
        print("Model loaded successfully")
    else:
        print("No model path provided, using untrained model for testing")
        model = DeploymentDynamics()
    
    # Create environment
    env = gym.make("Pendulum-v1", render_mode="human" if args.render else None,
                   disable_env_checker=True, g=9.81)
    
    # MPC setup
    TIMESTEPS = 10
    N_BATCH = 1
    LQR_ITER = 5
    nx, nu = 2, 1
    
    goal_weights = torch.tensor([10.0, 1.0], dtype=torch.double)
    goal_state = torch.tensor([0.0, 0.0], dtype=torch.double)
    ctrl_penalty = 0.01
    
    q = torch.cat([goal_weights, ctrl_penalty * torch.ones(nu, dtype=torch.double)])
    px = -torch.sqrt(goal_weights) * goal_state
    p = torch.cat([px, torch.zeros(nu, dtype=torch.double)])
    Q = torch.diag(q).repeat(TIMESTEPS, N_BATCH, 1, 1)
    p = p.repeat(TIMESTEPS, N_BATCH, 1)
    cost = mpc.QuadCost(Q, p)
    
    # Reset and start from hanging down
    observation, _ = env.reset()
    env.unwrapped.state = np.array([np.pi, 0], dtype=np.float32)
    observation = env.unwrapped._get_obs()
    
    total_reward = 0
    u_init = None
    success_count = 0
    
    print(f"Running for {args.steps} steps...")
    print("Starting from hanging down position, trying to swing up...")
    
    for step in range(args.steps):
        state = obs_to_state(observation)
        state_tensor = torch.tensor(state, dtype=torch.double).view(1, -1)
        
        if args.render:
            env.render()
        
        try:
            with torch.no_grad():
                ctrl = mpc.MPC(nx, nu, TIMESTEPS,
                              u_lower=-2.0, u_upper=2.0,
                              lqr_iter=LQR_ITER,
                              exit_unconverged=False,
                              eps=1e-3,
                              n_batch=N_BATCH,
                              backprop=False,
                              verbose=-1,
                              u_init=u_init,
                              grad_method=mpc.GradMethods.AUTO_DIFF)
                
                nominal_states, nominal_actions, nominal_objs = ctrl(state_tensor, cost, model)
                action = nominal_actions[0].numpy()
                u_init = torch.cat([nominal_actions[1:], torch.zeros(1, N_BATCH, nu, dtype=torch.double)], dim=0)
                
        except Exception as e:
            # Fallback control
            theta = angle_normalize(state[0])
            action = np.array([-5.0 * theta - 1.0 * state[1]])
            action = np.clip(action, -2.0, 2.0)
            u_init = None
            if step % 50 == 0:
                print(f"MPC failed at step {step}, using fallback: {e}")
        
        # Apply action
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
        
        # Check success (upright position)
        angle_error = abs(angle_normalize(state[0]))
        if angle_error < 0.3:  # Within ~17 degrees of upright
            success_count += 1
        
        # Progress report
        if step % 50 == 0:
            print(f"Step {step}: angle={math.degrees(angle_normalize(state[0])):.1f}°, "
                  f"reward={reward[0] if isinstance(reward, np.ndarray) else reward:.2f}")
        
        if terminated or truncated:
            print("Episode ended early")
            break
    
    # Final results
    print("\n" + "="*50)
    print("DEPLOYMENT RESULTS")
    print("="*50)
    print(f"Total steps: {step + 1}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per step: {total_reward/(step+1):.3f}")
    print(f"Success rate (upright): {success_count/(step+1):.2%}")
    
    final_angle = angle_normalize(state[0])
    print(f"Final angle: {math.degrees(final_angle):.1f}° from upright")
    
    if abs(final_angle) < 0.3:
        print("🎯 SUCCESS: Pendulum is upright!")
    elif abs(final_angle) < 1.0:
        print("👍 GOOD: Close to upright position")
    else:
        print("❌ FAILED: Could not balance pendulum")
    
    env.close()

if __name__ == "__main__":
    main()