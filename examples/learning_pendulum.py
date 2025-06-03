#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from datetime import datetime

from mpc import mpc
from mpc.mpc import QuadCost, GradMethods
from mpc.env_dx import pendulum
import json
import pickle

class LearnablePendulumDx(nn.Module):
    """Wrapper around pendulum.PendulumDx with learnable parameters."""
    
    def __init__(self, init_params=torch.tensor([10.0, 1.0, 1.0])):
        super().__init__()
        
        # Make parameters learnable
        self.params = nn.Parameter(init_params.clone())
        
    def forward(self, x, u):
        # Create pendulum dynamics with current parameters
        dynamics = pendulum.PendulumDx(self.params, simple=True)
        return dynamics(x, u)
    
    def grad_input(self, x, u):
        """Compute gradients for differentiable MPC."""
        # Create dynamics with current parameters
        dynamics = pendulum.PendulumDx(self.params, simple=True)
        return dynamics.grad_input(x, u)


class ParameterLearner:
    """Learn pendulum model parameters using differentiable MPC."""
    
    def __init__(self,
                 true_params=torch.tensor([10.0, 1.0, 1.0]),  # g, m, l
                 init_guess=None,
                 mpc_horizon=10,
                 lqr_iterations=10,
                 n_episodes=50,
                 episode_length=80,
                 learning_rate=0.01,
                 # Test
                 initial_angle = np.pi/5,
                 initial_vel = 0.0,
                 verbose=True,
                 save_video=True,
                 seed=42):
        
        self.true_params = true_params
        self.mpc_horizon = mpc_horizon
        self.lqr_iterations = lqr_iterations
        self.n_episodes = n_episodes
        self.episode_length = episode_length
        self.learning_rate = learning_rate
        self.initial_angle = initial_angle
        self.initial_vel = initial_vel
        self.verbose = verbose
        self.save_video = save_video
        
        # Create initial parameter guess (add noise to true params)
        if init_guess is None:
            init_guess = true_params + torch.randn_like(true_params) * 0.5
            init_guess = torch.clamp(init_guess, 0.1, 12.0) # TODO
        self.init_guess = init_guess
        
        # Create true dynamics (ground truth)
        self.true_dynamics = pendulum.PendulumDx(true_params, simple=True)
        
        # Create learnable dynamics
        self.learnable_dynamics = LearnablePendulumDx(init_guess)
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.learnable_dynamics.parameters(), lr=learning_rate)
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = f"pendulum_experiments/learn_params/{timestamp}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if verbose:
            print(f"Experiment: {self.experiment_dir}")
            print(f"True params (g,m,l): {true_params.tolist()}")
            print(f"Initial guess: {init_guess.tolist()}")
    
    def get_cost_matrices(self, dtype=torch.float32):
        """Get standard quadratic cost matrices for swing-up."""
        # Goal: upright poinitial_statesition [cos(0), sin(0), vel] = [1, 0, 0]
        goal_weights = torch.tensor([1.0, 1.0, 0.1], dtype=dtype)  # [cos, sin, vel]
        ctrl_weight = 0.001
        
        # Create Q matrix (state + control)
        q = torch.cat((goal_weights, torch.tensor([ctrl_weight], dtype=dtype)))
        Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(self.mpc_horizon, 1, 1, 1)
        
        # Create linear term p (target upright)
        target = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype)  # [cos, sin, vel, ctrl]
        p = (-q * target).unsqueeze(0).repeat(self.mpc_horizon, 1, 1)
        
        return Q, p
    
    def generate_expert_trajectory(self, initial_state):
        """Generate expert trajectory using true dynamics."""
        Q, p = self.get_cost_matrices()
        # Create MPC with true dynamics
        expert_mpc = mpc.MPC(
            n_state=3, n_ctrl=1, T=self.mpc_horizon,
            u_lower=-2.0, u_upper=2.0,
            lqr_iter=self.lqr_iterations,
            verbose=-1,
            exit_unconverged=False,
            backprop=False,  # No gradients needed for expert
            grad_method=GradMethods.AUTO_DIFF
        )
        
        states = []
        actions = []
        
        state = initial_state.float()
        for _ in range(self.episode_length):
            # Get expert action
            _, action_seq, _ = expert_mpc(state, QuadCost(Q, p), self.true_dynamics)
            action = action_seq[0]  # Take first action
            
            states.append(state)
            actions.append(action)
            
            # Step with true dynamics
            state = self.true_dynamics(state, action)
        
        return torch.cat(states, dim=0), torch.cat(actions, dim=0)
    
    def compute_imitation_loss(self, expert_states, expert_actions):
        """Compute loss by comparing learnable MPC actions to expert actions."""
        Q, p = self.get_cost_matrices()
        
        # Create MPC with learnable dynamics - IMPORTANT: backprop=True!
        learnable_mpc = mpc.MPC(
            n_state=3, n_ctrl=1, T=self.mpc_horizon,
            u_lower=-2.0, u_upper=2.0,
            lqr_iter=self.lqr_iterations,
            verbose=-1,
            exit_unconverged=False,
            backprop=True,  # Enable gradient flow!
            grad_method=GradMethods.AUTO_DIFF
        )
        
        total_loss = 0
        n_steps = min(self.episode_length, len(expert_actions))
        
        for t in range(0, n_steps, 5):  # Sample every 5 steps for efficiency
            try:
                state = expert_states[t:t+1]
                expert_action = expert_actions[t:t+1]
                
                # Get predicted action from learnable MPC
                _, predicted_actions, _ = learnable_mpc(
                    state, QuadCost(Q, p), self.learnable_dynamics
                )
                
                # Imitation loss: match expert's first action
                action_loss = torch.nn.functional.mse_loss(
                    predicted_actions[0:1], expert_action
                )
                total_loss += action_loss
                
            except Exception as e:
                if self.verbose:
                    print(f"MPC failed at step {t}: {e}")
                continue
        
        # Add regularization to keep parameters reasonable
        param_reg = 0.01 * torch.sum((self.learnable_dynamics.params - self.true_params)**2)
        total_loss += param_reg
        
        return total_loss
    
    def train_episode(self):
        """Train for one episode."""
        self.optimizer.zero_grad()
        
        # Generate random initial state
        angle = np.random.uniform(-np.pi, np.pi)
        velocity = np.random.uniform(-1.0, 1.0)
        initial_state = torch.tensor([[np.cos(angle), np.sin(angle), velocity]], dtype=torch.float32)
        
        # Generate expert trajectory
        expert_states, expert_actions = self.generate_expert_trajectory(initial_state)
        
        # Compute loss
        loss = self.compute_imitation_loss(expert_states, expert_actions)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.learnable_dynamics.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Keep parameters in reasonable bounds
        with torch.no_grad():
            self.learnable_dynamics.params.data = torch.clamp(
                self.learnable_dynamics.params.data, 0.1, 12.0 # TODO
            )
        
        return loss.item()
    
    def evaluate_learned_model(self):
        """Test learned model on swing-up task."""
        Q, p = self.get_cost_matrices()
        
        eval_mpc = mpc.MPC(
            n_state=3, n_ctrl=1, T=self.mpc_horizon,
            u_lower=-2.0, u_upper=2.0,
            lqr_iter=self.lqr_iterations,
            verbose=-1,
            exit_unconverged=False,
            backprop=False,
            grad_method=GradMethods.AUTO_DIFF
        )
        
        # Start hanging down
        state = torch.tensor([[np.cos(self.initial_angle), np.sin(self.initial_angle), self.initial_vel]], dtype=torch.float32)
        states = [state]
        
        for t in range(150):
            try:
                _, actions, _ = eval_mpc(state, QuadCost(Q, p), self.learnable_dynamics)
                action = actions[0]
                state = self.learnable_dynamics(state, action).float()
                states.append(state)
                
                if self.save_video:
                    self.save_frame(state, t)
                    
            except Exception as e:
                if self.verbose:
                    print(f"Evaluation failed at step {t}: {e}")
                break
        
        # Check success
        final_state = states[-1]
        cos_th, sin_th, _ = final_state.squeeze(0)
        final_angle = np.arctan2(sin_th.item(), cos_th.item()) * 180 / np.pi
        success = abs(final_angle) < 15  # Within 15 degrees of upright
        
        return success, final_angle
    
    def save_frame(self, state, timestep):
        """Save visualization frame."""
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        
        # Use true dynamics for consistent visualization
        self.true_dynamics.get_frame(state.squeeze(0), ax=ax)
        
        cos_th, sin_th, dth = state.squeeze(0)
        angle = np.arctan2(sin_th.item(), cos_th.item()) * 180 / np.pi
        
        # Show current parameter estimates
        current_params = self.learnable_dynamics.params.detach()
        ax.set_title(f'Learned Model - Step {timestep}\n'
                    f'θ={angle:.1f}°, Params: [{current_params[0]:.2f}, {current_params[1]:.2f}, {current_params[2]:.2f}]')
        ax.axis('off')
        
        frame_path = os.path.join(self.experiment_dir, f'{timestep:03d}.png')
        fig.savefig(frame_path, bbox_inches='tight', dpi=100, 
                   facecolor='white', pad_inches=0.1)
        plt.close(fig)
    
    def create_video(self):
        """Create video from frames."""
        video_path = os.path.join(self.experiment_dir, 'learned_params_swingup.mp4')
        cmd = f'ffmpeg -y -r 16 -i {self.experiment_dir}/%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -vcodec libx264 -pix_fmt yuv420p {video_path}'
        
        if os.system(cmd) == 0:
            # Clean up frames
            for f in os.listdir(self.experiment_dir):
                if f.endswith('.png'):
                    os.remove(os.path.join(self.experiment_dir, f))
            return video_path
        return None
    
    def run_experiment(self):
        """Run the complete parameter learning experiment."""
        if self.verbose:
            print("Starting parameter learning with differentiable MPC...")
        
        losses = []
        param_history = []
        
        # Training loop
        progress = tqdm(range(self.n_episodes)) if self.verbose else range(self.n_episodes)
        
        for episode in progress:
            loss = self.train_episode()
            losses.append(loss)
            
            # Track parameter evolution
            current_params = self.learnable_dynamics.params.detach().clone()
            param_history.append(current_params)
            
            if self.verbose and episode % 2 == 0: # TODO
                param_error = torch.norm(current_params - self.true_params).item()
                progress.set_postfix({
                    'loss': f'{loss:.4f}',
                    'params': f'[{current_params[0]:.2f}, {current_params[1]:.2f}, {current_params[2]:.2f}]',
                    'error': f'{param_error:.3f}'
                })
        
        # Final evaluation
        success, final_angle = self.evaluate_learned_model()
        
        # Create video
        video_path = self.create_video() if self.save_video else None
        
        # Final results
        final_params = self.learnable_dynamics.params.detach()
        param_error = torch.norm(final_params - self.true_params).item()
        
        results = {
            'final_angle': final_angle,
            'true_params': self.true_params.tolist(),
            'initial_guess': self.init_guess.tolist(),
            'learned_params': final_params.tolist(),
            'param_error': param_error,
            'losses': losses,
            'param_history': [p.tolist() for p in param_history],
            'video_path': video_path
        }
        
        if self.verbose:
            print(f"\nFinal Results:")
            print(f"Success: {success}")
            print(f"True params (g,m,l): {self.true_params.tolist()}")
            print(f"Learned params: {final_params.tolist()}")
            print(f"Parameter error: {param_error:.4f}")
            print(f"Final swing-up angle: {final_angle:.1f}°")
            if video_path:
                print(f"Video saved: {video_path}")
        
        return results
    
    def save_training_logs(self, results):
        """Save training logs for later analysis."""
        
        # Save as JSON (human readable)
        json_path = os.path.join(self.experiment_dir, 'training_logs.json')
        pickle_path = os.path.join(self.experiment_dir, 'training_logs.pkl')

        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)  # Note: using 'logs', not 'results'
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        if self.verbose:
            print(f"Training logs saved to:")
            print(f"  JSON: {json_path}")
            print(f"  Pickle: {pickle_path}")
        
        return json_path, pickle_path

def main():
    """Run parameter learning experiment."""
    
    learner = ParameterLearner(
        true_params=torch.tensor([10.0, 1.0, 1.0]),  # True: g=10, m=1.0, l=1.0
        n_episodes=150,
        learning_rate=0.005,
        verbose=True
    )
    
    results = learner.run_experiment()
    _, _ = learner.save_training_logs(results)
    
    print("\n" + "="*50)
    print("PARAMETER LEARNING SUMMARY")
    print("="*50)
    print(f"Parameter Error: {results['param_error']:.4f}")
    print(f"Swing-up Success: {results['success']}")
    print(f"True params:    {results['true_params']}")
    print(f"Learned params: {results['learned_params']}")

if __name__ == '__main__':
    main()