#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from datetime import datetime

from mpc import mpc
from mpc.mpc import QuadCost, GradMethods
from mpc.env_dx import pendulum


class PendulumMPCController:
    """Compact MPC controller for pendulum swing-up."""
    
    def __init__(self, 
                 # MPC tuning parameters
                 mpc_horizon=20, total_timesteps=100, lqr_iterations=50,
                 convergence_eps=1e-2, control_penalty=0.001,
                 # Physics parameters  
                 gravity=10.0, mass=1.0, length=1.0,
                 # Initial condition
                 initial_angle=torch.pi*3/4,  # Default: hanging down (π), upright: 0, 45°: π/4
                 # Experiment settings
                 save_video=True, verbose=True):
        
        # Store parameters
        self.mpc_horizon = mpc_horizon
        self.total_timesteps = total_timesteps
        self.lqr_iterations = lqr_iterations
        self.convergence_eps = convergence_eps
        self.control_penalty = control_penalty
        self.initial_angle = initial_angle
        self.save_video = save_video
        self.verbose = verbose
        
        # Setup pendulum dynamics
        params = torch.tensor((gravity, mass, length))
        self.dynamics = pendulum.PendulumDx(params, simple=True)
        
        # Setup cost function (upright position: cos(0)=1, sin(0)=0, vel=0)
        goal_weights = torch.tensor([1.0, 1.0, 0.1])  # [pos, pos, vel]
        q = torch.cat((goal_weights, control_penalty * torch.ones(1)))
        px = -torch.sqrt(goal_weights) * torch.tensor([1.0, 0.0, 0.0])
        p = torch.cat((px, torch.zeros(1)))
        
        self.Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(mpc_horizon, 1, 1, 1)
        self.p = p.unsqueeze(0).repeat(mpc_horizon, 1, 1)
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = f"pendulum_experiments/{timestamp}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        if verbose:
            print(f"Experiment: {self.experiment_dir}")
            print(f"Initial angle: {self.initial_angle * 180 / np.pi:.1f}°")
    
    def get_initial_state(self):
        """Get initial state with configurable angle."""
        angle = torch.tensor(self.initial_angle)
        return torch.tensor([[torch.cos(angle), torch.sin(angle), 0.0]])
    
    def save_frame(self, state, timestep):
        """Save visualization frame."""
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        self.dynamics.get_frame(state.squeeze(0), ax=ax)
        
        cos_th, sin_th, dth = state.squeeze(0)
        angle = np.arctan2(sin_th.item(), cos_th.item()) * 180 / np.pi
        ax.set_title(f'Step {timestep}: θ={angle:.1f}°, ω={dth.item():.3f}')
        ax.axis('off')
        
        frame_path = os.path.join(self.experiment_dir, f'{timestep:03d}.png')
        # Save with fixed dimensions (600x600 pixels, divisible by 2)
        fig.savefig(frame_path, bbox_inches='tight', dpi=100, 
                   facecolor='white', pad_inches=0.1)
        plt.close(fig)
        return frame_path
    
    def create_video(self):
        """Create video from frames."""
        video_path = os.path.join(self.experiment_dir, 'pendulum_swingup.mp4')
        # Use scale filter to ensure dimensions are divisible by 2
        cmd = f'ffmpeg -y -r 16 -i {self.experiment_dir}/%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -vcodec libx264 -pix_fmt yuv420p {video_path}'
        
        if os.system(cmd) == 0:
            # Clean up frames
            for f in os.listdir(self.experiment_dir):
                if f.endswith('.png'):
                    os.remove(os.path.join(self.experiment_dir, f))
            return video_path
        else:
            if self.verbose:
                print("Video creation failed - keeping frames for inspection")
            return None
    
    def run_experiment(self):
        """Run the pendulum swing-up experiment."""
        if self.verbose:
            print("Starting pendulum swing-up...")
        
        # Initialize
        state = self.get_initial_state()
        u_init = None
        
        # Create MPC controller
        mpc_controller = mpc.MPC(
            n_state=3, n_ctrl=1, T=self.mpc_horizon,
            u_lower=-2.0, u_upper=2.0, lqr_iter=self.lqr_iterations,
            verbose=0, exit_unconverged=False, detach_unconverged=False,
            grad_method=GradMethods.AUTO_DIFF, eps=self.convergence_eps
        )
        
        costs = []
        
        # Main control loop
        progress = tqdm(range(self.total_timesteps)) if self.verbose else range(self.total_timesteps)
        
        for t in progress:
            # Solve MPC
            _, actions, cost = mpc_controller(state, QuadCost(self.Q, self.p), self.dynamics)
            
            # Apply control and update state
            action = actions[0]
            state = self.dynamics(state, action)
            costs.append(cost[0].item())
            
            # Warm start next iteration
            u_init = torch.cat((actions[1:], actions[-1:]), dim=0)
            
            # Save frame
            if self.save_video:
                self.save_frame(state, t)
            
            # Update progress
            if self.verbose:
                cos_th, sin_th, _ = state.squeeze(0)
                angle = np.arctan2(sin_th.item(), cos_th.item()) * 180 / np.pi
                progress.set_postfix({'angle': f'{angle:.1f}°', 'cost': f'{cost[0].item():.3f}'})
        
        # Final analysis
        cos_th, sin_th, dth = state.squeeze(0)
        final_angle = np.arctan2(sin_th.item(), cos_th.item()) * 180 / np.pi
        success = abs(final_angle) < 10  # Within 10 degrees of upright
        
        # Create video
        video_path = self.create_video() if self.save_video else None
        
        results = {
            'success': success,
            'final_angle': final_angle,
            'final_velocity': dth.item(),
            'total_cost': sum(costs),
            'video_path': video_path
        }
        
        if self.verbose:
            print(f"\nResults: Success={success}, Final angle={final_angle:.1f}°")
            if video_path:
                print(f"Video: {video_path}")
        
        return results


def main():
    """Run pendulum swing-up experiment."""
    controller = PendulumMPCController(
        mpc_horizon=20,
        total_timesteps=100,
        lqr_iterations=50,
        control_penalty=0.001
    )
    
    results = controller.run_experiment()
    print(f"Swing-up {'successful' if results['success'] else 'failed'}!")


if __name__ == '__main__':
    main()