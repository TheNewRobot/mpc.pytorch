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
from simple_pendulum import PendulumMPCController

class DualPendulumComparison(PendulumMPCController):
    """Compare two pendulum models with different dynamics parameters."""
    
    def __init__(self, 
                 true_params=[10.0, 1.0, 1.0],
                 learned_params=[10.1, 1.1, 1.2],
                 **kwargs):
        
        # Initialize base class with true parameters
        kwargs.setdefault('gravity', true_params[0])
        kwargs.setdefault('mass', true_params[1])
        kwargs.setdefault('length', true_params[2])
        kwargs.setdefault('name_dir', 'pendulum_comparison')
        kwargs.setdefault('verbose', False)  # Reduce verbosity for cleaner output
        
        super().__init__(**kwargs)
        
        # Store parameters
        self.true_params = torch.tensor(true_params)
        self.learned_params = torch.tensor(learned_params)
        
        # Create both dynamics models
        self.true_dynamics = pendulum.PendulumDx(self.true_params, simple=True)
        self.learned_dynamics = pendulum.PendulumDx(self.learned_params, simple=True)
        
        # Storage for comparison data
        self.true_trajectory = {'states': [], 'actions': [], 'costs': []}
        self.learned_trajectory = {'states': [], 'actions': [], 'costs': []}
        
        if kwargs.get('verbose', False):
            print(f"True params (g,m,l): {true_params}")
            print(f"Learned params (g,m,l): {learned_params}")
    
    def run_dual_simulation(self):
        """Run both pendulums simultaneously and collect data."""
        # Initialize both with same initial state
        true_state = self.get_initial_state()
        learned_state = true_state.clone()
        
        # Create MPC controllers for both
        true_mpc = mpc.MPC(
            self.n_states, self.n_control, self.mpc_horizon,
            u_lower=-self.control_authority, u_upper=self.control_authority,
            lqr_iter=self.lqr_iterations, verbose=-1, exit_unconverged=False,
            grad_method=GradMethods.AUTO_DIFF, eps=self.convergence_eps
        )
        
        learned_mpc = mpc.MPC(
            self.n_states, self.n_control, self.mpc_horizon,
            u_lower=-self.control_authority, u_upper=self.control_authority,
            lqr_iter=self.lqr_iterations, verbose=-1, exit_unconverged=False,
            grad_method=GradMethods.AUTO_DIFF, eps=self.convergence_eps
        )
        
        # Main simulation loop
        progress = tqdm(range(self.total_timesteps), desc="Simulating both pendulums")
        
        for t in progress:
            # Solve MPC for true dynamics
            try:
                _, true_actions, true_cost = true_mpc(
                    true_state, QuadCost(self.Q, self.p), self.true_dynamics
                )
                true_action = true_actions[0]
            except:
                true_action = torch.zeros(1, 1)
                true_cost = [torch.tensor(0.0)]
            
            # Solve MPC for learned dynamics
            try:
                _, learned_actions, learned_cost = learned_mpc(
                    learned_state, QuadCost(self.Q, self.p), self.learned_dynamics
                )
                learned_action = learned_actions[0]
            except:
                learned_action = torch.zeros(1, 1)
                learned_cost = [torch.tensor(0.0)]
            
            # Store data
            self.true_trajectory['states'].append(true_state.clone())
            self.true_trajectory['actions'].append(true_action.clone())
            self.true_trajectory['costs'].append(true_cost[0].item())
            
            self.learned_trajectory['states'].append(learned_state.clone())
            self.learned_trajectory['actions'].append(learned_action.clone())
            self.learned_trajectory['costs'].append(learned_cost[0].item())
            
            # Update states with respective dynamics
            true_state = self.true_dynamics(true_state, true_action)
            learned_state = self.learned_dynamics(learned_state, learned_action)
            
            # Save comparison frame
            if self.save_video:
                self.save_comparison_frame(true_state, learned_state, t)
        
        return self.analyze_results()
    
    def save_comparison_frame(self, true_state, learned_state, timestep):
        """Save overlaid pendulum comparison frame with same origin."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
        
        # Get pendulum length for consistent scaling
        l = self.true_dynamics.params[2].item()
        
        # Extract states
        cos_th1, sin_th1, dth1 = true_state.squeeze(0)
        cos_th2, sin_th2, dth2 = learned_state.squeeze(0)
        
        # Calculate positions for both pendulums
        x1, y1 = sin_th1.item() * l, cos_th1.item() * l  # True pendulum
        x2, y2 = sin_th2.item() * l, cos_th2.item() * l  # Learned pendulum
        
        # Calculate angles for display
        angle1 = np.arctan2(sin_th1.item(), cos_th1.item()) * 180 / np.pi
        angle2 = np.arctan2(sin_th2.item(), cos_th2.item()) * 180 / np.pi
        
        # Plot both pendulums with same origin (0,0)
        ax.plot([0, x1], [0, y1], 'b-', linewidth=6, label='True', alpha=0.8)
        ax.plot([0, x2], [0, y2], 'r-', linewidth=4, label='Learned', alpha=0.8)
        
        # Add pendulum bobs (circles at the end)
        ax.plot(x1, y1, 'bo', markersize=12, alpha=0.8)
        ax.plot(x2, y2, 'ro', markersize=10, alpha=0.8)
        
        # Add pivot point
        ax.plot(0, 0, 'ko', markersize=8, label='Pivot')
        
        # Set equal aspect ratio and limits
        ax.set_xlim((-l*1.3, l*1.3))
        ax.set_ylim((-l*1.3, l*1.3))
        ax.set_aspect('equal')
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        # Title with current states
        ax.set_title(f'Step {timestep} - Overlaid Pendulum Comparison\n'
                    f'True: θ={angle1:.1f}°, ω={dth1.item():.2f} rad/s\n'
                    f'Learned: θ={angle2:.1f}°, ω={dth2.item():.2f} rad/s\n'
                    f'Δθ={abs(angle1-angle2):.1f}°', 
                    fontsize=12, pad=15)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        plt.tight_layout()
        frame_path = os.path.join(self.experiment_dir, f'{timestep:03d}.png')
        fig.savefig(frame_path, bbox_inches='tight', dpi=100, facecolor='white')
        plt.close(fig)
    
    def create_comparison_plots(self):
        """Create detailed comparison plots."""
        # Convert stored data to arrays and detach gradients
        true_states = torch.cat(self.true_trajectory['states'], dim=0).detach()
        learned_states = torch.cat(self.learned_trajectory['states'], dim=0).detach()
        true_actions = torch.cat(self.true_trajectory['actions'], dim=0).detach()
        learned_actions = torch.cat(self.learned_trajectory['actions'], dim=0).detach()
        
        # Extract time series
        time = np.arange(len(true_states)) * self.true_dynamics.dt
        
        # True pendulum data
        true_cos, true_sin, true_vel = true_states.T
        true_angles = np.arctan2(true_sin.numpy(), true_cos.numpy()) * 180 / np.pi
        true_controls = true_actions.squeeze().numpy()
        
        # Learned pendulum data
        learned_cos, learned_sin, learned_vel = learned_states.T
        learned_angles = np.arctan2(learned_sin.numpy(), learned_cos.numpy()) * 180 / np.pi
        learned_controls = learned_actions.squeeze().numpy()
        
        # Create comparison plot (vertical layout)
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Angle comparison
        axes[0].plot(time, true_angles, 'b-', linewidth=2, label=f'True {self.true_params.tolist()}')
        axes[0].plot(time, learned_angles, 'r--', linewidth=2, label=f'Learned {self.learned_params.tolist()}')
        axes[0].set_ylabel('Angle (degrees)')
        axes[0].set_title('Pendulum Angle Comparison')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3, label='Upright')
        
        # Angular velocity comparison
        axes[1].plot(time, true_vel.numpy(), 'b-', linewidth=2, label='True')
        axes[1].plot(time, learned_vel.numpy(), 'r--', linewidth=2, label='Learned')
        axes[1].set_ylabel('Angular Velocity (rad/s)')
        axes[1].set_title('Angular Velocity Comparison')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Control input comparison
        axes[2].plot(time, true_controls, 'b-', linewidth=2, label='True')
        axes[2].plot(time, learned_controls, 'r--', linewidth=2, label='Learned')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Control Torque (N⋅m)')
        axes[2].set_title('Control Input Comparison')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        axes[2].axhline(y=self.control_authority, color='k', linestyle=':', alpha=0.5, label='Limits')
        axes[2].axhline(y=-self.control_authority, color='k', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        plot_path = os.path.join(self.experiment_dir, 'comparison_plots.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Calculate RMS differences for quantitative comparison
        angle_rms = np.sqrt(np.mean((true_angles - learned_angles)**2))
        velocity_rms = np.sqrt(np.mean((true_vel.numpy() - learned_vel.numpy())**2))
        control_rms = np.sqrt(np.mean((true_controls - learned_controls)**2))
        
        return plot_path, {
            'angle_rms': angle_rms,
            'velocity_rms': velocity_rms,
            'control_rms': control_rms
        }
    
    def analyze_results(self):
        """Analyze the performance of both models."""
        # Final states
        true_final = self.true_trajectory['states'][-1].squeeze(0)
        learned_final = self.learned_trajectory['states'][-1].squeeze(0)
        
        # Final angles
        true_final_angle = np.arctan2(true_final[1].item(), true_final[0].item()) * 180 / np.pi
        learned_final_angle = np.arctan2(learned_final[1].item(), learned_final[0].item()) * 180 / np.pi
        
        # Success criteria (within 10 degrees of upright)
        true_success = abs(true_final_angle) < 10
        learned_success = abs(learned_final_angle) < 10
        
        # Total costs
        true_total_cost = sum(self.true_trajectory['costs'])
        learned_total_cost = sum(self.learned_trajectory['costs'])
        
        # Create comparison plots
        plot_path, rms_metrics = self.create_comparison_plots()
        
        # Create video
        video_path = self.create_video() if self.save_video else None
        
        results = {
            'true_params': self.true_params.tolist(),
            'learned_params': self.learned_params.tolist(),
            'true_final_angle': true_final_angle,
            'learned_final_angle': learned_final_angle,
            'true_success': true_success,
            'learned_success': learned_success,
            'true_total_cost': true_total_cost,
            'learned_total_cost': learned_total_cost,
            'angle_difference': abs(true_final_angle - learned_final_angle),
            'cost_difference': abs(true_total_cost - learned_total_cost),
            'rms_metrics': rms_metrics,
            'plot_path': plot_path,
            'video_path': video_path
        }
        
        return results
    
    def create_video(self):
        """Create comparison video."""
        video_path = os.path.join(self.experiment_dir, 'pendulum_comparison.mp4')
        cmd = f'ffmpeg -y -r 16 -i {self.experiment_dir}/%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -vcodec libx264 -pix_fmt yuv420p {video_path}'
        
        if os.system(cmd) == 0:
            # Clean up frames
            for f in os.listdir(self.experiment_dir):
                if f.endswith('.png') and f != 'comparison_plots.png':
                    os.remove(os.path.join(self.experiment_dir, f))
            return video_path
        return None

def main():
    """Run pendulum comparison experiment."""
    
    # Example: Compare true vs learned parameters
    comparison = DualPendulumComparison(
        true_params=[10.0, 1.0, 1.0],      # Ground truth
        learned_params=[10.12, 0.87, 1.09],   # Learned parameters
        mpc_horizon=20,
        total_timesteps=150,
        lqr_iterations=50,
        save_video=True,
        verbose=True
    )
    
    print("Running dual pendulum comparison...")
    results = comparison.run_dual_simulation()
    
    # Print results
    print("\n" + "="*60)
    print("PENDULUM COMPARISON RESULTS")
    print("="*60)
    print(f"True params (g,m,l):    {results['true_params']}")
    print(f"Learned params (g,m,l): {results['learned_params']}")
    print(f"")
    print(f"Final Angles:")
    print(f"  True:    {results['true_final_angle']:6.1f}° ({'Success' if results['true_success'] else 'Failed'})")
    print(f"  Learned: {results['learned_final_angle']:6.1f}° ({'Success' if results['learned_success'] else 'Failed'})")
    print(f"  Difference: {results['angle_difference']:6.1f}°")
    print(f"")
    print(f"Total Costs:")
    print(f"  True:    {results['true_total_cost']:8.2f}")
    print(f"  Learned: {results['learned_total_cost']:8.2f}")
    print(f"  Difference: {results['cost_difference']:8.2f}")
    print(f"")
    print(f"RMS Differences (Quantitative Analysis):")
    print(f"  Angle RMS:     {results['rms_metrics']['angle_rms']:8.3f}°")
    print(f"  Velocity RMS:  {results['rms_metrics']['velocity_rms']:8.3f} rad/s")
    print(f"  Control RMS:   {results['rms_metrics']['control_rms']:8.3f} N⋅m")
    print(f"")
    if results['video_path']:
        print(f"Video: {results['video_path']}")
    if results['plot_path']:
        print(f"Plots: {results['plot_path']}")

if __name__ == '__main__':
    main()