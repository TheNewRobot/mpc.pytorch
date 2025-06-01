"""
Simple Pendulum Test
===================

Let's test if the basic setup works without MPC complexity.
This will help us identify the real issues.
"""

import gymnasium as gym
import numpy as np
import math
import time

def angle_normalize(x):
    return math.atan2(math.sin(x), math.cos(x))

class SimpleEnergyController:
    """Simple energy-based controller that should work"""
    
    def __init__(self, action_low=-2.0, action_high=2.0):
        self.action_low = action_low
        self.action_high = action_high
    
    def get_action(self, obs):
        # Convert observation to state
        theta = math.atan2(obs[1], obs[0])  # atan2(sin, cos)
        theta_dot = obs[2]
        
        # Normalize angle
        theta = angle_normalize(theta)
        
        # Energy-based swing-up control
        # Current energy (normalized)
        E = 0.5 * theta_dot**2 + 10.0 * (1 - math.cos(theta))
        # Desired energy to reach top
        E_desired = 20.0
        
        if abs(theta) < 0.5 and abs(theta_dot) < 2.0:
            # Near upright - stabilize
            action = -10.0 * theta - 3.0 * theta_dot
        else:
            # Swing up - pump energy
            if E < E_desired:
                # Add energy
                action = 2.0 * math.copysign(1, theta_dot * math.cos(theta))
            else:
                # Damp oscillations
                action = -0.5 * theta_dot
        
        # Clip action
        return np.clip(action, self.action_low, self.action_high)

def test_simple_controller():
    """Test if simple controller can swing up the pendulum"""
    
    print("Testing Simple Energy Controller...")
    
    # Create environment
    env = gym.make("Pendulum-v1", render_mode="human", g=10.0)
    controller = SimpleEnergyController()
    
    # Initialize
    obs, _ = env.reset()
    env.unwrapped.state = np.array([np.pi, 0.0], dtype=np.float32)  # Start hanging down
    obs = env.unwrapped._get_obs()
    
    total_reward = 0
    success_count = 0
    
    print("Starting from hanging down position...")
    print("Goal: Swing up to upright position (0°)")
    
    for i in range(1000):
        # Get action
        action = controller.get_action(obs)
        
        # Step environment
        obs, reward, terminated, truncated, _ = env.step([action])
        total_reward += reward
        
        # Get current angle for monitoring
        theta = math.atan2(obs[1], obs[0])
        theta = angle_normalize(theta)
        theta_dot = obs[2]
        
        # Check success
        if abs(theta) < 0.2 and abs(theta_dot) < 1.0:
            success_count += 1
        else:
            success_count = 0
            
        # Progress report
        if i % 50 == 0:
            print(f"Step {i}: θ={theta:.3f} ({math.degrees(theta):.1f}°), "
                  f"θ̇={theta_dot:.3f}, action={action:.3f}, reward={reward:.3f}")
        
        # Check for success
        if success_count > 100:
            print(f"\n🎉 SUCCESS! Pendulum balanced at step {i}")
            print(f"Final angle: {theta:.3f} rad ({math.degrees(theta):.1f}°)")
            break
        
        # Render
        env.render()
        time.sleep(0.01)  # Small delay for visualization
    
    print(f"\nResults:")
    print(f"Total steps: {i+1}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final angle: {theta:.3f} rad ({math.degrees(theta):.1f}°)")
    print(f"Success: {'YES' if success_count > 100 else 'NO'}")
    
    env.close()
    return success_count > 100

def test_true_dynamics():
    """Test if we can manually implement the true dynamics correctly"""
    
    print("\nTesting True Dynamics Implementation...")
    
    # Test parameters
    g = 10.0
    m = 1.0
    l = 1.0
    dt = 0.05
    
    # Test state: hanging down with small velocity
    theta = np.pi  # hanging down
    theta_dot = 0.1
    action = 1.0  # positive torque
    
    print(f"Initial state: θ={theta:.3f} ({math.degrees(theta):.1f}°), θ̇={theta_dot:.3f}")
    print(f"Action: {action}")
    
    # CORRECTED True dynamics - pendulum equation
    # For pendulum: θ̈ = -g/l * sin(θ) + torque/(m*l²)
    # Note: θ=0 is upright, θ=π is hanging down
    theta_ddot = -g / l * math.sin(theta) + action / (m * l**2)
    new_theta_dot = theta_dot + theta_ddot * dt
    new_theta = theta + new_theta_dot * dt
    
    print(f"θ̈={theta_ddot:.3f}")
    print(f"Next state: θ={new_theta:.3f} ({math.degrees(new_theta):.1f}°), θ̇={new_theta_dot:.3f}")
    
    # When hanging down (θ≈π), positive torque should reduce θ (move towards 0)
    # So |new_theta - π| should be less than |theta - π| = 0
    distance_from_upright_old = abs(theta)  # Distance from θ=0
    distance_from_upright_new = abs(angle_normalize(new_theta))
    
    print(f"Distance from upright: {distance_from_upright_old:.3f} → {distance_from_upright_new:.3f}")
    
    if distance_from_upright_new < distance_from_upright_old:
        print("✅ Dynamics working correctly - pendulum moving towards upright")
        return True
    else:
        print("❌ Dynamics issue - pendulum not moving as expected")
        
        # Test the gym environment's actual dynamics
        print("\nTesting Gym Environment Dynamics...")
        env = gym.make("Pendulum-v1", g=g)
        obs, _ = env.reset()
        env.unwrapped.state = np.array([theta, theta_dot], dtype=np.float32)
        obs_before = env.unwrapped._get_obs()
        
        obs_after, reward, _, _, _ = env.step([action])
        state_after = env.unwrapped.state
        
        print(f"Gym result: θ={state_after[0]:.3f} ({math.degrees(state_after[0]):.1f}°), θ̇={state_after[1]:.3f}")
        
        env.close()
        return False

def main():
    print("="*60)
    print("PENDULUM DIAGNOSIS")
    print("="*60)
    
    # Test 1: True dynamics
    dynamics_ok = test_true_dynamics()
    
    # Test 2: Simple controller
    if dynamics_ok:
        controller_ok = test_simple_controller()
        
        if controller_ok:
            print("\n✅ DIAGNOSIS: Basic setup works!")
            print("The issue is likely in:")
            print("1. MPC optimization parameters")
            print("2. Neural network learning")
            print("3. Cost function formulation")
        else:
            print("\n❌ DIAGNOSIS: Controller issues")
            print("Even simple controller fails - check:")
            print("1. Environment setup")
            print("2. Action limits")
            print("3. Reward function")
    else:
        print("\n❌ DIAGNOSIS: Dynamics implementation issues")

if __name__ == "__main__":
    main()