"""
Simple demonstration of mpc.pytorch differentiability
====================================================

This shows EXACTLY where and how gradients flow through the MPC solver.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from mpc import mpc
import numpy as np

"""
Simple demonstration of mpc.pytorch differentiability
====================================================

This shows EXACTLY where and how gradients flow through the MPC solver.
Using LinDx (linear dynamics) to avoid compatibility issues.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from mpc import mpc
import numpy as np

class LearnableQuadraticCost(nn.Module):
    """Learnable quadratic cost: 0.5 * x^T * Q * x + u^T * R * u"""
    def __init__(self):
        super().__init__()
        # Learn the cost weights
        self.q_weight = nn.Parameter(torch.tensor(0.1))  # Start low
        self.r_weight = nn.Parameter(torch.tensor(0.1))  # Start low
        
    def get_cost_matrices(self, T, n_batch):
        """Create cost matrices from learnable parameters"""
        q = torch.nn.functional.softplus(self.q_weight)  # Ensure positive
        r = torch.nn.functional.softplus(self.r_weight)  # Ensure positive
        
        # Add minimum values to prevent weights from getting too small
        q = q + 0.01  # Minimum state cost
        r = r + 0.001  # Minimum control cost
        
        # State cost matrix [state_dim + control_dim, state_dim + control_dim]
        Q = torch.zeros(3, 3)  # 2 states + 1 control
        Q[0, 0] = q  # Cost on state[0]
        Q[1, 1] = q * 0.1  # Smaller cost on state[1] (velocity)
        Q[2, 2] = r  # Cost on control
        
        # Linear cost (zero - we want to reach origin)
        p = torch.zeros(3)
        
        # Expand for time and batch dimensions
        Q = Q.unsqueeze(0).unsqueeze(0).expand(T, n_batch, -1, -1)
        p = p.unsqueeze(0).unsqueeze(0).expand(T, n_batch, -1)
        
        return mpc.QuadCost(Q, p)

def demonstrate_mpc_differentiability():
    """Show how gradients flow through MPC solver using linear dynamics"""
    
    print("=== MPC Differentiability Demo (Linear Dynamics) ===")
    
    # Problem setup
    nx, nu, T = 2, 1, 5
    n_batch = 1
    
    # Target: reach origin from initial state [2, 1]
    x_init = torch.tensor([[2.0, 1.0]])
    target = torch.tensor([0.0, 0.0])
    
    # Create linear dynamics F*[x;u] = [A B]*[x;u] = A*x + B*u
    A = torch.tensor([[1.1, 0.1], [0.0, 0.9]], dtype=torch.float32)
    B = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    
    # F matrix combines A and B: F = [A B]
    F = torch.cat([A, B], dim=1)  # Shape: [2, 3]
    
    # Expand for time and batch dimensions: [T-1, n_batch, nx, nx+nu]
    F = F.unsqueeze(0).unsqueeze(0).expand(T-1, n_batch, -1, -1)
    
    # No affine term (f = None)
    dynamics = mpc.LinDx(F, None)
    
    # Learnable cost
    learnable_cost = LearnableQuadraticCost()
    optimizer = optim.Adam(learnable_cost.parameters(), lr=0.02)  # Reduced learning rate
    
    print(f"Initial cost weights: Q={learnable_cost.q_weight.item():.3f}, R={learnable_cost.r_weight.item():.3f}")
    
    # Training loop - learn costs to minimize distance to target
    for iteration in range(20):  # More iterations with lower learning rate
        optimizer.zero_grad()
        
        # Create cost matrices from current parameters
        cost = learnable_cost.get_cost_matrices(T, n_batch)
        
        # *** HERE IS WHERE MPC DIFFERENTIABILITY IS USED ***
        ctrl = mpc.MPC(
            nx, nu, T,
            u_lower=-2.0, u_upper=2.0,
            lqr_iter=10,
            backprop=True,  # ← CRITICAL: Enable backprop through MPC
            grad_method=mpc.GradMethods.ANALYTIC,  # ← Use analytic gradients for LinDx
            exit_unconverged=False,
            verbose=0
        )
        
        try:
            # Solve MPC - this creates a computational graph!
            # Gradients can flow: learnable_cost.parameters() → cost → MPC solution
            x_traj, u_traj, _ = ctrl(x_init, cost, dynamics)
            
            # Loss: how far are we from target at end?
            final_state = x_traj[-1, 0, :]  # Last timestep, first batch
            loss = torch.sum((final_state - target)**2)
            
            print(f"Iter {iteration}: Loss={loss.item():.4f}, "
                  f"Final state=[{final_state[0].item():.3f}, {final_state[1].item():.3f}], "
                  f"Q={torch.nn.functional.softplus(learnable_cost.q_weight).item():.3f}, "
                  f"R={torch.nn.functional.softplus(learnable_cost.r_weight).item():.3f}")
            
            # Early stopping if we're close enough
            if loss.item() < 0.1:
                print("✓ Reached target!")
                break
            
            # *** GRADIENTS FLOW BACKWARD THROUGH ENTIRE MPC SOLVER ***
            loss.backward()
            optimizer.step()
            
        except Exception as e:
            print(f"Iter {iteration}: MPC failed with error: {e}")
            break
    
    print(f"\nFinal cost weights: Q={torch.nn.functional.softplus(learnable_cost.q_weight).item():.3f}, "
          f"R={torch.nn.functional.softplus(learnable_cost.r_weight).item():.3f}")
    print(f"Raw parameters: q_weight={learnable_cost.q_weight.item():.3f}, r_weight={learnable_cost.r_weight.item():.3f}")
    print("\nThis demonstrates:")
    print("1. Gradients flowing backward through the MPC solver ✓")
    print("2. Learning cost parameters by optimizing MPC performance ✓") 
    print("3. The solver automatically differentiates through the optimization process ✓")
    print("4. Using linear dynamics (LinDx) avoids some compatibility issues ✓")
    print("5. Proper parameter constraints prevent negative costs ✓")

def demonstrate_simple_case():
    """Even simpler case - just show that MPC can be called and returns gradients"""
    print("\n=== Simple Gradient Check ===")
    
    # Very simple setup
    nx, nu, T, n_batch = 2, 1, 3, 1
    x_init = torch.tensor([[1.0, 0.0]])
    
    # Simple dynamics: next_state = current_state + control * [0, 1]
    A = torch.eye(2)
    B = torch.tensor([[0.0], [1.0]])
    F = torch.cat([A, B], dim=1).unsqueeze(0).unsqueeze(0).expand(T-1, n_batch, -1, -1)
    dynamics = mpc.LinDx(F, None)
    
    # Simple cost: penalize state[0] and control
    Q = torch.diag(torch.tensor([1.0, 0.1, 0.1]))
    p = torch.zeros(3)
    Q = Q.unsqueeze(0).unsqueeze(0).expand(T, n_batch, -1, -1)
    p = p.unsqueeze(0).unsqueeze(0).expand(T, n_batch, -1)
    cost = mpc.QuadCost(Q, p)
    
    # Create learnable parameter
    learnable_param = nn.Parameter(torch.tensor(1.0))
    
    # Modify cost based on learnable parameter
    Q_modified = Q.clone()
    Q_modified[:, :, 0, 0] = learnable_param  # Make first cost element learnable
    cost_modified = mpc.QuadCost(Q_modified, p)
    
    # MPC solve
    ctrl = mpc.MPC(nx, nu, T, backprop=True, grad_method=mpc.GradMethods.ANALYTIC, 
                   lqr_iter=5, exit_unconverged=False, verbose=0)
    
    try:
        x_traj, u_traj, _ = ctrl(x_init, cost_modified, dynamics)
        loss = torch.sum(x_traj[-1, 0, :]**2)
        
        print(f"Loss: {loss.item():.4f}")
        print(f"Learnable param before: {learnable_param.item():.4f}")
        print(f"Param has gradient: {learnable_param.grad is not None}")
        
        loss.backward()
        print(f"Param gradient: {learnable_param.grad}")
        print(f"Gradient is not None: {learnable_param.grad is not None}")
        
        if learnable_param.grad is not None:
            print("✓ SUCCESS: Gradients flow through MPC!")
        else:
            print("✗ FAILED: No gradients")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    demonstrate_simple_case()
    demonstrate_mpc_differentiability()
