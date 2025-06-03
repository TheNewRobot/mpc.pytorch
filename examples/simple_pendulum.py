#!/usr/bin/env python3

import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods
from mpc.env_dx import pendulum

# Parameters
params = torch.tensor((10., 1., 1.))
dx = pendulum.PendulumDx(params, simple=True)

n_batch, T, mpc_T = 16, 100, 20

def uniform(shape, low, high):
    r = high-low
    return torch.rand(shape)*r+low

# Initialize random starting states
torch.manual_seed(0)
th = uniform(n_batch, -(1/2)*np.pi, (1/2)*np.pi)
thdot = uniform(n_batch, -1., 1.)
xinit = torch.stack((torch.cos(th), torch.sin(th), thdot), dim=1)

x = xinit
u_init = None

# Cost setup for swingup
mode = 'swingup'

if mode == 'swingup':
    goal_weights = torch.Tensor((1., 1., 0.1))
    goal_state = torch.Tensor((1., 0. ,0.))
    ctrl_penalty = 0.001
    q = torch.cat((
        goal_weights,
        ctrl_penalty*torch.ones(dx.n_ctrl)
    ))
    px = -torch.sqrt(goal_weights)*goal_state
    p = torch.cat((px, torch.zeros(dx.n_ctrl)))
    Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
        mpc_T, n_batch, 1, 1
    )
    p = p.unsqueeze(0).repeat(mpc_T, n_batch, 1)

# Create temporary directory for frames
import tempfile
t_dir = tempfile.mkdtemp()
print('Creating frames in temporary directory...')

# Main MPC loop
for t in tqdm(range(T)):
    nominal_states, nominal_actions, nominal_objs = mpc.MPC(
        dx.n_state, dx.n_ctrl, mpc_T,
        u_init=u_init,
        u_lower=dx.lower, u_upper=dx.upper,
        lqr_iter=50,
        verbose=0,
        exit_unconverged=False,
        detach_unconverged=False,
        linesearch_decay=dx.linesearch_decay,
        max_linesearch_iter=dx.max_linesearch_iter,
        grad_method=GradMethods.AUTO_DIFF,
        eps=1e-2,
    )(x, QuadCost(Q, p), dx)
    
    next_action = nominal_actions[0]
    u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, dx.n_ctrl)), dim=0)
    u_init[-2] = u_init[-3]
    x = dx(x, next_action)

    # Create visualization
    n_row, n_col = 4, 4
    fig, axs = plt.subplots(n_row, n_col, figsize=(3*n_col,3*n_row))
    axs = axs.reshape(-1)
    for i in range(n_batch):
        dx.get_frame(x[i], ax=axs[i])
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(t_dir, '{:03d}.png'.format(t)))
    plt.close(fig)

# Create video in current directory
vid_fname = 'pendulum-{}.mp4'.format(mode)

if os.path.exists(vid_fname):
    os.remove(vid_fname)

# Try different ffmpeg commands
cmd = 'ffmpeg -y -r 16 -f image2 -i {}/%03d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p {}'.format(
    t_dir, vid_fname
)
print(f'Running: {cmd}')
result = os.system(cmd)

if result != 0:
    print("ffmpeg failed, trying alternative...")
    cmd = 'ffmpeg -y -framerate 16 -i {}/%03d.png -c:v libx264 -pix_fmt yuv420p {}'.format(t_dir, vid_fname)
    result = os.system(cmd)

if os.path.exists(vid_fname):
    print(f'Video saved as: {os.path.abspath(vid_fname)}')
else:
    print('Video creation failed. Check if ffmpeg is installed.')
    print(f'Frames are in: {t_dir}')

# Clean up temporary files only if video was created
if os.path.exists(vid_fname):
    import shutil
    shutil.rmtree(t_dir)
    print('Temporary frames cleaned up')
else:
    print('Keeping frames for debugging')

# Display final states
print("Final pendulum states (first 4):")
for i in range(min(4, n_batch)):
    cos_th, sin_th, dth = x[i]
    th = np.arctan2(sin_th.item(), cos_th.item())
    print(f"Pendulum {i}: θ={th*180/np.pi:.1f}°, ω={dth.item():.3f} rad/s")