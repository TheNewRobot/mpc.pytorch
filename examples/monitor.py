"""
Monitoring Module for MPC Controllers
=====================================

This module provides plotting capabilities for tracking variables in MPC and other control applications.
It creates real-time plots that update as the simulation runs, allowing for monitoring of 
state variables, control inputs, rewards, and other metrics.

Features:
- Real-time plotting of multiple variables
- Configurable plot appearance and update frequency
- Minimal performance impact when monitoring is disabled
- Thread-safe implementation to avoid blocking the main simulation

Usage:
    from monitor import Monitor
    
    # Create a monitor instance (can be disabled via flag)
    monitor = Monitor(enabled=args.plot)
    
    # In simulation loop:
    monitor.update(theta=state[0], theta_dot=state[1], control=action, reward=reward)
    
    # When finished:
    monitor.save_plots("experiment_results")  # Optional
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import time
from collections import defaultdict
import threading
from typing import Dict, List, Optional, Union, Any


class Monitor:
    """Monitor class for tracking and plotting variables during simulations."""
    
    def __init__(self, enabled: bool = True, update_freq: int = 5, max_points: int = 1000,
                 figsize: tuple = (12, 8), style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize the monitoring system.
        """
        self.enabled = enabled
        self.update_freq = update_freq
        self.max_points = max_points
        self.figsize = figsize
        self.style = style
        
        # Data storage
        self.data = defaultdict(list)
        self.time_steps = []
        self.current_step = 0
        
        # Plot setup
        self.fig = None
        self.axes = None
        self.lines = {}
        self.lock = threading.Lock()
        
        if self.enabled:
            self._initialize_plot()
    
    def _initialize_plot(self):
        """Set up the plotting environment."""
        with plt.style.context(self.style):
            plt.ion()  # Interactive mode
            # Create an empty figure - we'll add subplots later
            # when we have data to display
            self.fig = plt.figure(figsize=self.figsize)
            self.axes = None
            plt.show(block=False)
    
    def update(self, **kwargs):
        """
        Update the monitor with new values.
        """
        if not self.enabled:
            return
        
        # Store data
        self.current_step += 1
        self.time_steps.append(self.current_step)
        
        for name, value in kwargs.items():
            # Convert to scalar if needed
            if hasattr(value, 'item'):
                value = value.item()
            self.data[name].append(value)
        
        # Trim data if needed
        if len(self.time_steps) > self.max_points:
            self.time_steps = self.time_steps[-self.max_points:]
            for name in self.data:
                self.data[name] = self.data[name][-self.max_points:]
        
        # Update plots every update_freq steps
        if self.current_step % self.update_freq == 0:
            self._update_plot()
    
    def _update_plot(self):
        """Update the plot with current data."""
        with self.lock:
            # Skip if no data available yet
            if not self.data or not self.time_steps:
                return
                
            # Check if we need to create/recreate the plot (e.g., if variables changed)
            if self.axes is None or len(self.axes) != len(self.data):
                # Clear the existing figure
                plt.figure(self.fig.number)
                plt.clf()
                
                # Create new subplots
                self.axes = []
                self.lines = {}
                
                # Create a subplot for each variable
                num_vars = len(self.data)
                for i, (name, values) in enumerate(self.data.items()):
                    if i == 0:
                        ax = plt.subplot(num_vars, 1, i+1)
                    else:
                        ax = plt.subplot(num_vars, 1, i+1, sharex=self.axes[0])
                    
                    line, = ax.plot(self.time_steps, values, label=name)
                    ax.set_ylabel(name)
                    ax.grid(True)
                    
                    self.axes.append(ax)
                    self.lines[name] = line
                
                # Add x-label to the bottom subplot
                self.axes[-1].set_xlabel('Time Steps')
                
                # Adjust layout
                plt.tight_layout()
            else:
                # Update existing lines
                for name, values in self.data.items():
                    if name in self.lines:
                        self.lines[name].set_data(self.time_steps, values)
                
                # Adjust axes limits
                for ax in self.axes:
                    ax.relim()
                    ax.autoscale_view()
            
            # Refresh the plot
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
    
    def save_plots(self, directory: str = "plots", filename_prefix: str = ""):
        if not self.enabled or self.fig is None or not self.data:
            return
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        experiment_path = os.path.join(directory, timestamp)
        os.makedirs(experiment_path, exist_ok=True)
        
        # Save combined plot if it exists
        if hasattr(self, 'axes') and self.axes:
            filename = f"{filename_prefix}_combined_{timestamp}.png" if filename_prefix else f"combined_{timestamp}.png"
            self.fig.savefig(os.path.join(experiment_path, filename), dpi=300, bbox_inches='tight')
    
    
    def close(self):
        """Close all plots and clean up resources."""
        if self.enabled and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None
            self.lines = {}