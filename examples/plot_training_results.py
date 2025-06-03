import matplotlib.pyplot as plt
import json
import pickle
import numpy as np 
import os

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']

# Global font size defaults (you can modify these)
plt.rcParams['font.size'] = 14          # Default font size
plt.rcParams['axes.titlesize'] = 16     # Title font size  
plt.rcParams['axes.labelsize'] = 14     # Axis label font size
plt.rcParams['xtick.labelsize'] = 14    # X-tick font size
plt.rcParams['ytick.labelsize'] = 14    # Y-tick font size
plt.rcParams['legend.fontsize'] = 14    # Legend font size

def plot_training_logs(log_path, 
                      font_size_labels=25, 
                      font_size_title=25, 
                      font_size_legend=18,
                      font_size_ticks=18,
                      figsize_loss=(8, 6),
                      figsize_params=(8, 12),
                      save_plots=True,
                      save_format='pdf',
                      dpi=300):
    """
    Plot training logs with LaTeX formatting and customizable font sizes.
    
    Args:
        log_path: Path to training logs
        font_size_labels: Font size for axis labels
        font_size_title: Font size for titles
        font_size_legend: Font size for legends
        font_size_ticks: Font size for tick labels
        figsize_loss: Figure size for loss plot
        figsize_params: Figure size for parameter plots
        save_plots: Whether to save plots automatically
        save_format: Format to save plots ('pdf', 'png', 'svg', 'eps')
        dpi: DPI for saved plots
    """
    
    # Load logs
    if log_path.endswith('.json'):
        with open(log_path, 'r') as f:
            logs = json.load(f)
    else:
        with open(log_path, 'rb') as f:
            logs = pickle.load(f)
    
    # Get directory where training logs are stored
    log_directory = os.path.dirname(os.path.abspath(log_path))
    
    # Figure 1: Loss evolution
    fig1, ax1 = plt.subplots(figsize=figsize_loss)
    
    ax1.plot(logs['losses'], 'b-', linewidth=2, alpha=0.8)
    ax1.set_xlabel(r'Episode', fontsize=font_size_labels)
    ax1.set_ylabel(r'Loss', fontsize=font_size_labels)
    ax1.set_title(r'Training Loss Evolution', fontsize=font_size_title)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=font_size_ticks)
    
    # Set axis limits to use full range
    ax1.set_xlim(0, len(logs['losses'])-1)
    
    # Add some statistics as text
    final_loss = logs['losses'][-1]
    min_loss = min(logs['losses'])
    ax1.text(0.98, 0.95, f'Final Loss: {final_loss:.4f}\\\\Min Loss: {min_loss:.4f}', 
             transform=ax1.transAxes, fontsize=font_size_legend,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save loss plot
    if save_plots:
        loss_filename = f'training_loss_evolution.{save_format}'
        loss_path = os.path.join(log_directory, loss_filename)
        fig1.savefig(loss_path, bbox_inches='tight', dpi=dpi, 
                     facecolor='white', edgecolor='none')
        print(f"Loss plot saved: {loss_path}")
    
    # Figure 2: Parameter evolution (3 vertical subplots)
    fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize_params)
    
    param_history = np.array(logs['param_history'])
    true_params = logs['true_params']
    initial_guess = logs['initial_guess']
    
    param_names = ['Gravity $g$', 'Mass $m$', 'Length $l$']
    param_units = [r'$\mathrm{m/s^2}$', r'$\mathrm{kg}$', r'$\mathrm{m}$']
    colors = ['tab:red', 'tab:green', 'tab:blue']
    axes = [ax1, ax2, ax3]
    
    for i, (ax, param_name, unit, color) in enumerate(zip(axes, param_names, param_units, colors)):
        
        # Plot learned parameter evolution
        ax.plot(param_history[:, i], color=color, linewidth=2.5, alpha=0.9, 
                label=f'{param_name} (learned)')
        
        # Plot true parameter as horizontal line
        ax.axhline(y=true_params[i], color=color, linestyle='--', linewidth=2, 
                   alpha=0.8, label=f'{param_name} (true)')
        
        # Plot initial guess as a point
        ax.scatter(0, initial_guess[i], color=color, marker='o', s=100, 
                   alpha=0.7, label=f'{param_name} (initial)', zorder=5)
        
        # Formatting
        ax.set_ylabel(f'{param_name} {unit}', fontsize=font_size_labels)
        ax.set_title(f'{param_name} Evolution', fontsize=font_size_title)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=font_size_ticks)

        # Only show x-axis label and ticks for bottom subplot
        if i == 2:  # Bottom subplot (Length)
            ax.set_xlabel(r'Episode', fontsize=font_size_labels)
        else:  # Top and middle subplots
            ax.tick_params(axis='x', which='both', labelbottom=False)
        
        # Set axis limits to use full range
        ax.set_xlim(0-1, len(param_history)-1)
        
        # Position legends differently for each subplot
        legend_positions = ['lower left', 'upper left', 'lower left']
        ax.legend(fontsize=font_size_legend, loc=legend_positions[i])
        
        # Add comprehensive parameter info in top-right
        final_param = param_history[-1, i]
        initial_param = param_history[0, i]
        true_param = true_params[i]
        final_error = abs(final_param - true_param)
        initial_error = abs(initial_param - true_param)
        error_percent = (final_error / true_param) * 100
        initial_error_percent = (initial_error / true_param) * 100
        
        info_text = (f'True: {true_param:.3f}\\\\' +
                     f'Initial: {initial_param:.3f} ({initial_error_percent:.1f}\\% err)\\\\' +
                     f'Final: {final_param:.3f} ({error_percent:.1f}\\% err)')
        
        ax.text(0.98, 0.98, info_text,
                transform=ax.transAxes, fontsize=font_size_legend-1,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save parameter evolution plot
    if save_plots:
        params_filename = f'parameter_evolution.{save_format}'
        params_path = os.path.join(log_directory, params_filename)
        fig2.savefig(params_path, bbox_inches='tight', dpi=dpi,
                     facecolor='white', edgecolor='none')
        print(f"Parameter plot saved: {params_path}")
    
    # Save both plots as combined figure if requested
    if save_plots:
        # Create a summary figure with both plots side by side
        fig_combined, ((ax_loss), (ax_params)) = plt.subplots(2, 1, figsize=(12, 16))
        
        # Recreate loss plot in combined figure
        ax_loss.plot(logs['losses'], 'b-', linewidth=2, alpha=0.8)
        ax_loss.set_xlabel(r'Episode', fontsize=font_size_labels)
        ax_loss.set_ylabel(r'Loss', fontsize=font_size_labels)
        ax_loss.set_title(r'Training Loss Evolution', fontsize=font_size_title)
        ax_loss.grid(True, alpha=0.3)
        ax_loss.tick_params(axis='both', which='major', labelsize=font_size_ticks)
        ax_loss.set_xlim(0, len(logs['losses'])-1)
        ax_loss.text(0.98, 0.95, f'Final Loss: {final_loss:.4f}\\\\Min Loss: {min_loss:.4f}', 
                     transform=ax_loss.transAxes, fontsize=font_size_legend,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Create mini parameter evolution plot
        for i, (param_name, color) in enumerate(zip(param_names, colors)):
            ax_params.plot(param_history[:, i], color=color, linewidth=2.5, alpha=0.9, 
                          label=f'{param_name} (learned)')
            ax_params.axhline(y=true_params[i], color=color, linestyle='--', linewidth=2, 
                             alpha=0.8)
        
        ax_params.set_xlabel(r'Episode', fontsize=font_size_labels)
        ax_params.set_ylabel(r'Parameter Values', fontsize=font_size_labels)
        ax_params.set_title(r'Parameter Evolution Overview', fontsize=font_size_title)
        ax_params.grid(True, alpha=0.3)
        ax_params.tick_params(axis='both', which='major', labelsize=font_size_ticks)
        ax_params.legend(fontsize=font_size_legend)
        ax_params.set_xlim(0-1, len(param_history)-1)
        
        plt.tight_layout()
        
        combined_filename = f'training_summary.{save_format}'
        combined_path = os.path.join(log_directory, combined_filename)
        fig_combined.savefig(combined_path, bbox_inches='tight', dpi=dpi,
                            facecolor='white', edgecolor='none')
        print(f"Combined plot saved: {combined_path}")
        plt.close(fig_combined)
    
    return fig1, fig2

def plot_training_logs_simple(log_path, save_plots=True, save_format='pdf'):
    """Simple version with default settings."""
    return plot_training_logs(log_path, save_plots=save_plots, save_format=save_format)

def plot_training_logs_paper(log_path, save_plots=True, save_format='pdf'):
    """Version optimized for papers/publications."""
    return plot_training_logs(
        log_path,
        font_size_labels=18,
        font_size_title=20,
        font_size_legend=16,
        font_size_ticks=16,
        figsize_loss=(10, 7),
        figsize_params=(10, 15),
        save_plots=save_plots,
        save_format=save_format
    )

def plot_training_logs_presentation(log_path, save_plots=True, save_format='png'):
    """Version optimized for presentations."""
    return plot_training_logs(
        log_path,
        font_size_labels=22,
        font_size_title=26,
        font_size_legend=20,
        font_size_ticks=20,
        figsize_loss=(12, 8),
        figsize_params=(12, 18),
        save_plots=save_plots,
        save_format=save_format,
        dpi=150  # Lower DPI for presentations
    )

def plot_training_logs_custom_fonts(log_path, base_font_size=14, save_plots=True, save_format='pdf'):
    """Quick way to scale all fonts proportionally."""
    return plot_training_logs(
        log_path,
        font_size_labels=base_font_size + 2,
        font_size_title=base_font_size + 4, 
        font_size_legend=base_font_size,
        font_size_ticks=base_font_size - 2,
        save_plots=save_plots,
        save_format=save_format
    )

def plot_and_save_multiple_formats(log_path, formats=['pdf', 'png']):
    """Save plots in multiple formats."""
    results = []
    for fmt in formats:
        print(f"\nGenerating {fmt.upper()} plots...")
        fig1, fig2 = plot_training_logs_simple(log_path, save_plots=True, save_format=fmt)
        results.append((fig1, fig2, fmt))
        plt.close(fig1)
        plt.close(fig2)
    return results

def auto_plot_from_experiment_dir(experiment_dir, save_plots=True, save_format='pdf'):
    """
    Automatically find and plot training logs from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory containing training logs
        save_plots: Whether to save plots
        save_format: Format to save plots in
    """
    # Look for training logs
    json_path = os.path.join(experiment_dir, 'training_logs.json')
    pkl_path = os.path.join(experiment_dir, 'training_logs.pkl')
    
    if os.path.exists(json_path):
        log_path = json_path
    elif os.path.exists(pkl_path):
        log_path = pkl_path
    else:
        raise FileNotFoundError(f"No training logs found in {experiment_dir}")
    
    print(f"Found training logs: {log_path}")
    return plot_training_logs_simple(log_path, save_plots=save_plots, save_format=save_format)

if __name__ == '__main__':
    # Example usage with automatic saving
    
    # Method 1: Direct path to training logs (saves in same directory)
    log_path = 'pendulum_experiments/learn_params/20250603_095123/training_logs.json'
    if os.path.exists(log_path):
        print("Generating plots with automatic saving...")
        fig1, fig2 = plot_training_logs_simple(log_path, save_plots=True, save_format='pdf')
        plt.show()
    else:
        print(f"Log file not found: {log_path}")
    
    # Method 2: From experiment directory (automatically finds logs)
    # experiment_dir = 'pendulum_experiments/learn_params/20250603_095123'
    # if os.path.exists(experiment_dir):
    #     print("\nGenerating plots from experiment directory...")
    #     fig1, fig2 = auto_plot_from_experiment_dir(experiment_dir, save_plots=True, save_format='pdf')
    #     plt.show()
    
    # Method 3: Multiple formats
    # if os.path.exists(log_path):
    #     print("\nGenerating plots in multiple formats...")
    #     results = plot_and_save_multiple_formats(log_path, formats=['pdf', 'png', 'svg'])
    
    # Method 4: Different presets with saving
    # if os.path.exists(log_path):
    #     print("\nGenerating paper-quality plots...")
    #     fig1, fig2 = plot_training_logs_paper(log_path, save_plots=True, save_format='pdf')
    #     plt.close(fig1)
    #     plt.close(fig2)
    #     
    #     print("Generating presentation plots...")
    #     fig1, fig2 = plot_training_logs_presentation(log_path, save_plots=True, save_format='png')
    #     plt.close(fig1)
    #     plt.close(fig2)
    
    print("\nPlots generated successfully!")
    print("\nNew features:")
    print("- Plots are automatically saved in the same directory as training logs")
    print("- Use save_plots=True/False to control saving")
    print("- Use save_format='pdf'/'png'/'svg'/'eps' to choose format")
    print("- Use plot_and_save_multiple_formats() to save in multiple formats")
    print("- Use auto_plot_from_experiment_dir() to automatically find logs")
    print("\nSaved files:")
    print("- training_loss_evolution.{format}")
    print("- parameter_evolution.{format}")
    print("- training_summary.{format} (combined overview)")