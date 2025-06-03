import matplotlib.pyplot as plt
import json
import pickle
import numpy as np 

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
                      figsize_params=(8, 12)):
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
    """
    
    # Load logs
    if log_path.endswith('.json'):
        with open(log_path, 'r') as f:
            logs = json.load(f)
    else:
        with open(log_path, 'rb') as f:
            logs = pickle.load(f)
    
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
    
    return fig1, fig2

def plot_training_logs_simple(log_path):
    """Simple version with default settings."""
    return plot_training_logs(log_path)

def plot_training_logs_paper(log_path):
    """Version optimized for papers/publications."""
    return plot_training_logs(
        log_path,
        font_size_labels=18,
        font_size_title=20,
        font_size_legend=16,
        font_size_ticks=16,
        figsize_loss=(10, 7),
        figsize_params=(10, 15)
    )

def plot_training_logs_presentation(log_path):
    """Version optimized for presentations."""
    return plot_training_logs(
        log_path,
        font_size_labels=22,
        font_size_title=26,
        font_size_legend=20,
        font_size_ticks=20,
        figsize_loss=(12, 8),
        figsize_params=(12, 18)
    )

def plot_training_logs_custom_fonts(log_path, base_font_size=14):
    """Quick way to scale all fonts proportionally."""
    return plot_training_logs(
        log_path,
        font_size_labels=base_font_size + 2,
        font_size_title=base_font_size + 4, 
        font_size_legend=base_font_size,
        font_size_ticks=base_font_size - 2
    )

if __name__ == '__main__':
    # You can use any of these versions:
    
    # Default version
    fig1, fig2 = plot_training_logs('pendulum_experiments/learn_params/20250603_095123/training_logs.json')
    
    # Paper version (larger fonts)
    # fig1, fig2 = plot_training_logs_paper('pendulum_experiments/learn_params/20250603_032417/training_logs.json')
    
    # Presentation version (even larger fonts)
    # fig1, fig2 = plot_training_logs_presentation('pendulum_experiments/learn_params/20250603_032417/training_logs.json')
    
    # Custom font scaling
    # fig1, fig2 = plot_training_logs_custom_fonts('pendulum_experiments/learn_params/20250603_032417/training_logs.json', base_font_size=16)
    
    # Full custom version
    # fig1, fig2 = plot_training_logs(
    #     'pendulum_experiments/learn_params/20250603_032417/training_logs.json',
    #     font_size_labels=20,
    #     font_size_title=24,
    #     font_size_legend=18,
    #     font_size_ticks=18,
    #     figsize_loss=(10, 8),
    #     figsize_params=(10, 16)
    # )
    
    plt.show()
    
    # Optional: Save figures as high-quality PDFs
    # fig1.savefig('loss_evolution.pdf', bbox_inches='tight', dpi=300)
    # fig2.savefig('parameter_evolution.pdf', bbox_inches='tight', dpi=300)
    
    print("Plots generated successfully!")
    print("\nTo adjust font sizes:")
    print("- Modify the function parameters (font_size_labels, font_size_title, etc.)")
    print("- Use plot_training_logs_custom_fonts() for proportional scaling")
    print("- Use plot_training_logs_paper() or plot_training_logs_presentation() for presets")
    print("\nTo save figures:")
    print("- Uncomment the fig.savefig() lines at the bottom")