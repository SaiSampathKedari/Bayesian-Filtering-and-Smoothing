"""
ENHANCED Bayesian Linear Regression Animation
Based on the original red color scheme with professional enhancements.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from matplotlib import patheffects

from regression.linear_regression import *

# Professional styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']


def confidence_ellipse(mean, cov, ax, n_std=2.0, facecolor='none', **kwargs):
    """Create a confidence ellipse for a 2D Gaussian distribution."""
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = (plt.matplotlib.transforms.Affine2D()
              .scale(scale_x, scale_y)
              .translate(mean[0], mean[1]))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def create_bayesian_regression_animation_enhanced(
    dataset,
    means,
    mu_batch,
    t_plot,
    noise_std,
    output_path='bayesian_regression_enhanced.mp4',
    fps=8,
    dpi=150,
    show_parameter_space=True
):
    """
    Enhanced animation with ORIGINAL red color scheme + professional polish.
    
    Enhancements over original:
    - Three-panel layout with parameter space
    - Confidence ellipses showing uncertainty
    - Current observation highlighting
    - Dynamic statistics
    - Smoother animation (8 fps)
    - Better visual polish
    
    But keeps the ORIGINAL color palette:
    - Red for posterior
    - Deep blue for true model
    - Teal for used data
    - Orange/Green for parameters
    
    Parameters
    ----------
    dataset : RegressionDataSet
        Dataset object
    means : np.ndarray, shape (num_data+1, 2)
        Parameter means at each step
    mu_batch : np.ndarray, shape (2,)
        Batch posterior mean
    t_plot : np.ndarray
        Plotting grid
    noise_std : float
        Observation noise std
    output_path : str
        Output file path
    fps : int
        Frames per second
    dpi : int
        Resolution
    show_parameter_space : bool
        Whether to show 3rd panel with parameter space
    """
    
    # ========================================================================
    # PRECOMPUTE STATISTICS
    # ========================================================================
    
    num_frames = means.shape[0]
    num_data = num_frames - 1
    
    H_plot = np.column_stack([np.ones_like(t_plot), t_plot])
    
    all_latent_means = np.zeros((num_frames, len(t_plot)))
    all_latent_stds = np.zeros((num_frames, len(t_plot)))
    all_covariances = np.zeros((num_frames, 2, 2))
    
    mu_0 = means[0]
    C_0 = np.eye(2)
    
    updated_mean = mu_0.copy()
    updated_covariance = C_0.copy()
    
    latent_mean = H_plot @ updated_mean
    latent_cov = H_plot @ updated_covariance @ H_plot.T
    latent_std = np.sqrt(np.diag(latent_cov))
    all_latent_means[0] = latent_mean
    all_latent_stds[0] = latent_std
    all_covariances[0] = updated_covariance
    
    for ii in range(1, num_frames):
        y_k = np.array([dataset.y[ii - 1]])
        H_k = np.array([[1.0, dataset.t[ii - 1]]])
        
        updated_mean, updated_covariance = batch_linear_gaussian_update(
            y_k, H_k, updated_mean, updated_covariance, noise_std
        )
        
        latent_mean = H_plot @ updated_mean
        latent_cov = H_plot @ updated_covariance @ H_plot.T
        latent_std = np.sqrt(np.diag(latent_cov))
        all_latent_means[ii] = latent_mean
        all_latent_stds[ii] = latent_std
        all_covariances[ii] = updated_covariance
    
    # Uncertainty metrics
    det_prior = np.linalg.det(all_covariances[0])
    det_frames = np.array([np.linalg.det(all_covariances[i]) for i in range(num_frames)])
    uncertainty_reduction = 100 * (1 - det_frames / det_prior)
    
    print(f"Precomputed statistics for {num_frames} frames")
    
    # ========================================================================
    # ORIGINAL COLOR SCHEME (Enhanced)
    # ========================================================================
    
    COLOR_POSTERIOR = '#E63946'          # Original vibrant red
    COLOR_POSTERIOR_LIGHT = '#F77F7F'    # Lighter red
    COLOR_TRUE = '#1D3557'               # Original deep blue
    COLOR_DATA_USED = '#2A9D8F'          # Original teal
    COLOR_DATA_CURRENT = '#FFB703'       # Golden yellow for current
    COLOR_DATA_UNUSED = '#8D99AE'        # Original gray
    COLOR_THETA1 = '#F77F00'             # Original orange
    COLOR_THETA2 = '#06A77D'             # Original green
    COLOR_BATCH = '#264653'              # Original dark teal
    
    # ========================================================================
    # FIGURE SETUP
    # ========================================================================
    
    if show_parameter_space:
        fig = plt.figure(figsize=(20, 11), dpi=dpi, facecolor='white')
        gs = GridSpec(2, 2, height_ratios=[1.3, 1], width_ratios=[1.3, 1],
                      hspace=0.30, wspace=0.30,
                      left=0.06, right=0.96, top=0.94, bottom=0.06)
        
        ax_reg = fig.add_subplot(gs[0, :])      # Top: full width
        ax_params = fig.add_subplot(gs[1, 0])   # Bottom left
        ax_space = fig.add_subplot(gs[1, 1])    # Bottom right
    else:
        fig = plt.figure(figsize=(18, 10), dpi=dpi, facecolor='white')
        gs = GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.3,
                      left=0.08, right=0.95, top=0.94, bottom=0.08)
        ax_reg = fig.add_subplot(gs[0])
        ax_params = fig.add_subplot(gs[1])
    
    # ========================================================================
    # PANEL 1: REGRESSION PLOT
    # ========================================================================
    
    line_posterior, = ax_reg.plot([], [], color=COLOR_POSTERIOR, linewidth=4,
                                   label='Posterior mean', zorder=10)
    
    line_true, = ax_reg.plot(dataset.t_plot, dataset.y_true,
                              color=COLOR_TRUE, linewidth=3,
                              label='True model', zorder=8, linestyle='--',
                              alpha=0.85)
    
    scatter_unused = ax_reg.scatter([], [], color=COLOR_DATA_UNUSED, s=60,
                                    alpha=0.35, label='Future data', zorder=5,
                                    edgecolors='none')
    scatter_used = ax_reg.scatter([], [], color=COLOR_DATA_USED, s=140,
                                  edgecolors='white', linewidths=2,
                                  label='Observed data', zorder=9, marker='o')
    scatter_current = ax_reg.scatter([], [], color=COLOR_DATA_CURRENT, s=350,
                                     edgecolors=COLOR_DATA_CURRENT, linewidths=4,
                                     label='Current observation', zorder=11,
                                     marker='*', alpha=0.95)
    
    # Text boxes - adjusted positions to avoid overlaps
    text_n_obs = ax_reg.text(0.02, 0.97, '', transform=ax_reg.transAxes,
                             fontsize=14, verticalalignment='top', fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.7', facecolor='white',
                                      alpha=0.95, edgecolor=COLOR_POSTERIOR, linewidth=2.5),
                             zorder=15)
    
    text_stats = ax_reg.text(0.98, 0.97, '', transform=ax_reg.transAxes,
                             fontsize=13, verticalalignment='top',
                             horizontalalignment='right',
                             bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                                      alpha=0.95, edgecolor=COLOR_THETA1, linewidth=2.5),
                             zorder=15, family='monospace')
    
    # Styling
    ax_reg.set_xlabel(r'Time $t$', fontsize=19, fontweight='bold', labelpad=12)
    ax_reg.set_ylabel(r'Output $y(t)$', fontsize=19, fontweight='bold', labelpad=12)
    ax_reg.set_title('Recursive Bayesian Linear Regression: Posterior Updates',
                     fontsize=21, fontweight='bold', pad=25, color=COLOR_TRUE)
    
    y_range = dataset.y_true.max() - dataset.y_true.min()
    ax_reg.set_xlim(dataset.t_plot.min() - 0.1, dataset.t_plot.max() + 0.1)
    ax_reg.set_ylim(dataset.y_true.min() - 0.35 * y_range,
                    dataset.y_true.max() + 0.35 * y_range)
    
    ax_reg.grid(True, alpha=0.25, linestyle='--', linewidth=1)
    ax_reg.set_facecolor('#FAFAFA')
    ax_reg.spines['top'].set_visible(False)
    ax_reg.spines['right'].set_visible(False)
    ax_reg.spines['left'].set_linewidth(2)
    ax_reg.spines['bottom'].set_linewidth(2)
    
    # Legend positioned to avoid text box overlap
    legend1 = ax_reg.legend(loc='lower left', fontsize=12, framealpha=0.95,
                           edgecolor=COLOR_POSTERIOR, fancybox=True, shadow=True,
                           frameon=True, facecolor='white')
    legend1.get_frame().set_linewidth(2)
    
    # ========================================================================
    # PANEL 2: PARAMETER CONVERGENCE
    # ========================================================================
    
    line_theta1, = ax_params.plot([], [], color=COLOR_THETA1, linewidth=4,
                                  label=r'$\theta_1$ (intercept)', marker='o',
                                  markersize=6, markevery=[-1])
    line_theta2, = ax_params.plot([], [], color=COLOR_THETA2, linewidth=4,
                                  label=r'$\theta_2$ (slope)', marker='s',
                                  markersize=6, markevery=[-1])
    
    ax_params.axhline(y=mu_batch[0], color=COLOR_THETA1,
                      linestyle=':', linewidth=3, alpha=0.65, zorder=1,
                      label=r'$\theta_1$ (batch)')
    ax_params.axhline(y=mu_batch[1], color=COLOR_THETA2,
                      linestyle=':', linewidth=3, alpha=0.65, zorder=1,
                      label=r'$\theta_2$ (batch)')
    
    ax_params.axhspan(mu_batch[0] - 0.05, mu_batch[0] + 0.05,
                      color=COLOR_THETA1, alpha=0.1, zorder=0)
    ax_params.axhspan(mu_batch[1] - 0.05, mu_batch[1] + 0.05,
                      color=COLOR_THETA2, alpha=0.1, zorder=0)
    
    text_progress = ax_params.text(0.02, 0.97, '', transform=ax_params.transAxes,
                                  fontsize=13, verticalalignment='top', fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                                           alpha=0.95, edgecolor=COLOR_THETA2, linewidth=2.5),
                                  zorder=15)
    
    ax_params.set_xlabel('Number of Observations', fontsize=19, fontweight='bold', labelpad=12)
    ax_params.set_ylabel('Parameter Value', fontsize=19, fontweight='bold', labelpad=12)
    ax_params.set_title('Parameter Convergence: Recursive → Batch Solution',
                        fontsize=21, fontweight='bold', pad=20, color=COLOR_TRUE)
    ax_params.set_xlim(-1, num_data + 1)
    
    all_vals = np.concatenate([means[:, 0], means[:, 1], [mu_batch[0], mu_batch[1]]])
    param_range = all_vals.max() - all_vals.min()
    ax_params.set_ylim(all_vals.min() - 0.25 * param_range,
                       all_vals.max() + 0.25 * param_range)
    
    ax_params.grid(True, alpha=0.25, linestyle='--', linewidth=1)
    ax_params.set_facecolor('#FAFAFA')
    ax_params.spines['top'].set_visible(False)
    ax_params.spines['right'].set_visible(False)
    ax_params.spines['left'].set_linewidth(2)
    ax_params.spines['bottom'].set_linewidth(2)
    
    legend2 = ax_params.legend(loc='right', fontsize=13, framealpha=0.95,
                              edgecolor=COLOR_THETA1, fancybox=True, shadow=True,
                              ncol=2, frameon=True, facecolor='white')
    legend2.get_frame().set_linewidth(2)
    
    # ========================================================================
    # PANEL 3: PARAMETER SPACE (with red color scheme)
    # ========================================================================
    
    if show_parameter_space:
        # True parameters
        true_theta = np.array([dataset.y_true[0],
                              (dataset.y_true[-1] - dataset.y_true[0]) /
                              (dataset.t_plot[-1] - dataset.t_plot[0])])
        
        ax_space.scatter(true_theta[0], true_theta[1],
                        s=500, marker='*', color=COLOR_TRUE,
                        edgecolors='white', linewidths=3,
                        label='True parameters', zorder=20)
        
        ax_space.scatter(mu_batch[0], mu_batch[1],
                        s=250, marker='X', color=COLOR_BATCH,
                        edgecolors='white', linewidths=2.5,
                        label='Batch solution', zorder=19)
        
        ellipse_2sig = None
        ellipse_1sig = None
        
        line_trajectory, = ax_space.plot([], [], 'o-', color=COLOR_POSTERIOR,
                                        markersize=5, linewidth=2, alpha=0.5,
                                        zorder=10)
        scatter_current_param = ax_space.scatter([], [], s=280, marker='o',
                                                color=COLOR_POSTERIOR,
                                                edgecolors='white', linewidths=3,
                                                label='Current estimate', zorder=18)
        
        text_uncertainty = ax_space.text(0.02, 0.97, '', transform=ax_space.transAxes,
                                        fontsize=12, verticalalignment='top',
                                        bbox=dict(boxstyle='round,pad=0.6',
                                                 facecolor='white', alpha=0.95,
                                                 edgecolor=COLOR_POSTERIOR, linewidth=2.5),
                                        zorder=15, family='monospace')
        
        ax_space.set_xlabel(r'$\theta_1$ (intercept)', fontsize=17, fontweight='bold', labelpad=12)
        ax_space.set_ylabel(r'$\theta_2$ (slope)', fontsize=17, fontweight='bold', labelpad=12)
        ax_space.set_title('Parameter Space: Posterior Evolution',
                          fontsize=19, fontweight='bold', pad=20, color=COLOR_TRUE)
        
        theta1_range = means[:, 0].max() - means[:, 0].min()
        theta2_range = means[:, 1].max() - means[:, 1].min()
        margin1 = max(0.3 * theta1_range, 0.5)
        margin2 = max(0.3 * theta2_range, 0.3)
        
        ax_space.set_xlim(means[:, 0].min() - margin1, means[:, 0].max() + margin1)
        ax_space.set_ylim(means[:, 1].min() - margin2, means[:, 1].max() + margin2)
        
        ax_space.grid(True, alpha=0.25, linestyle='--', linewidth=1)
        ax_space.set_facecolor('#FAFAFA')
        ax_space.spines['top'].set_visible(False)
        ax_space.spines['right'].set_visible(False)
        ax_space.spines['left'].set_linewidth(2)
        ax_space.spines['bottom'].set_linewidth(2)
        ax_space.set_aspect('auto')
        
        legend3 = ax_space.legend(loc='upper right', fontsize=12, framealpha=0.95,
                                 edgecolor=COLOR_POSTERIOR, fancybox=True, shadow=True,
                                 frameon=True, facecolor='white')
        legend3.get_frame().set_linewidth(2)
    
    # ========================================================================
    # ANIMATION UPDATE FUNCTION
    # ========================================================================
    
    # Keep track of fill objects
    current_fills = {'fill_2sig': None, 'fill_1sig': None}
    
    def animate(frame):
        """Update function."""
        
        # Panel 1: Regression
        latent_mean = all_latent_means[frame]
        latent_std = all_latent_stds[frame]
        
        line_posterior.set_data(t_plot, latent_mean)
        
        # Remove old fills if they exist
        if current_fills['fill_2sig'] is not None:
            try:
                current_fills['fill_2sig'].remove()
            except:
                pass
        if current_fills['fill_1sig'] is not None:
            try:
                current_fills['fill_1sig'].remove()
            except:
                pass
        
        # Create NEW fills for CURRENT frame only
        current_fills['fill_2sig'] = ax_reg.fill_between(
            t_plot, latent_mean - 2 * latent_std, latent_mean + 2 * latent_std,
            color=COLOR_POSTERIOR, alpha=0.2,
            label=r'$\pm 2\sigma$ (parameter uncertainty)', zorder=3
        )
        current_fills['fill_1sig'] = ax_reg.fill_between(
            t_plot, latent_mean - latent_std, latent_mean + latent_std,
            color=COLOR_POSTERIOR, alpha=0.3, zorder=4
        )
        
        # Update data points
        if frame > 0:
            scatter_used.set_offsets(np.c_[dataset.t[:frame], dataset.y[:frame]])
            scatter_current.set_offsets(np.c_[[dataset.t[frame-1]], [dataset.y[frame-1]]])
            
            if frame < num_data:
                scatter_unused.set_offsets(np.c_[dataset.t[frame:], dataset.y[frame:]])
            else:
                scatter_unused.set_offsets(np.empty((0, 2)))
        else:
            scatter_used.set_offsets(np.empty((0, 2)))
            scatter_current.set_offsets(np.empty((0, 2)))
            scatter_unused.set_offsets(np.c_[dataset.t, dataset.y])
        
        # Update text
        if frame == 0:
            text_n_obs.set_text('Prior\nn = 0')
            text_stats.set_text('Waiting for data...')
        else:
            text_n_obs.set_text(f'Observations: n = {frame}\nProgress: {100*frame/num_data:.0f}%')
            text_stats.set_text(
                f'θ₁ = {means[frame, 0]:.4f}\n'
                f'θ₂ = {means[frame, 1]:.4f}\n'
                f'Uncertainty: ↓{uncertainty_reduction[frame]:.1f}%'
            )
        
        # Panel 2: Parameters
        steps = np.arange(frame + 1)
        line_theta1.set_data(steps, means[:frame + 1, 0])
        line_theta2.set_data(steps, means[:frame + 1, 1])
        
        if frame > 0:
            dist_to_batch = np.linalg.norm(means[frame] - mu_batch)
            text_progress.set_text(f'Converged: {dist_to_batch:.4f}')
        else:
            text_progress.set_text('Starting...')
        
        # Panel 3: Parameter space
        if show_parameter_space:
            nonlocal ellipse_2sig, ellipse_1sig
            
            if ellipse_2sig is not None:
                ellipse_2sig.remove()
            if ellipse_1sig is not None:
                ellipse_1sig.remove()
            
            cov = all_covariances[frame]
            mean = means[frame]
            
            ellipse_2sig = confidence_ellipse(mean, cov, ax_space, n_std=2.0,
                                             edgecolor=COLOR_POSTERIOR, linewidth=3,
                                             facecolor=COLOR_POSTERIOR, alpha=0.12,
                                             zorder=5)
            ellipse_1sig = confidence_ellipse(mean, cov, ax_space, n_std=1.0,
                                             edgecolor=COLOR_POSTERIOR_LIGHT, linewidth=2.5,
                                             facecolor=COLOR_POSTERIOR, alpha=0.25,
                                             zorder=6)
            
            line_trajectory.set_data(means[:frame + 1, 0], means[:frame + 1, 1])
            
            if frame > 0:
                scatter_current_param.set_offsets(np.c_[[mean[0]], [mean[1]]])
            else:
                scatter_current_param.set_offsets(np.c_[[mean[0]], [mean[1]]])
            
            det_current = np.linalg.det(cov)
            text_uncertainty.set_text(
                f'Covariance:\n'
                f'det(Σ) = {det_current:.2e}\n'
                f'Reduced: {uncertainty_reduction[frame]:.1f}%'
            )
        
        return_list = [line_posterior, current_fills['fill_2sig'], current_fills['fill_1sig'],
                      scatter_used, scatter_unused, scatter_current,
                      text_n_obs, text_stats, line_theta1, line_theta2, text_progress]
        
        if show_parameter_space:
            return_list.extend([ellipse_2sig, ellipse_1sig, line_trajectory,
                              scatter_current_param, text_uncertainty])
        
        return tuple(return_list)
    
    # ========================================================================
    # CREATE AND SAVE
    # ========================================================================
    
    print("\n" + "="*70)
    print("CREATING ENHANCED ANIMATION (Original Red Color Scheme)")
    print("="*70)
    print(f"  Frames: {num_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {num_frames / fps:.1f} seconds")
    print(f"  Resolution: {fig.get_figwidth() * dpi:.0f} x {fig.get_figheight() * dpi:.0f} px")
    print(f"  Layout: {'Three-panel' if show_parameter_space else 'Two-panel'}")
    print("="*70)
    
    anim = FuncAnimation(fig, animate, frames=num_frames,
                        interval=1000 / fps, blit=False, repeat=True)
    
    writer = FFMpegWriter(fps=fps, bitrate=5000, codec='libx264',
                         metadata={'title': 'Enhanced Bayesian Regression',
                                  'artist': 'Sai Sampath'})
    
    print(f"\nSaving to: {output_path}")
    anim.save(output_path, writer=writer, dpi=dpi)
    
    print("\n" + "="*70)
    print("✓ ENHANCED ANIMATION COMPLETE!")
    print("="*70)
    print(f"  Output: {output_path}")
    print("\nEnhancements:")
    print("  ✓ Original RED color scheme (much better!)")
    print("  ✓ Three-panel layout with parameter space")
    print("  ✓ Confidence ellipses")
    print("  ✓ Current observation highlighting")
    print("  ✓ Dynamic statistics")
    print("  ✓ 8 fps smooth animation")
    print("  ✓ Professional polish")
    print("="*70 + "\n")
    
    plt.close(fig)
    return output_path