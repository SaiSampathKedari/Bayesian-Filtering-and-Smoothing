import numpy as np
from typing import Tuple, Callable
from filters.kalman_filter import *

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

def pendulum_dynamics(
    x   :   np.ndarray,
    dt  :   float,
    g   :   float = 9.81
) -> np.ndarray:
    """
    Discrete-time nonlinear pendulum dynamics.

    State:
        x = [angle, angular_velocity]

    Dynamics:
        x1_{k+1} = x1_k + x2_k * dt
        x2_{k+1} = x2_k - g * sin(x1_k) * dt
    """
    
    x1, x2 = x
    
    x1_next = x1 + x2 * dt
    x2_next = x2 - g * np.sin(x1) * dt
    
    return np.array([x1_next, x2_next])



def pendulum_jacobian_dynamics(
    x   :   np.ndarray,
    dt  :   float,
    g   :   float=9.81
) -> np.ndarray:
    """
    Jacobian of pendulum dynamics with respect to the state.

    A_k = d Phi(x) / d x evaluated at x
    """
    
    x1 = x[0]
    
    A_k = np.array([
        [1                      ,   dt], 
        [-g * np.cos(x1) * dt   ,   1 ]])
    
    return A_k



def pendulum_measurement(
    x   :   np.ndarray
) -> np.ndarray:
    """
    Measurement model.

    Only the angle is observed through a nonlinear sensor:
        y = sin(x1)
    """
    
    return np.array([np.sin(x[0])])



def pendulum_jacobian_measurement(
    x   :   np.ndarray
) -> np.ndarray:
    """
    Jacobian of measurement model with respect to the state.

    H_k = d h(x) / d x evaluated at x
    """
    x1 = x[0]
    
    return np.array([[np.cos(x1), 0.0]])



def pendulum_process_noise_cov(
    dt  :   float,
    qc  :   float
) -> np.ndarray:
    """
    Discrete-time process noise covariance for the pendulum.

    Derived from continuous-time white acceleration noise:

        Q = [[ qc * dt^3 / 3,  qc * dt^2 / 2 ],
             [ qc * dt^2 / 2,  qc * dt       ]]
    """
    
    return np.array([
        [qc * dt**3 / 3.0, qc * dt**2 / 2.0],
        [qc * dt**2 / 2.0, qc * dt]
    ])
    

@dataclass
class TrueNonlinearSystem:
    """
    Ground-truth nonlinear system used ONLY for data generation.

    Dynamics are deterministic.
    Measurement noise is added.
    """

    Phi: Callable[[np.ndarray], np.ndarray]
    """True nonlinear state transition: x_{k+1} = Phi(x_k)"""

    h: Callable[[np.ndarray], np.ndarray]
    """True nonlinear measurement model: y_k = h(x_k)"""

    measurement_noise_cov: np.ndarray
    """Sensor noise covariance (R) used ONLY for data generation"""

def make_true_pendulum_system(
    dt: float,
    R: np.ndarray,
    g: float = 9.81
) -> TrueNonlinearSystem:
    """
    Ground-truth pendulum system for data generation.
    """

    return TrueNonlinearSystem(
        Phi=lambda x: pendulum_dynamics(x, dt, g),
        h=pendulum_measurement,
        measurement_noise_cov=R
    )

    
def propagate_true_nonlinear_dynamics(
    x_k: np.ndarray,
    system: TrueNonlinearSystem
) -> np.ndarray:
    """
    Deterministic nonlinear ground-truth propagation.
    """

    return system.Phi(x_k)

def observe_nonlinear(
    x_k: np.ndarray,
    system: TrueNonlinearSystem,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate noisy nonlinear measurement.
    """

    v_k = rng.multivariate_normal(
        mean=np.zeros(system.measurement_noise_cov.shape[0]),
        cov=system.measurement_noise_cov
    )

    return system.h(x_k) + v_k

def generate_nonlinear_data(
    times: np.ndarray,
    x0: np.ndarray,
    obs_ind: np.ndarray,
    system: TrueNonlinearSystem,
    rng: np.random.Generator
) -> tuple[Observations, Observations]:
    """
    Generate ground-truth nonlinear states and noisy measurements.
    """

    N = len(times)
    n = x0.shape[0]
    m = system.measurement_noise_cov.shape[0]

    # Allocate storage
    x_true = np.zeros((N, n))
    y_obs = np.zeros((len(obs_ind), m))

    # Initial condition
    x_true[0] = x0
    
    obs_counter = 0

    # Simulate dynamics
    for k in range(1, N):
        x_true[k] = propagate_true_nonlinear_dynamics(x_true[k-1], system)

        if obs_counter < len(obs_ind) and k == obs_ind[obs_counter]:
            y_obs[obs_counter] = observe_nonlinear(x_true[k], system, rng)
            obs_counter += 1

    truth_states = Observations(
        times=times,
        obs_ind=np.arange(N),
        obs=x_true,
        names=["angle", "angular_velocity"]
    )

    measurements = Observations(
        times=times,
        obs_ind=obs_ind,
        obs=y_obs,
        names=["sin(angle)"]
    )

    return truth_states, measurements


def pendulum_continuous_dynamics(t, x, g=9.81, l=1.0):
    """
    Continuous-time nonlinear pendulum dynamics.

    State:
        x = [theta, theta_dot]

    Equations:
        dtheta/dt     = theta_dot
        dtheta_dot/dt = -(g/l) * sin(theta)
    """
    theta, theta_dot = x
    return np.array([
        theta_dot,
        -(g / l) * np.sin(theta)
    ])

def make_pendulum_phase_portrait(
    theta_min=-1.2*np.pi,
    theta_max= 1.2*np.pi,
    omega_min=-6.0,
    omega_max= 6.0,
    g=9.81,
    l=1.0,
    grid_size=220
):
    """
    Continuous-time pendulum phase portrait.
    PURELY for visualization.
    """

    theta = np.linspace(theta_min, theta_max, grid_size)
    omega = np.linspace(omega_min, omega_max, grid_size)

    Theta, Omega = np.meshgrid(theta, omega)

    U = Omega
    V = -(g / l) * np.sin(Theta)

    speed = np.sqrt(U**2 + V**2)
    U /= (speed + 1e-8)
    V /= (speed + 1e-8)

    return theta, omega, U, V

  

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
import os

def animate_truth_vs_filter(
    truth,
    measurements,
    kf_history,
    make_phase_portrait_fn,
    save_path: str,
    filter_name: str = "EKF",
    title: str = "True vs Filter",
    slow_factor: float = 4.0,
    output_format: str = "gif",   # "gif" or "mp4"
):


    # =================================================
    # Extract data
    # =================================================
    t = truth.times
    dt = t[1] - t[0]
    T = t[-1]
    N = len(t)

    θ_true = truth.obs[:, 0]
    ω_true = truth.obs[:, 1]

    θ_est = kf_history.means[:, 0]
    ω_est = kf_history.means[:, 1]

    θ_std = kf_history.stds[:, 0]
    ω_std = kf_history.stds[:, 1]

    obs_idx = measurements.obs_ind
    y_meas = measurements.obs.squeeze()

    # =================================================
    # Phase portrait (continuous, visual only)
    # =================================================
    θ_grid, ω_grid, U, V = make_phase_portrait_fn()

    # =================================================
    # Figure & layout
    # =================================================
    fig = plt.figure(
        figsize=(16, 9),
        facecolor="black",
        constrained_layout=True
    )
    fig.suptitle(f"{title} ({filter_name})", color="white", fontsize=16)

    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[1.15, 1.0],
        height_ratios=[2.2, 2.2, 1.6]
    )

    ax_phase = fig.add_subplot(gs[:, 0])
    ax_θ     = fig.add_subplot(gs[0, 1])
    ax_ω     = fig.add_subplot(gs[1, 1])
    ax_pend  = fig.add_subplot(gs[2, 1])

    for ax in (ax_phase, ax_θ, ax_ω, ax_pend):
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("white")

    # =================================================
    # Phase portrait background
    # =================================================
    ax_phase.streamplot(
        θ_grid, ω_grid, U, V,
        color="#777777",
        density=2.0,
        linewidth=0.6,
        arrowsize=0.8
    )

    ax_phase.set_xlim(θ_grid[0], θ_grid[-1])
    ax_phase.set_ylim(ω_grid[0], ω_grid[-1])
    ax_phase.set_box_aspect(1.0)
    ax_phase.set_xlabel(r"$\theta$", color="white")
    ax_phase.set_ylabel(r"$\dot{\theta}$", color="white")

    true_lc = LineCollection([], linewidths=2.6)
    est_lc  = LineCollection([], linewidths=2.2)
    ax_phase.add_collection(true_lc)
    ax_phase.add_collection(est_lc)

    pt_true, = ax_phase.plot([], [], "o", color="#00E5FF", ms=6, label="True")
    pt_est,  = ax_phase.plot([], [], "o", color="white", ms=5,
                             label=f"{filter_name} mean")

    # =================================================
    # Covariance ellipse helper
    # =================================================
    def covariance_ellipse(mean, cov, ax, n_std=2.45, **kwargs):
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * n_std * np.sqrt(eigvals)

        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            **kwargs
        )
        ax.add_patch(ellipse)
        return ellipse

    cov_ellipse = covariance_ellipse(
        mean=[θ_est[0], ω_est[0]],
        cov=kf_history.covs[0],
        ax=ax_phase,
        edgecolor="white",
        facecolor="none",
        linewidth=2.4,
        alpha=0.65,
        label=r"95% covariance"
    )

    ax_phase.legend(
        facecolor="black",
        edgecolor="white",
        labelcolor="white",
        loc="upper right"
    )

    # =================================================
    # θ(t)
    # =================================================
    ax_θ.set_xlim(0, T)
    ax_θ.set_ylim(-2.2, 2.2)
    ax_θ.set_ylabel(r"$\theta$", color="white")

    θ_true_line, = ax_θ.plot([], [], "--", lw=2.6, color="#00E5FF", label="True θ")
    θ_est_line,  = ax_θ.plot([], [], "-", lw=2.2, color="white",
                             label=f"{filter_name} mean θ")

    θ_ci = ax_θ.fill_between([], [], [], color="white", alpha=0.25,
                             label=r"$\pm 2\sigma_\theta$")

    meas_scatter = ax_θ.scatter([], [], s=14, color="#AAAAAA", alpha=0.7,
                                label=r"$y=\sin(\theta)+v$")

    ax_θ.legend(facecolor="black", edgecolor="white", labelcolor="white")

    # =================================================
    # ω(t)
    # =================================================
    ax_ω.set_xlim(0, T)
    ax_ω.set_ylim(-6.5, 6.5)
    ax_ω.set_ylabel(r"$\dot{\theta}$", color="white")
    ax_ω.set_xlabel("time (s)", color="white")

    ω_true_line, = ax_ω.plot([], [], "--", lw=2.6, color="#FFB000", label="True ω")
    ω_est_line,  = ax_ω.plot([], [], "-", lw=2.2, color="white",
                             label=f"{filter_name} mean ω")

    ω_ci = ax_ω.fill_between([], [], [], color="white", alpha=0.25,
                             label=r"$\pm 2\sigma_\omega$")

    ax_ω.legend(facecolor="black", edgecolor="white", labelcolor="white")

    # =================================================
    # Pendulum
    # =================================================
    l_draw = 3.0
    span = 1.3 * l_draw
    x_pivot = 0.0
    y_pivot = 0.25 * span   # ← OPTIONAL upward shift (see below)

    ax_pend.set_xlim(-span, span)
    ax_pend.set_ylim(-span, 0.35 * span)
    ax_pend.set_box_aspect(1.0)
    ax_pend.set_autoscale_on(False)
    ax_pend.axis("off")

    ax_pend.plot(x_pivot, y_pivot, "wo", ms=5)
    pend_line, = ax_pend.plot([], [], lw=4.2, color="white")
    pend_mass, = ax_pend.plot([], [], "o", ms=12, color="#FF3B3B")


    # =================================================
    # Helpers
    # =================================================
    def fade_colors(n, rgb):
        a = np.linspace(0.15, 1.0, n)
        return [(rgb[0], rgb[1], rgb[2], ai) for ai in a]

    true_segs, est_segs = [], []
    max_len = 160

    # =================================================
    # Update
    # =================================================
    def update(k):
        nonlocal θ_ci, ω_ci, cov_ellipse

        if k > 0:
            true_segs.append([[θ_true[k-1], ω_true[k-1]],
                              [θ_true[k],   ω_true[k]]])
            est_segs.append([[θ_est[k-1], ω_est[k-1]],
                             [θ_est[k],   ω_est[k]]])

            true_lc.set_segments(true_segs[-max_len:])
            true_lc.set_color(fade_colors(len(true_segs[-max_len:]), (0, 0.9, 1)))

            est_lc.set_segments(est_segs[-max_len:])
            est_lc.set_color(fade_colors(len(est_segs[-max_len:]), (1, 1, 1)))

        pt_true.set_data([θ_true[k]], [ω_true[k]])
        pt_est.set_data([θ_est[k]],  [ω_est[k]])

        cov_ellipse.remove()
        cov_ellipse = covariance_ellipse(
            mean=[θ_est[k], ω_est[k]],
            cov=kf_history.covs[k],
            ax=ax_phase,
            edgecolor="white",
            facecolor="none",
            linewidth=2.4,
            alpha=0.65
        )

        x = x_pivot + l_draw * np.sin(θ_true[k])
        y = y_pivot - l_draw * np.cos(θ_true[k])

        pend_line.set_data([x_pivot, x], [y_pivot, y])
        pend_mass.set_data([x], [y])

        θ_true_line.set_data(t[:k+1], θ_true[:k+1])
        θ_est_line.set_data(t[:k+1], θ_est[:k+1])

        θ_ci.remove()
        θ_ci = ax_θ.fill_between(
            t[:k+1],
            θ_est[:k+1] - 2*θ_std[:k+1],
            θ_est[:k+1] + 2*θ_std[:k+1],
            color="white", alpha=0.25
        )

        ω_true_line.set_data(t[:k+1], ω_true[:k+1])
        ω_est_line.set_data(t[:k+1], ω_est[:k+1])

        ω_ci.remove()
        ω_ci = ax_ω.fill_between(
            t[:k+1],
            ω_est[:k+1] - 2*ω_std[:k+1],
            ω_est[:k+1] + 2*ω_std[:k+1],
            color="white", alpha=0.25
        )

        valid = obs_idx[obs_idx <= k]
        meas_scatter.set_offsets(np.c_[t[valid], y_meas[:len(valid)]])

        return ()

    # =================================================
    # Animate & save
    # =================================================
    ani = FuncAnimation(
        fig, update, frames=N,
        interval=slow_factor * dt * 1000,
        blit=False
    )

    fps = int(1.0 / (slow_factor * dt))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if output_format.lower() == "gif":
        writer = PillowWriter(fps=fps)
    elif output_format.lower() == "mp4":
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=1800,
            extra_args=[
                "-pix_fmt", "yuv420p",
                "-vf",
                "scale=1920:1080:force_original_aspect_ratio=decrease,"
                "pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
            ]
        )
    else:
        raise ValueError("output_format must be 'gif' or 'mp4'")

    ani.save(save_path, writer=writer)

    update(N - 1)
    plt.show()
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
import os

def animate_truth_vs_filter2(
    truth,
    measurements,
    kf_history,
    make_phase_portrait_fn,
    save_path: str,
    filter_name: str = "EKF",
    title: str = "Pendulum State Estimation",
    slow_factor: float = 4.0,
    output_format: str = "gif",
):
    # Extract data
    t = truth.times
    dt = t[1] - t[0]
    T = t[-1]
    N = len(t)

    θ_true = truth.obs[:, 0]
    ω_true = truth.obs[:, 1]

    θ_est = kf_history.means[:, 0]
    ω_est = kf_history.means[:, 1]

    θ_std = kf_history.stds[:, 0]
    ω_std = kf_history.stds[:, 1]

    obs_idx = measurements.obs_ind
    y_meas = measurements.obs.squeeze()

    # Phase portrait
    θ_grid, ω_grid, U, V = make_phase_portrait_fn()

    # Figure setup
    fig = plt.figure(figsize=(22, 12), facecolor='#0a0a0a')
    
    # Move main title higher and reduce font size slightly
    fig.suptitle(f"{title} ({filter_name})", color='#ffffff', fontsize=22, 
                 fontweight='bold', y=0.98)

    gs = fig.add_gridspec(3, 3, width_ratios=[1.6, 1, 1], 
                          height_ratios=[1.15, 1.15, 1.25],
                          hspace=0.38, wspace=0.38,
                          left=0.06, right=0.97, top=0.935, bottom=0.06)

    ax_phase = fig.add_subplot(gs[:, 0])
    ax_θ = fig.add_subplot(gs[0, 1:])
    ax_ω = fig.add_subplot(gs[1, 1:])
    ax_pend = fig.add_subplot(gs[2, 1:])

    # Style all axes
    for ax in [ax_phase, ax_θ, ax_ω, ax_pend]:
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='#cccccc', labelsize=13, width=1.5, length=6)
        ax.xaxis.label.set_color('#ffffff')
        ax.yaxis.label.set_color('#ffffff')
        for spine in ax.spines.values():
            spine.set_color('#444444')
            spine.set_linewidth(2)

    # ========== PHASE PORTRAIT - USE CHATGPT STYLE ==========
    # Simple streamplot like the original ChatGPT code
    ax_phase.streamplot(
        θ_grid, ω_grid, U, V,
        color='#777777',
        density=2.0,
        linewidth=0.6,
        arrowsize=0.8
    )

    ax_phase.set_xlim(θ_grid[0], θ_grid[-1])
    ax_phase.set_ylim(ω_grid[0], ω_grid[-1])
    ax_phase.set_aspect('equal', adjustable='box')
    ax_phase.set_xlabel(r'$\theta$ (rad)', fontsize=16, fontweight='bold')
    ax_phase.set_ylabel(r'$\dot{\theta}$ (rad/s)', fontsize=16, fontweight='bold')
    ax_phase.set_title('Phase Portrait', color='#ffffff', fontsize=16, 
                      fontweight='bold', pad=12)
    ax_phase.grid(True, alpha=0.15, linestyle='-', linewidth=0.8, color='#333333')

    # Trajectory collections - thicker and more visible
    true_lc = LineCollection([], linewidths=4.5, zorder=6)
    est_lc = LineCollection([], linewidths=4.0, zorder=5)
    ax_phase.add_collection(true_lc)
    ax_phase.add_collection(est_lc)

    # Current position markers - much larger!
    pt_true, = ax_phase.plot([], [], 'o', color='#00ffff', ms=16, 
                             markeredgecolor='#ffffff', markeredgewidth=2.5,
                             label='True', zorder=10)
    pt_est, = ax_phase.plot([], [], 's', color='#ff3366', ms=14,
                           markeredgecolor='#ffffff', markeredgewidth=2.5,
                           label=f'{filter_name}', zorder=9)

    # Covariance ellipse - MUCH MORE VISIBLE with thick border and fill
    def create_ellipse(mean, cov, n_std=2.0):
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * n_std * np.sqrt(eigvals)
        return Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor='#ff3366', facecolor='#ff3366', 
                      linewidth=4, alpha=0.35, linestyle='-', zorder=4)

    cov_ellipse = create_ellipse([θ_est[0], ω_est[0]], kf_history.covs[0])
    ax_phase.add_patch(cov_ellipse)

    # Add dummy line for covariance in legend
    ax_phase.plot([], [], '-', color='#ff3366', linewidth=4, alpha=0.35, 
                 label='95% covariance')

    legend = ax_phase.legend(loc='upper right', facecolor='#1a1a1a', 
                   edgecolor='#666666', labelcolor='#ffffff',
                   fontsize=13, framealpha=0.95, borderpad=1)
    legend.get_frame().set_linewidth(2)

    # ========== THETA TIME SERIES ==========
    ax_θ.set_xlim(0, T)
    y_margin = 0.3
    ax_θ.set_ylim(np.min(θ_true) - y_margin, np.max(θ_true) + y_margin)
    ax_θ.set_ylabel(r'$\theta$ (rad)', fontsize=16, fontweight='bold')
    ax_θ.set_title('Angle Evolution', color='#ffffff', fontsize=16, 
                   fontweight='bold', pad=12)
    ax_θ.grid(True, alpha=0.15, linestyle='-', linewidth=0.8, color='#333333')

    θ_true_line, = ax_θ.plot([], [], '-', lw=3.5, color='#00ffff', 
                             label='True θ', alpha=1.0, zorder=5)
    θ_est_line, = ax_θ.plot([], [], '-', lw=3.0, color='#ff3366',
                            label=f'{filter_name} θ', alpha=0.95, zorder=4)
    θ_ci = ax_θ.fill_between([], [], [], color='#ff3366', alpha=0.25,
                             label=r'$\pm 2\sigma$', zorder=3)
    
    # Measurements - smaller, less dominant
    meas_scatter = ax_θ.scatter([], [], s=20, color='#ffaa00', 
                               marker='o', alpha=0.5, linewidths=1,
                               edgecolors='#ffffff',
                               label='Measurements', zorder=6)

    legend = ax_θ.legend(loc='upper right', facecolor='#1a1a1a',
               edgecolor='#666666', labelcolor='#ffffff',
               fontsize=12, framealpha=0.95, ncol=2, borderpad=0.8)
    legend.get_frame().set_linewidth(2)

    # ========== OMEGA TIME SERIES ==========
    ax_ω.set_xlim(0, T)
    ω_margin = 0.5
    ax_ω.set_ylim(np.min(ω_true) - ω_margin, np.max(ω_true) + ω_margin)
    ax_ω.set_ylabel(r'$\dot{\theta}$ (rad/s)', fontsize=16, fontweight='bold')
    ax_ω.set_xlabel('Time (s)', fontsize=16, fontweight='bold')
    ax_ω.set_title('Angular Velocity Evolution', color='#ffffff', fontsize=16, 
                   fontweight='bold', pad=12)
    ax_ω.grid(True, alpha=0.15, linestyle='-', linewidth=0.8, color='#333333')

    ω_true_line, = ax_ω.plot([], [], '-', lw=3.5, color='#00ffff',
                             label='True ω', alpha=1.0, zorder=5)
    ω_est_line, = ax_ω.plot([], [], '-', lw=3.0, color='#ff3366',
                            label=f'{filter_name} ω', alpha=0.95, zorder=4)
    ω_ci = ax_ω.fill_between([], [], [], color='#ff3366', alpha=0.25,
                             label=r'$\pm 2\sigma$', zorder=3)

    legend = ax_ω.legend(loc='upper right', facecolor='#1a1a1a',
               edgecolor='#666666', labelcolor='#ffffff',
               fontsize=12, framealpha=0.95, borderpad=0.8)
    legend.get_frame().set_linewidth(2)

    # ========== PENDULUM ANIMATION ==========
    ax_pend.set_aspect('equal', adjustable='box')
    ax_pend.axis('off')
    ax_pend.set_title('Physical System', color='#ffffff', fontsize=16, 
                     fontweight='bold', pad=12)
    
    L = 1.0
    ax_pend.set_xlim(-1.5, 1.5)
    ax_pend.set_ylim(-1.5, 0.3)

    pivot_x, pivot_y = 0.0, 0.0
    ax_pend.plot(pivot_x, pivot_y, 'o', color='#888888', ms=12, zorder=10,
                markeredgecolor='#ffffff', markeredgewidth=2)
    
    # Ceiling
    ceiling_line, = ax_pend.plot([-1.5, 1.5], [0, 0], '-', lw=6, 
                                 color='#444444', solid_capstyle='butt', zorder=1)

    # Pendulum rod
    pend_rod, = ax_pend.plot([], [], '-', lw=5, color='#aaaaaa', 
                            solid_capstyle='round', zorder=5)
    
    # Pendulum bob
    pend_bob, = ax_pend.plot([], [], 'o', ms=28, color='#ff3366',
                            markeredgecolor='#ffffff', markeredgewidth=3, zorder=6)
    
    # Motion trail - more visible
    trail_line, = ax_pend.plot([], [], '-', lw=2.5, color='#ff3366', 
                              alpha=0.4, zorder=3)
    trail_x, trail_y = [], []
    max_trail = 70

    # Angle arc indicator
    arc_line, = ax_pend.plot([], [], '-', lw=2.5, color='#00ffff', 
                            alpha=0.7, zorder=4)

    # Trajectory fading
    def fade_colors(n, color_hex):
        r = int(color_hex[1:3], 16) / 255
        g = int(color_hex[3:5], 16) / 255
        b = int(color_hex[5:7], 16) / 255
        alpha = np.linspace(0.2, 1.0, n)
        return [(r, g, b, a) for a in alpha]

    true_segments, est_segments = [], []
    max_segments = 80  # Shorter trails - fade faster to hide discretization artifacts

    # Update function
    def update(k):
        nonlocal θ_ci, ω_ci, cov_ellipse

        θ_wrapped = np.arctan2(np.sin(θ_true[k]), np.cos(θ_true[k]))
        
        # Phase Portrait
        if k > 0:
            true_segments.append([[θ_true[k-1], ω_true[k-1]], 
                                 [θ_true[k], ω_true[k]]])
            est_segments.append([[θ_est[k-1], ω_est[k-1]], 
                                [θ_est[k], ω_est[k]]])

            true_lc.set_segments(true_segments[-max_segments:])
            true_lc.set_color(fade_colors(len(true_segments[-max_segments:]), '#00ffff'))

            est_lc.set_segments(est_segments[-max_segments:])
            est_lc.set_color(fade_colors(len(est_segments[-max_segments:]), '#ff3366'))

        pt_true.set_data([θ_true[k]], [ω_true[k]])
        pt_est.set_data([θ_est[k]], [ω_est[k]])

        cov_ellipse.remove()
        cov_ellipse = create_ellipse([θ_est[k], ω_est[k]], kf_history.covs[k])
        ax_phase.add_patch(cov_ellipse)

        # Pendulum
        x_bob = pivot_x + L * np.sin(θ_wrapped)
        y_bob = pivot_y - L * np.cos(θ_wrapped)

        pend_rod.set_data([pivot_x, x_bob], [pivot_y, y_bob])
        pend_bob.set_data([x_bob], [y_bob])

        trail_x.append(x_bob)
        trail_y.append(y_bob)
        if len(trail_x) > max_trail:
            trail_x.pop(0)
            trail_y.pop(0)
        trail_line.set_data(trail_x, trail_y)

        if abs(θ_wrapped) > 0.05:
            arc_angles = np.linspace(0, θ_wrapped, 30)
            arc_r = 0.35
            arc_x = pivot_x + arc_r * np.sin(arc_angles)
            arc_y = pivot_y - arc_r * np.cos(arc_angles)
            arc_line.set_data(arc_x, arc_y)
        else:
            arc_line.set_data([], [])

        # Time Series
        θ_true_line.set_data(t[:k+1], θ_true[:k+1])
        θ_est_line.set_data(t[:k+1], θ_est[:k+1])

        θ_ci.remove()
        θ_ci = ax_θ.fill_between(
            t[:k+1],
            θ_est[:k+1] - 2*θ_std[:k+1],
            θ_est[:k+1] + 2*θ_std[:k+1],
            color='#ff3366', alpha=0.25, zorder=3
        )

        ω_true_line.set_data(t[:k+1], ω_true[:k+1])
        ω_est_line.set_data(t[:k+1], ω_est[:k+1])

        ω_ci.remove()
        ω_ci = ax_ω.fill_between(
            t[:k+1],
            ω_est[:k+1] - 2*ω_std[:k+1],
            ω_est[:k+1] + 2*ω_std[:k+1],
            color='#ff3366', alpha=0.25, zorder=3
        )

        # Measurements - plot them UNDER the lines
        valid = obs_idx[obs_idx <= k]
        meas_scatter.set_offsets(np.c_[t[valid], y_meas[:len(valid)]])

        return ()

    # Create animation
    ani = FuncAnimation(fig, update, frames=N,
                       interval=slow_factor * dt * 1000,
                       blit=False)

    fps = int(1.0 / (slow_factor * dt))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if output_format.lower() == "gif":
        writer = PillowWriter(fps=fps)
    elif output_format.lower() == "mp4":
        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=3000,
                             extra_args=["-pix_fmt", "yuv420p"])
    else:
        raise ValueError("output_format must be 'gif' or 'mp4'")

    print(f"Saving animation to {save_path}...")
    ani.save(save_path, writer=writer)
    print("Animation saved successfully!")
    
    update(N - 1)
    plt.show()
    
    
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os

def animate_particle_filter(
    truth,
    measurements,
    particle_set,
    make_phase_portrait_fn,
    save_path: str,
    filter_name: str = "Bootstrap PF",
    title: str = "Particle Filter State Estimation",
    slow_factor: float = 4.0,
    output_format: str = "mp4",
    num_particles_display: int = 20,
    ess_threshold_ratio: float = 0.5,
):
    """
    Animate particle filter results showing particle distributions and filter health.
    
    Parameters
    ----------
    truth : Observations
        True state trajectory
    measurements : Observations
        Measurement data
    particle_set : ParticleSet
        Full particle filter output with trajectories and weights
    make_phase_portrait_fn : Callable
        Function that returns (θ_grid, ω_grid, U, V) for phase portrait
    save_path : str
        Output file path
    filter_name : str
        Name of the particle filter variant
    title : str
        Main title for the animation
    slow_factor : float
        Animation speed multiplier
    output_format : str
        'gif' or 'mp4'
    num_particles_display : int
        Number of particles to display in visualizations
    ess_threshold_ratio : float
        ESS threshold for resampling indicator
    """
    
    # Extract data
    t = truth.times
    dt = t[1] - t[0]
    T = t[-1]
    N_steps = len(t)

    θ_true = truth.obs[:, 0]
    ω_true = truth.obs[:, 1]

    obs_idx = measurements.obs_ind
    y_meas = measurements.obs.squeeze()

    # Convert particle set to Gaussian summary
    kf_summary = particle_set.to_kf_tracker()
    θ_est = kf_summary.means[:, 0]
    ω_est = kf_summary.means[:, 1]
    θ_std = kf_summary.stds[:, 0]
    ω_std = kf_summary.stds[:, 1]

    # Extract particle data
    particles = particle_set.particles  # (N_particles, N_steps, d)
    weights = particle_set.weights      # (N_particles, N_steps)
    N_particles = particles.shape[0]

    # Compute ESS over time
    ess_history = np.zeros(N_steps)
    for k in range(N_steps):
        ess_history[k] = particle_set.effective_sample_size(k)

    # Phase portrait
    θ_grid, ω_grid, U, V = make_phase_portrait_fn()

    # Figure setup
    fig = plt.figure(figsize=(24, 13), facecolor='#0a0a0a')
    
    fig.suptitle(f"{title} ({filter_name}, N={N_particles})", 
                 color='#ffffff', fontsize=22, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(4, 4, 
                          width_ratios=[1.5, 1, 1, 1], 
                          height_ratios=[1.2, 1.2, 1.0, 0.8],
                          hspace=0.35, wspace=0.35,
                          left=0.05, right=0.97, top=0.935, bottom=0.05)

    ax_phase = fig.add_subplot(gs[:3, 0])
    ax_θ = fig.add_subplot(gs[0, 1:])
    ax_ω = fig.add_subplot(gs[1, 1:])
    ax_pend = fig.add_subplot(gs[2, 1:])
    ax_weights = fig.add_subplot(gs[3, 0])
    ax_ess = fig.add_subplot(gs[3, 1:])

    # Style all axes
    for ax in [ax_phase, ax_θ, ax_ω, ax_pend, ax_weights, ax_ess]:
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='#cccccc', labelsize=11, width=1.5, length=6)
        ax.xaxis.label.set_color('#ffffff')
        ax.yaxis.label.set_color('#ffffff')
        for spine in ax.spines.values():
            spine.set_color('#444444')
            spine.set_linewidth(2)

    # ========== PHASE PORTRAIT ==========
    ax_phase.streamplot(
        θ_grid, ω_grid, U, V,
        color='#555555',
        density=1.8,
        linewidth=0.6,
        arrowsize=0.8
    )

    ax_phase.set_xlim(θ_grid[0], θ_grid[-1])
    ax_phase.set_ylim(ω_grid[0], ω_grid[-1])
    ax_phase.set_aspect('equal', adjustable='box')
    ax_phase.set_xlabel(r'$\theta$ (rad)', fontsize=15, fontweight='bold')
    ax_phase.set_ylabel(r'$\dot{\theta}$ (rad/s)', fontsize=15, fontweight='bold')
    ax_phase.set_title('Phase Portrait with Particle Cloud', color='#ffffff', 
                      fontsize=16, fontweight='bold', pad=12)
    ax_phase.grid(True, alpha=0.15, linestyle='-', linewidth=0.8, color='#333333')

    # Particle scatter (will be updated)
    particle_scatter = ax_phase.scatter([], [], s=15, alpha=0.5, 
                                       cmap='YlOrRd', edgecolors='none',
                                       vmin=0, zorder=5)

    # Trajectory collections
    true_lc = LineCollection([], linewidths=3.5, zorder=6)
    est_lc = LineCollection([], linewidths=3.0, zorder=5)
    ax_phase.add_collection(true_lc)
    ax_phase.add_collection(est_lc)

    # Current position markers
    pt_true, = ax_phase.plot([], [], 'o', color='#00ffff', ms=14, 
                             markeredgecolor='#ffffff', markeredgewidth=2,
                             label='True', zorder=10)
    pt_est, = ax_phase.plot([], [], 's', color='#ff3366', ms=12,
                           markeredgecolor='#ffffff', markeredgewidth=2,
                           label='Mean Est.', zorder=9)

    # Covariance ellipse
    def create_ellipse(mean, cov, n_std=2.0):
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * n_std * np.sqrt(eigvals)
        return Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor='#ff3366', facecolor='#ff3366', 
                      linewidth=3, alpha=0.25, linestyle='-', zorder=4)

    cov_ellipse = create_ellipse([θ_est[0], ω_est[0]], kf_summary.covs[0])
    ax_phase.add_patch(cov_ellipse)

    # Dummy for particle cloud in legend
    ax_phase.scatter([], [], s=15, c='#ffaa00', alpha=0.5, label='Particles')
    ax_phase.plot([], [], '-', color='#ff3366', linewidth=3, alpha=0.25, 
                 label='95% covariance')

    legend = ax_phase.legend(loc='upper right', facecolor='#1a1a1a', 
                            edgecolor='#666666', labelcolor='#ffffff',
                            fontsize=12, framealpha=0.95, borderpad=1)
    legend.get_frame().set_linewidth(2)

    # ========== THETA TIME SERIES ==========
    ax_θ.set_xlim(0, T)
    y_margin = 0.3
    ax_θ.set_ylim(np.min(θ_true) - y_margin, np.max(θ_true) + y_margin)
    ax_θ.set_ylabel(r'$\theta$ (rad)', fontsize=15, fontweight='bold')
    ax_θ.set_title('Angle Evolution', color='#ffffff', fontsize=16, 
                   fontweight='bold', pad=12)
    ax_θ.grid(True, alpha=0.15, linestyle='-', linewidth=0.8, color='#333333')

    θ_true_line, = ax_θ.plot([], [], '-', lw=3.0, color='#00ffff', 
                             label='True θ', alpha=1.0, zorder=5)
    θ_est_line, = ax_θ.plot([], [], '-', lw=2.5, color='#ff3366',
                            label='Mean θ', alpha=0.95, zorder=4)
    θ_ci = ax_θ.fill_between([], [], [], color='#ff3366', alpha=0.25,
                             label=r'$\pm 2\sigma$', zorder=3)
    
    # Particle scatter for θ
    θ_particle_scatter = ax_θ.scatter([], [], s=8, alpha=0.3, 
                                     cmap='YlOrRd', edgecolors='none',
                                     vmin=0, zorder=6)
    
    meas_scatter = ax_θ.scatter([], [], s=20, color='#a0a0a0', 
                               marker='o', alpha=0.5, linewidths=1,
                               edgecolors='#ffffff',
                               label='Meas.', zorder=7)

    legend = ax_θ.legend(loc='upper right', facecolor='#1a1a1a',
                        edgecolor='#666666', labelcolor='#ffffff',
                        fontsize=11, framealpha=0.95, ncol=2, borderpad=0.7)
    legend.get_frame().set_linewidth(2)

    # ========== OMEGA TIME SERIES ==========
    ax_ω.set_xlim(0, T)
    ω_margin = 0.5
    ax_ω.set_ylim(np.min(ω_true) - ω_margin, np.max(ω_true) + ω_margin)
    ax_ω.set_ylabel(r'$\dot{\theta}$ (rad/s)', fontsize=15, fontweight='bold')
    ax_ω.set_xlabel('Time (s)', fontsize=15, fontweight='bold')
    ax_ω.set_title('Angular Velocity Evolution', color='#ffffff', fontsize=16, 
                   fontweight='bold', pad=12)
    ax_ω.grid(True, alpha=0.15, linestyle='-', linewidth=0.8, color='#333333')

    ω_true_line, = ax_ω.plot([], [], '-', lw=3.0, color='#00ffff',
                             label='True ω', alpha=1.0, zorder=5)
    ω_est_line, = ax_ω.plot([], [], '-', lw=2.5, color='#ff3366',
                            label='Mean ω', alpha=0.95, zorder=4)
    ω_ci = ax_ω.fill_between([], [], [], color='#ff3366', alpha=0.25,
                             label=r'$\pm 2\sigma$', zorder=3)
    
    # Particle scatter for ω
    ω_particle_scatter = ax_ω.scatter([], [], s=8, alpha=0.3, 
                                     cmap='YlOrRd', edgecolors='none',
                                     vmin=0, zorder=6)

    legend = ax_ω.legend(loc='upper right', facecolor='#1a1a1a',
                        edgecolor='#666666', labelcolor='#ffffff',
                        fontsize=11, framealpha=0.95, borderpad=0.7)
    legend.get_frame().set_linewidth(2)

    # ========== PENDULUM ANIMATION ==========
    ax_pend.set_aspect('equal', adjustable='box')
    ax_pend.axis('off')
    ax_pend.set_title('Physical System', color='#ffffff', fontsize=16, 
                     fontweight='bold', pad=12)
    
    L = 1.0
    ax_pend.set_xlim(-1.5, 1.5)
    ax_pend.set_ylim(-1.5, 0.3)

    pivot_x, pivot_y = 0.0, 0.0
    ax_pend.plot(pivot_x, pivot_y, 'o', color='#888888', ms=12, zorder=10,
                markeredgecolor='#ffffff', markeredgewidth=2)
    
    ceiling_line, = ax_pend.plot([-1.5, 1.5], [0, 0], '-', lw=6, 
                                 color='#444444', solid_capstyle='butt', zorder=1)

    pend_rod, = ax_pend.plot([], [], '-', lw=5, color='#aaaaaa', 
                            solid_capstyle='round', zorder=5)
    
    pend_bob, = ax_pend.plot([], [], 'o', ms=28, color='#ff3366',
                            markeredgecolor='#ffffff', markeredgewidth=3, zorder=6)
    
    trail_line, = ax_pend.plot([], [], '-', lw=2.5, color='#ff3366', 
                              alpha=0.4, zorder=3)
    trail_x, trail_y = [], []
    max_trail = 70

    arc_line, = ax_pend.plot([], [], '-', lw=2.5, color='#00ffff', 
                            alpha=0.7, zorder=4)

    # ========== WEIGHT HISTOGRAM ==========
    ax_weights.set_xlabel('Weight', fontsize=13, fontweight='bold')
    ax_weights.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax_weights.set_title('Particle Weight Distribution', color='#ffffff', 
                        fontsize=14, fontweight='bold', pad=10)
    ax_weights.grid(True, alpha=0.15, linestyle='-', linewidth=0.8, color='#333333')
    
    # Will be updated each frame
    weight_bars = None

    # ========== ESS OVER TIME ==========
    ax_ess.set_xlim(0, T)
    ax_ess.set_ylim(0, N_particles * 1.1)
    ax_ess.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
    ax_ess.set_ylabel('ESS', fontsize=13, fontweight='bold')
    ax_ess.set_title('Effective Sample Size', color='#ffffff', 
                    fontsize=14, fontweight='bold', pad=10)
    ax_ess.grid(True, alpha=0.15, linestyle='-', linewidth=0.8, color='#333333')

    # Threshold line
    threshold = ess_threshold_ratio * N_particles
    ax_ess.axhline(threshold, color='#ff6666', linestyle='--', linewidth=2, 
                   alpha=0.7, label=f'Threshold ({ess_threshold_ratio:.1%})')
    
    ess_line, = ax_ess.plot([], [], '-', lw=2.5, color='#00ff88', 
                            label='ESS', alpha=0.9)
    
    # Resampling markers (will add as we go)
    resample_markers = ax_ess.scatter([], [], s=80, color='#ff3366', 
                                     marker='v', edgecolors='#ffffff',
                                     linewidths=2, zorder=10,
                                     label='Resampling')

    legend = ax_ess.legend(loc='upper right', facecolor='#1a1a1a',
                          edgecolor='#666666', labelcolor='#ffffff',
                          fontsize=11, framealpha=0.95, borderpad=0.7)
    legend.get_frame().set_linewidth(2)

    # Track resampling events
    resample_times = []
    resample_ess = []

    # Trajectory fading
    def fade_colors(n, color_hex):
        r = int(color_hex[1:3], 16) / 255
        g = int(color_hex[3:5], 16) / 255
        b = int(color_hex[5:7], 16) / 255
        alpha = np.linspace(0.2, 1.0, n)
        return [(r, g, b, a) for a in alpha]

    true_segments, est_segments = [], []
    max_segments = 80

    # Update function
    def update(k):
        nonlocal θ_ci, ω_ci, cov_ellipse, weight_bars

        θ_wrapped = np.arctan2(np.sin(θ_true[k]), np.cos(θ_true[k]))
        
        # ===== Phase Portrait =====
        if k > 0:
            true_segments.append([[θ_true[k-1], ω_true[k-1]], 
                                 [θ_true[k], ω_true[k]]])
            est_segments.append([[θ_est[k-1], ω_est[k-1]], 
                                [θ_est[k], ω_est[k]]])

            true_lc.set_segments(true_segments[-max_segments:])
            true_lc.set_color(fade_colors(len(true_segments[-max_segments:]), '#00ffff'))

            est_lc.set_segments(est_segments[-max_segments:])
            est_lc.set_color(fade_colors(len(est_segments[-max_segments:]), '#ff3366'))

        # Sample particles for display
        w_k = weights[:, k]
        # Normalize for sampling
        w_k_norm = w_k / np.sum(w_k)
        
        # Sample with replacement biased by weights
        sample_idx = np.random.choice(N_particles, size=num_particles_display, 
                                     replace=True, p=w_k_norm)
        particles_sample = particles[sample_idx, k, :]
        weights_sample = w_k[sample_idx]
        
        # Update particle scatter in phase portrait
        particle_scatter.set_offsets(particles_sample)
        particle_scatter.set_array(weights_sample)
        particle_scatter.set_clim(vmin=0, vmax=np.max(w_k))

        pt_true.set_data([θ_true[k]], [ω_true[k]])
        pt_est.set_data([θ_est[k]], [ω_est[k]])

        cov_ellipse.remove()
        cov_ellipse = create_ellipse([θ_est[k], ω_est[k]], kf_summary.covs[k])
        ax_phase.add_patch(cov_ellipse)

        # ===== Pendulum =====
        x_bob = pivot_x + L * np.sin(θ_wrapped)
        y_bob = pivot_y - L * np.cos(θ_wrapped)

        pend_rod.set_data([pivot_x, x_bob], [pivot_y, y_bob])
        pend_bob.set_data([x_bob], [y_bob])

        trail_x.append(x_bob)
        trail_y.append(y_bob)
        if len(trail_x) > max_trail:
            trail_x.pop(0)
            trail_y.pop(0)
        trail_line.set_data(trail_x, trail_y)

        if abs(θ_wrapped) > 0.05:
            arc_angles = np.linspace(0, θ_wrapped, 30)
            arc_r = 0.35
            arc_x = pivot_x + arc_r * np.sin(arc_angles)
            arc_y = pivot_y - arc_r * np.cos(arc_angles)
            arc_line.set_data(arc_x, arc_y)
        else:
            arc_line.set_data([], [])

        # ===== Time Series =====
        θ_true_line.set_data(t[:k+1], θ_true[:k+1])
        θ_est_line.set_data(t[:k+1], θ_est[:k+1])

        θ_ci.remove()
        θ_ci = ax_θ.fill_between(
            t[:k+1],
            θ_est[:k+1] - 2*θ_std[:k+1],
            θ_est[:k+1] + 2*θ_std[:k+1],
            color='#ff3366', alpha=0.25, zorder=3
        )
        
        # Particle scatter on θ plot
        θ_particles_k = particles[sample_idx, k, 0]
        t_particles = np.full(len(sample_idx), t[k])
        θ_particle_scatter.set_offsets(np.c_[t_particles, θ_particles_k])
        θ_particle_scatter.set_array(weights_sample)
        θ_particle_scatter.set_clim(vmin=0, vmax=np.max(w_k))

        ω_true_line.set_data(t[:k+1], ω_true[:k+1])
        ω_est_line.set_data(t[:k+1], ω_est[:k+1])

        ω_ci.remove()
        ω_ci = ax_ω.fill_between(
            t[:k+1],
            ω_est[:k+1] - 2*ω_std[:k+1],
            ω_est[:k+1] + 2*ω_std[:k+1],
            color='#ff3366', alpha=0.25, zorder=3
        )
        
        # Particle scatter on ω plot
        ω_particles_k = particles[sample_idx, k, 1]
        ω_particle_scatter.set_offsets(np.c_[t_particles, ω_particles_k])
        ω_particle_scatter.set_array(weights_sample)
        ω_particle_scatter.set_clim(vmin=0, vmax=np.max(w_k))

        # Measurements
        valid = obs_idx[obs_idx <= k]
        meas_scatter.set_offsets(np.c_[t[valid], y_meas[:len(valid)]])

        # ===== Weight Histogram =====
        ax_weights.clear()
        ax_weights.set_xlabel('Weight', fontsize=13, fontweight='bold')
        ax_weights.set_ylabel('Count', fontsize=13, fontweight='bold')
        ax_weights.set_title('Particle Weight Distribution', color='#ffffff', 
                            fontsize=14, fontweight='bold', pad=10)
        ax_weights.grid(True, alpha=0.15, linestyle='-', linewidth=0.8, color='#333333')
        ax_weights.set_facecolor('#0a0a0a')
        ax_weights.tick_params(colors='#cccccc', labelsize=11)
        for spine in ax_weights.spines.values():
            spine.set_color('#444444')
            spine.set_linewidth(2)
        
        ax_weights.hist(w_k, bins=30, color='#ffaa00', alpha=0.7, edgecolor='#ffffff')
        ax_weights.set_xlim(0, np.max(w_k) * 1.1 if np.max(w_k) > 0 else 1)

        # ===== ESS Plot =====
        ess_line.set_data(t[:k+1], ess_history[:k+1])
        
        # Check if resampling occurred (ESS jumped back up)
        if k > 0 and ess_history[k] > ess_history[k-1] * 1.5:
            resample_times.append(t[k])
            resample_ess.append(ess_history[k])
        
        if len(resample_times) > 0:
            resample_markers.set_offsets(np.c_[resample_times, resample_ess])

        return ()

    # Create animation
    ani = FuncAnimation(fig, update, frames=N_steps,
                       interval=slow_factor * dt * 1000,
                       blit=False)

    fps = int(1.0 / (slow_factor * dt))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if output_format.lower() == "gif":
        writer = PillowWriter(fps=fps)
    elif output_format.lower() == "mp4":
        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=3000,
                             extra_args=["-pix_fmt", "yuv420p"])
    else:
        raise ValueError("output_format must be 'gif' or 'mp4'")

    print(f"Saving particle filter animation to {save_path}...")
    ani.save(save_path, writer=writer)
    print("Animation saved successfully!")
    
    update(N_steps - 1)
    plt.show()