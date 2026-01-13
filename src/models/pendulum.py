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
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse


def animate_truth_vs_filter(
    truth,
    measurements,
    kf_history,
    make_phase_portrait_fn,
    save_path: str,
    filter_name: str = "EKF",
    title: str = "True vs Filter",
    slow_factor: float = 4.0,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Ellipse

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

    ax_pend.set_xlim(-span, span)
    ax_pend.set_ylim(-span, 0.35 * span)
    ax_pend.set_box_aspect(1.0)
    ax_pend.axis("off")

    ax_pend.plot(0, 0, "wo", ms=5)
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

        x = l_draw * np.sin(θ_true[k])
        y = -l_draw * np.cos(θ_true[k])
        pend_line.set_data([0, x], [0, y])
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
    ani.save(save_path, writer=PillowWriter(fps=fps))

    update(N - 1)
    plt.show()