import numpy as np


def lif_step_exact(v, I_t, dt, tau_m, v_rest, v_reset, v_th, R=1.0):
    """
    Exact (exponential) integration step for current-based LIF neuron
    with piecewise constant input over [t, t+dt].
    """
    # Exponential decay factor
    a = np.exp(-dt / tau_m)

    # Steady-state voltage under constant input I_t
    v_inf = v_rest + R * I_t

    # Exact solution at t + dt
    v_next = v_inf + (v - v_inf) * a

    spike = (v_next >= v_th).astype(float)
    v_next = np.where(spike > 0, v_reset, v_next)
    return v_next, spike


def simulate_lif_exact(I_t, dt, tau_m=20e-3, v_rest=0.0, v_reset=0.0, v_th=1.0, R=1.0):
    """
    Simulate a single LIF neuron using exact (exponential) integration.
    """
    T = len(I_t)
    t = np.arange(T) * dt
    v_hist = np.zeros(T)
    s_hist = np.zeros(T)
    v = v_rest

    for n in range(T):
        v, s = lif_step_exact(v, I_t[n], dt, tau_m, v_rest, v_reset, v_th, R)
        v_hist[n] = v
        s_hist[n] = s

    return t, v_hist, s_hist
