import numpy as np


def lif_step_backward(v, I_t, dt, tau_m, v_rest, v_reset, v_th, R=1.0):
    """
    Backward (implicit) Euler step for current-based LIF neuron.

    Assumes input current I_t is constant over the interval [t, t+dt].
    """
    # Closed-form backward Euler update for linear LIF dynamics:
    # v_{n+1} = (v_n + (dt/tau_m)*(v_rest + R*I_t)) / (1 + dt/tau_m)
    alpha = dt / tau_m
    v_next = (v + alpha * (v_rest + R * I_t)) / (1.0 + alpha)

    spike = (v_next >= v_th).astype(float)
    v_next = np.where(spike > 0, v_reset, v_next)
    return v_next, spike


def simulate_lif_backward(I_t, dt, tau_m=20e-3, v_rest=0.0, v_reset=0.0, v_th=1.0, R=1.0):
    """
    Simulate a single LIF neuron using backward (implicit) Euler integration.
    """
    T = len(I_t)
    t = np.arange(T) * dt
    v_hist = np.zeros(T)
    s_hist = np.zeros(T)
    v = v_rest

    for n in range(T):
        v, s = lif_step_backward(v, I_t[n], dt, tau_m, v_rest, v_reset, v_th, R)
        v_hist[n] = v
        s_hist[n] = s

    return t, v_hist, s_hist
