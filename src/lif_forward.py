import numpy as np


def lif_step_forward(v, I_t, dt, tau_m, v_rest, v_reset, v_th, R=1.0):
    """
    Forward (explicit) Euler step for current-based LIF neuron.
    """
    dv = (-(v - v_rest) + R * I_t) * (dt / tau_m)
    v_next = v + dv

    spike = (v_next >= v_th).astype(float)
    v_next = np.where(spike > 0, v_reset, v_next)
    return v_next, spike


def simulate_lif_forward(I_t, dt, tau_m=20e-3, v_rest=0.0, v_reset=0.0, v_th=1.0, R=1.0, v_min=-1e3, v_max=1e3):
    """
    Simulate a single LIF neuron using forward Euler integration.

    Parameters
    ----------
    I_t : np.ndarray (T,)
        Input current at each time step.
    dt : float
        Time step (seconds).

    Returns
    -------
    t : np.ndarray (T,)
        Time vector.
    v_hist : np.ndarray (T,)
        Membrane potential over time.
    s_hist : np.ndarray (T,)
        Spike train (0 or 1) over time.
    """
    T = len(I_t)
    t = np.arange(T) * dt
    v_hist = np.zeros(T)
    s_hist = np.zeros(T)
    v = v_rest

    for n in range(T):
        v, s = lif_step_forward(v, I_t[n], dt, tau_m, v_rest, v_reset, v_th, R)

        # Safety check
        if not np.isfinite(v):
            raise FloatingPointError(f"Non-finite voltage at step {n}: v={v}")

        if v < v_min or v > v_max:
            raise FloatingPointError(
                f"Voltage out of bounds at step {n}: v={v}, "
                f"consider reducing dt or input strength."
            )

        v_hist[n] = v
        s_hist[n] = s


    return t, v_hist, s_hist
