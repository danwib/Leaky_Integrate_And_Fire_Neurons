import os
import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if Qt is installed
import matplotlib.pyplot as plt


# Allow importing from src/ when running this script directly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.lif_forward import simulate_lif_forward
from src.lif_backward import simulate_lif_backward
from src.lif_exact import simulate_lif_exact


def constant_current(T, dt, I_value):
    steps = int(T / dt)
    return np.full(steps, I_value, dtype=float)


def compute_spike_times(t, s_hist):
    """Return an array of spike times where s_hist is 1."""
    return t[s_hist > 0.5]


def compare_dt_errors(dt_values, T=1.0, I_value=1.5):
    """
    For each dt and integrator, compute simple error metrics against
    a reference 'exact' simulation with very small dt_ref.
    """
    dt_ref = min(dt_values) / 10.0
    I_ref = constant_current(T, dt_ref, I_value)
    t_ref, v_ref, s_ref = simulate_lif_exact(I_ref, dt_ref)
    spike_times_ref = compute_spike_times(t_ref, s_ref)

    errors = {
        "forward": [],
        "backward": [],
        "exact": [],
    }

    for dt in dt_values:
        I_t = constant_current(T, dt, I_value)

        for name, sim_fn in [
            ("forward", simulate_lif_forward),
            ("backward", simulate_lif_backward),
            ("exact", simulate_lif_exact),
        ]:
            t, v, s = sim_fn(I_t, dt)
            spike_times = compute_spike_times(t, s)

            # Simple metrics: spike count error, first spike time error (if exists)
            spike_count_err = len(spike_times) - len(spike_times_ref)

            if len(spike_times) > 0 and len(spike_times_ref) > 0:
                first_spike_err = spike_times[0] - spike_times_ref[0]
            else:
                first_spike_err = np.nan

            errors[name].append((dt, spike_count_err, first_spike_err))

    return errors


def plot_traces_for_dt(dt, T=0.5, I_value=1.5):
    """
    Plot v(t) and spikes for all three integrators at a given dt.
    """
    I_t = constant_current(T, dt, I_value)

    results = {}
    for name, sim_fn in [
        ("forward", simulate_lif_forward),
        ("backward", simulate_lif_backward),
        ("exact", simulate_lif_exact),
    ]:
        t, v, s = sim_fn(I_t, dt)
        results[name] = (t, v, s)

    # Plot membrane potentials
    plt.figure(figsize=(8, 4))
    for name, (t, v, s) in results.items():
        plt.plot(t * 1000.0, v, label=name)  # time in ms
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential v(t)")
    plt.title(f"LIF membrane traces for dt = {dt*1000:.2f} ms")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot spike rasters
    plt.figure(figsize=(8, 2.5))
    for idx, (name, (t, v, s)) in enumerate(results.items()):
        spike_times = t[s > 0.5] * 1000.0
        plt.vlines(spike_times, idx + 0.1, idx + 0.9, label=name)
    plt.yticks([1, 2, 3], ["forward", "backward", "exact"])
    plt.xlabel("Time (ms)")
    plt.title(f"LIF spike trains for dt = {dt*1000:.2f} ms")
    plt.tight_layout()
    plt.show()


def plot_error_vs_dt(errors):
    """
    Given errors dict from compare_dt_errors, plot spike count and
    first spike time errors vs dt for each integrator.
    """
    # Spike count error
    plt.figure(figsize=(8, 4))
    for name, vals in errors.items():
        dts = [v[0] * 1000.0 for v in vals]  # ms
        spike_count_errs = [v[1] for v in vals]
        plt.plot(dts, spike_count_errs, marker="o", label=name)
    plt.xlabel("dt (ms)")
    plt.ylabel("Spike count error (vs reference)")
    plt.title("Spike count error vs dt")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # First spike timing error
    plt.figure(figsize=(8, 4))
    for name, vals in errors.items():
        dts = [v[0] * 1000.0 for v in vals]
        first_spike_errs = [
            v[2] * 1000.0 if not np.isnan(v[2]) else np.nan for v in vals
        ]  # ms
        plt.plot(dts, first_spike_errs, marker="o", label=name)
    plt.xlabel("dt (ms)")
    plt.ylabel("First spike time error (ms)")
    plt.title("First spike timing error vs dt")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # Choose a moderately suprathreshold current so the neuron fires regularly
    I_value = 1.5
    T = 1.0  # simulation duration (s)

    # Compare integrators at a single dt
    dt_single = 1e-3  # 1 ms
    plot_traces_for_dt(dt_single, T=0.5, I_value=I_value)

    # Explore how errors grow as dt increases
    dt_values = [0.1e-3, 0.5e-3, 1e-3, 2e-3, 5e-3]  # 0.1 ms to 5 ms
    errors = compare_dt_errors(dt_values, T=T, I_value=I_value)
    plot_error_vs_dt(errors)


if __name__ == "__main__":
    main()
