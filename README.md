# Leaky Integrate-and-Fire: Comparing Numerical Integrators

This repo explores simple numerical integration methods for the leaky integrate-and-fire (LIF) neuron model:

- Forward (explicit) Euler
- Backward (implicit) Euler
- Exact (exponential) integration for the LIF ODE

The aim is to show how each method behaves for different time steps `dt`, both in terms of membrane traces and spike timing.

---

## 1. Repository structure

```text
.
├── requirements.txt
├── src
│   ├── lif_forward.py   # Forward Euler LIF simulator
│   ├── lif_backward.py  # Backward Euler LIF simulator
│   └── lif_exact.py     # Exact (exponential) LIF simulator
├── experiments
│   └── compare_integrators.py  # Runs all three methods and produces plots
```

---

## 2. Setup

Create and activate a virtual environment if you like, then install dependencies:

```bash
pip install -r requirements.txt
```

Requirements are minimal:

- `numpy`
- `matplotlib`

---

## 3. Running the experiment

From the repo root:

```bash
python experiments/compare_integrators.py
```

This will:

1. Simulate a single LIF neuron driven by a constant input using all three methods.
2. Produce plots comparing:
   - Membrane potential traces over time.
   - Spike trains (raster) for each integrator.
3. Sweep a range of time steps `dt` and compute simple error metrics vs a high-resolution reference:
   - Spike count error.
   - First-spike timing error.


## 4. What each file does

### `src/lif_forward.py`

Implements a current-based LIF neuron with **forward Euler**:


$v_{n+1} = v_n + \frac{\Delta t}{\tau_m}\left( - (v_n - v_\text{rest}) + R I_n \right)$

This is explicit and easy to understand, but can become inaccurate or unstable if `dt` is too large.

### `src/lif_backward.py`

Implements **backward (implicit) Euler** for the same LIF dynamics. For this linear ODE we can solve the implicit step in closed form, giving a more stable update that allows larger `dt` without blowing up.

### `src/lif_exact.py`

Implements the **exact (exponential) integration** of the LIF equation over a time step, assuming the input is constant over `[t, t + dt]`. This uses the analytical solution of the linear ODE and is effectively the most accurate of the three for a given `dt`.

---

## 5. Interpreting the results

The main things to look for in the figures are:

- How closely the forward and backward Euler traces match the exact integration for small `dt`.
- How the error grows as `dt` increases:
  - Spike count error: do we get more or fewer spikes than in the high-resolution reference?
  - First-spike timing error: how far off is the first spike time?

This is a minimal demo, but it illustrates the tradeoff between simplicity, stability and accuracy in simulating even a very basic spiking neuron model.
