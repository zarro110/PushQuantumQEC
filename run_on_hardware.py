import numpy as np
import matplotlib.pyplot as plt
import pymatching
from scipy.optimize import curve_fit
from qiskit import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveResetInZeroState
 
from surface_code import make_stim_circuit
from stim_to_qiskit import stim_to_qiskit
from extract_syndromes import extract_syndromes
from internal_helpers import counts_to_measurements

import glob
import re


TOKEN = "p8rSyk3UBMSFgEAFnAQBMfRCwOYanmb9msOtVCqWaSkBof25/uJ3TKkrE5jtG7jk"
SERVER_URL = "https://resonance.iqm.tech"

DEFAULT_NOISE = dict(
    after_clifford_depolarization  = 0.03,
    after_reset_flip_probability   = 0.03,
    before_measure_flip_probability= 0.03,
    before_round_data_depolarization = 0.03,
)


# ─────────────────────────────────────────────────────────────────────────────
#  5.  HARDWARE EXECUTION ON IQM RESONANCE
#
#  Two equal tracks. Pick ONE. Both return the same syndromes dict
#  (at least "det_events" and "obs_flips"), so decode_hardware_results()
#  works for either.
#
#  NOTE: Emerald does not support mid-circuit reset -> use the no-reset circuit.
# ─────────────────────────────────────────────────────────────────────────────

def run_qiskit(
    distance: int    = 3,
    rounds: int      = 1,
    shots: int       = 5_000,
    token: str       = None,
    quantum_computer: str = "emerald",
    no_reset: bool   = True,
    basis: str = "Z"
) -> dict:
    """
    Build in qiskit. Runs the surface code on IQM Emerald via IQM Resonance.

        Stim circuit → Qiskit circuit → transpile → run → counts
        → raw measurements → extract_syndromes()

    Parameters
    ----------
    distance         : code distance (use 3 for emerald)
    rounds           : measurement rounds
    shots            : number of circuit executions
    token            : IQM Resonance API token (falls back to TOKEN)
    quantum_computer : e.g. "emerald"
    no_reset         : build the no-reset circuit (required on Emerald)

    Returns
    -------
    syndromes : dict  output of extract_syndromes()
                "det_events" (shots, num_detectors), "obs_flips" (shots, num_observables), ...
    """
    from iqm.qiskit_iqm import IQMProvider

    # Build circuits, exchange for make_qiskit_circuit() if desired
    stim_circ  = make_stim_circuit(distance, rounds, basis = basis, no_reset=no_reset, noise=None)

    #CREATE A CIRCUIT TO BE EXECUTED ON HARDWARE
    qc = []
    qc, stim_to_dense, meas_order = stim_to_qiskit(stim_circ)
    qc = PassManager([RemoveResetInZeroState()]).run(qc) # Removes the reset operations that are not mid-circuit
    #qc = make_qiskit_circuit(args)


    # OPTIONAL: IF YOU WANT TO ASSIGN QUBIT MAP BY HAND
    # Map abstract Qiskit qubits to emerald physical qubits
    #qubit_map  = build_emerald_qubit_map(stim_circ)

    # HIGHLY RECOMMEND PRINTING THIS, if you are using it.
    # print(qubit_map)
    # initial_layout: Qiskit needs a list where position i = physical qubit for logical i
    #n_qubits   = qc.num_qubits
    #layout     = [qubit_map.get(i, i) for i in range(n_qubits)]

    # Connect to IQM Resonance
    provider   = IQMProvider(
        SERVER_URL,
        quantum_computer=quantum_computer,
        token=TOKEN
    )
    backend    = provider.get_backend()

    # Transpile: Qiskit transpiler will insert SWAPs for diagonal CX pairs automatically
    # think about whether this is necessary
    
    # Better transpiler that takes into account the native gates of the Emerald reducing depth
    qc_t  =  transpile(qc, backend=backend, basis_gates =["r", "cz", "measure"], optimization_level=3, seed_transpiler=0)
    used = qc_t.layout.initial_index_layout(filter_ancillas=True) # the physical qubits used for data qubits
    
    #print(f"Transpiled circuit depth: {qc_t.depth()}  "
          #f"(ideal: {qc.depth()}, extra depth from SWAPs)")

    # This is better because we want to compare two qubit gates e.g. swaps
    # Depth could be coming from extra single qubit gates
    print("two-qubit gates:", qc.count_ops().get("cx", 0), "->", qc_t.count_ops().get("cz", 0),
      "(an increase means SWAPs were inserted)") 
    print("physical qubits used:", used)

    # Run
    job = backend.run(qc_t, shots=shots)
    print("job id:", job.job_id())
    counts     = job.result().get_counts()

    # Convert counts dict → (shots, num_measurements) numpy array
    raw_meas   = counts_to_measurement_array(counts, stim_circ.num_measurements, shots)
    
    # m2d converter must come from the circuit that actually ran
    syndromes = extract_syndromes(raw_meas, stim_circ, print_summary=False)   # circuit that actually ran
    return syndromes



def run_qrisp(
    distance: int    = 3,
    rounds: int      = 1,
    shots: int       = 5_000,
    token: str       = None,
    quantum_computer: str = "emerald",
    no_reset: bool   = True,
) -> dict:
    """
    Build in qrisp. Runs the surface code on IQM Emerald via IQM Resonance.

        Stim circuit → Qrisp DetectorExperiment → Compile / Backend Run
        → detection events + observable flips

    Parameters
    ----------
    distance         : code distance (use 3 for emerald)
    rounds           : measurement rounds
    shots            : number of circuit executions
    token            : IQM Resonance API token 
    quantum_computer : e.g. "emerald"
    no_reset         : build the no-reset circuit (required on Emerald)

    Returns
    -------
    syndromes : dict  same keys as extract_syndromes() from qiskit function
                "det_events" (shots, num_detectors), "obs_flips" (shots, num_observables)
    """
    import os
    from iqm.qrisp_iqm import IQMBackend
    from stim_to_qrisp import stim_to_detector_experiment
    import numpy as np

    os.environ["IQM_TOKEN"] = token or TOKEN

    # 1. Build the stim circuit
    stim_circ  = make_stim_circuit(distance, rounds, no_reset=no_reset, noise=None)

    # 2. Convert the Stim circuit into a Qrisp DetectorExperiment
    experiment = stim_to_detector_experiment(stim_circ, no_reset=no_reset)

    # 3. Connect to IQM Resonance via IQMBackend
    backend = IQMBackend(
        server_url=SERVER_URL,
        device_instance=quantum_computer,
    )

    # 4. Compile the experiment to circuits/job description and execute via backend
    # Depending on your specific stim_to_qrisp / Qrisp interface version, 
    # the experiment compiles down to quantum circuits executed via backend.run()
    compiled_circuits = experiment.compile() if hasattr(experiment, "compile") else experiment
    
    # Run execution on backend
    result = backend.run(compiled_circuits, shots=shots)
    
    # Extract detector events and observable flips from the result matching decoder formats
    # (Adapting based on whether experiment handles decoding natively or returns raw arrays)
    det_events = np.asarray(result.get("det_events", []))
    obs_flips = np.asarray(result.get("obs_flips", []))

    return {"det_events": det_events, "obs_flips": obs_flips}




def decode_hardware_results(syndromes: dict, rounds: int, distance=3, noise=DEFAULT_NOISE, basis: str = "Z") -> tuple[float, float]:
    """
    Decodes the output of run_qiskit() or run_qrisp() via your desired decoder.

    NOTE: Other parameters can be important for this function, such as a noise model.
    If you decode with a detector error model (DEM): the DEM’s detectors must match the circuit that ran.


    Parameters
    ----------
    syndromes           : dict  output of run_qiskit() / run_qrisp()

    Returns
    -------
    ler : float   estimated logical error probability for this round count (fit over several round counts to get the logical error rate, see README)
    err : float   OPTIONAL 1-sigma statistical error
    """

    # USE e.g. PyMatching to decode

    # Does decoding using Pymatcing for a single round instance
    circuit  = make_stim_circuit(distance, rounds, basis = basis, no_reset=True, noise=noise)
    model    = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)

    # syndrome gives determined and events and observed flips
    det, obs = syndromes["det_events"], syndromes["obs_flips"].astype(bool)
    
    assert det.shape[1] == model.num_detectors, (
        f"Hardware has {det.shape[1]} detectors but DEM has {model.num_detectors}")

    # Calculates logical error probability using decode_batch
    pred = matching.decode_batch(det).astype(bool)
    ler  = float(np.mean(np.any(pred != obs, axis=1)))
    err  = float(np.sqrt(ler * (1 - ler) / syndromes["num_shots"]))
    return ler, err

def _p_model(x, e):
    """P_error after x rounds for a per-round logical error rate e."""
    return 0.5 * (1 - (1 - 2 * e) ** x)
 
# Does a curve fit
def _fit_eps(rounds, ler, err, shots):
    """Weighted fit of P(n) = [1 - (1 - 2 eps)^n] / 2. Returns eps, eps_err, chi2, dof, (n, p, s) of the points used."""
    ler = np.asarray(ler)
    err = np.maximum(np.asarray(err), 1.0 / shots)
    use = ler < 0.45                                              # points near 0.5 carry almost no information
    n, p, s = rounds[use], ler[use], err[use]
    guess = float(np.clip((1 - (1 - 2 * p[0]) ** (1 / n[0])) / 2, 1e-6, 0.4))
    popt, pcov = curve_fit(_p_model, n, p, p0=[guess], sigma=s, absolute_sigma=True, bounds=(0, 0.5))
    eps, eps_err = float(popt[0]), float(np.sqrt(pcov[0, 0]))
    chi2 = float(np.sum(((p - _p_model(n, eps)) / s) ** 2))
    return eps, eps_err, chi2, len(n) - 1, (n, p, s)
 
 
def _report_and_plot(eps, eps_err, chi2, dof, pts, ax, label):
    print(f"[{label}] eps_L = {eps:.5f} +- {eps_err:.5f}   chi2/dof = {chi2:.1f}/{dof}")
    ax = plt.gca() if ax is None else ax
    n, p, s = pts
    grid = np.linspace(0, n.max() * 1.05, 200)
    data = ax.errorbar(n, p, yerr=s, fmt="o", capsize=3, label=f"{label} data")
    ax.plot(grid, _p_model(grid, eps), color=data[0].get_color(), label=f"{label} fit: eps = {eps:.4f}")
    ax.set_xlabel("rounds"); ax.set_ylabel("P_error"); ax.legend()

# Encodes pipeline for hardware qiskit setup
def logical_error_rate_hardware(rounds, distance=3, shots=5_000, token=None, noise=DEFAULT_NOISE,
                       backend=None, save_prefix="syndromes", basis: str = "Z"):
    rounds = np.asarray(rounds, dtype=int)
    ler, err = [], []
    
    # Iterate run_qiskit over number of rounds
    for n in rounds:
        syn = run_qiskit(distance=distance, rounds=int(n), shots=shots, token=token, basis = basis)
        if save_prefix:                                         # keep the raw data: hardware time is scarce
            np.savez_compressed(f"{save_prefix}_d{distance}_r{n}.npz", **{k: np.asarray(v) for k, v in syn.items()})
        p, e = decode_hardware_results(syn, int(n), distance, noise)
        ler.append(p); err.append(e)
        print(f"rounds {n:>3}: P_error = {p:.4f} +- {e:.4f}")
    ler, err = np.array(ler), np.maximum(np.array(err), 1.0 / shots)
    use = ler < 0.45                                              # points near 0.5 carry almost no information
    n, p, s = rounds[use], ler[use], err[use]

    # implicit function for fitting
    model = lambda x, e: 0.5 * (1 - (1 - 2 * e) ** x)
    guess = float(np.clip((1 - (1 - 2 * p[0]) ** (1 / n[0])) / 2, 1e-6, 0.4))

    # Use curve fit from scipy.optimize to fit.
    popt, pcov = curve_fit(model, n, p, p0=[guess], sigma=s, absolute_sigma=True, bounds=(0, 0.5))
    eps, eps_err = float(popt[0]), float(np.sqrt(pcov[0, 0]))
    chi2 = float(np.sum(((p - model(n, eps)) / s) ** 2))
    print(f"eps_L = {eps:.5f} +- {eps_err:.5f}   chi2/dof = {chi2:.1f}/{len(n) - 1}")

    # plot fit and data points
    grid = np.linspace(0, n.max() * 1.05, 200)
    plt.errorbar(n, p, yerr=s, fmt="o", capsize=3, label="data")
    plt.plot(grid, model(grid, eps), label=f"fit: eps = {eps:.4f}")
    plt.xlabel("rounds"); plt.ylabel("P_error"); plt.legend()
    return eps, eps_err, ler, err

# Encodes pipeline for hardware qrisp setup
def logical_error_rate_hardware_qrisp(rounds, distance=3, shots=5_000, token=None, noise=DEFAULT_NOISE,
                       backend=None, save_prefix="syndromes"):
    rounds = np.asarray(rounds, dtype=int)
    ler, err = [], []
    for n in rounds:
        syn = run_qrisp(distance=distance, rounds=int(n), shots=shots, token=token)
        if save_prefix:                                         # keep the raw data: hardware time is scarce
            np.savez_compressed(f"{save_prefix}_d{distance}_qr{n}.npz", **{k: np.asarray(v) for k, v in syn.items()})
        p, e = decode_hardware_results(syn, int(n), distance, noise)
        ler.append(p); err.append(e)
        print(f"rounds {n:>3}: P_error = {p:.4f} +- {e:.4f}")
    ler, err = np.array(ler), np.maximum(np.array(err), 1.0 / shots)
 
    use = ler < 0.45                                              # points near 0.5 carry almost no information
    n, p, s = rounds[use], ler[use], err[use]
    model = lambda x, e: 0.5 * (1 - (1 - 2 * e) ** x)
    guess = float(np.clip((1 - (1 - 2 * p[0]) ** (1 / n[0])) / 2, 1e-6, 0.4))
    popt, pcov = curve_fit(model, n, p, p0=[guess], sigma=s, absolute_sigma=True, bounds=(0, 0.5))
    eps, eps_err = float(popt[0]), float(np.sqrt(pcov[0, 0]))
    chi2 = float(np.sum(((p - model(n, eps)) / s) ** 2))
    print(f"eps_L = {eps:.5f} +- {eps_err:.5f}   chi2/dof = {chi2:.1f}/{len(n) - 1}")
 
    grid = np.linspace(0, n.max() * 1.05, 200)
    plt.errorbar(n, p, yerr=s, fmt="o", capsize=3, label="data")
    plt.plot(grid, model(grid, eps), label=f"fit: eps = {eps:.4f}")
    plt.xlabel("rounds"); plt.ylabel("P_error"); plt.legend()
    return eps, eps_err, ler, err

# Encodes pipeline for hardware stim simulation setup
def logical_error_rate_stim(rounds, distance=3, shots=100_000, noise=DEFAULT_NOISE, decode_noise=None,
                            seed=None, ax=None, label="Stim", basis: str = "Z"):
    """
    The same experiment and fit as logical_error_rate(), with the "hardware" data simulated in Stim.
 
    The simulated raw measurements go through the same extract_syndromes() + decode_hardware_results() as
    real data, so this also exercises the post-processing.
 
    noise        : noise of the SIMULATED device (readout error is a flipped recorded bit, like misassignment)
    decode_noise : noise the DECODER assumes. Defaults to `noise` (a perfectly known device); pass a different
                   dict to see what a mismatched decoder model costs.
    ax, label    : pass the same `ax` to both functions to overlay hardware and simulation.
    Returns eps, eps_err, ler, err.
    """
    rounds = np.asarray(rounds, dtype=int)
    decode_noise = noise if decode_noise is None else decode_noise
    ler, err = [], []

    # Simulates and calculates logical error probability for n rounds
    for n in rounds:
        hw  = make_stim_circuit(distance, int(n), basis, no_reset=True, noise=None)      # defines the detectors
        sim = make_stim_circuit(distance, int(n), basis, no_reset=True, noise=noise)
        raw = sim.compile_sampler(seed=seed).sample(shots)                              # stand-in for hardware bits
        syn = extract_syndromes(raw, hw, print_summary=False)
        p, e = decode_hardware_results(syn, int(n), distance, decode_noise)
        ler.append(p); err.append(e)
        print(f"rounds {n:>3}: P_error = {p:.5f} +- {e:.5f}")
    
    ler, err = np.array(ler), np.maximum(np.array(err), 1.0 / shots)
    # Plots and fits logical error rate
    eps, eps_err, chi2, dof, pts = _fit_eps(rounds, ler, err, shots)
    _report_and_plot(eps, eps_err, chi2, dof, pts, ax, label)
    return eps, eps_err, ler, err

def load_runs(save_prefix="syndromes", distance=3) -> dict:
    """{rounds: {key: array}} for every file `<save_prefix>_d<distance>_r<rounds>.npz`, sorted by rounds."""
    runs = {}
    # FIX: added '*' between 'r' and '.npz'
    for path in glob.glob(f"{save_prefix}_d{distance}_r*.npz"):
        n = int(re.search(r"_r(\d+)\.npz$", path).group(1))
        with np.load(path) as f:
            runs[n] = {k: f[k] for k in f.files}
        runs[n]["num_shots"] = int(runs[n]["num_shots"])
        runs[n]["num_detectors"] = int(runs[n]["num_detectors"])
    return dict(sorted(runs.items()))

# This function analyses saved round data e.g. from the
# logical_error_rate_hardware that is done using qiskit.
def analyse_saved_runs(save_prefix="syndromes", distance=3, noise=DEFAULT_NOISE,
                       label="saved data", show_diagnostics=True) -> dict:
    """Re-decode every saved run, fit eps_L, and plot. Returns a dict with the numbers."""
    runs = load_runs(save_prefix, distance)
    if not runs:
        raise FileNotFoundError(f"no files matching {save_prefix}_d{distance}_r*.npz")
    rounds = np.array(list(runs))
 
    ler, err, raw_flip = [], [], []
    for n, syn in runs.items():
        p, e = decode_hardware_results(syn, int(n), distance, noise)
        ler.append(p); err.append(e)
        raw_flip.append(float(syn["obs_flips"].mean()))              # logical flip rate BEFORE decoding
        print(f"rounds {n:>3}: shots {syn['num_shots']:>6}  decoded P_error = {p:.4f} +- {e:.4f}   "
              f"raw logical flips = {raw_flip[-1]:.4f}")
    shots = min(s["num_shots"] for s in runs.values())
    ler, err, raw_flip = np.array(ler), np.maximum(np.array(err), 1.0 / shots), np.array(raw_flip)
 
    sets = {tuple(int(q) for q in s["physical_qubits"]) for s in runs.values() if "physical_qubits" in s}
    if len(sets) > 1:
        print(f"WARNING: the runs used {len(sets)} different sets of physical qubits; the round-dependence "
              f"is then mixed with hardware differences. Pass a fixed initial_layout.")
 
    eps, eps_err, chi2, dof, pts = _fit_eps(rounds, ler, err, shots)
 
    if show_diagnostics:
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
        _report_and_plot(eps, eps_err, chi2, dof, pts, axs[0], label)
        axs[0].plot(rounds, raw_flip, "s", mfc="none", color="gray", label="raw (undecoded) flips")
        axs[0].legend(); axs[0].set_title("logical error vs rounds")
 
        rate = []                                                     # interior rounds only: the first and last
        for n, s in runs.items():                                     # round have different detectors and rates
            c = make_stim_circuit(distance, int(n), "Z", no_reset=True, noise=None).get_detector_coordinates()
            tt = np.array([c[i][2] for i in range(len(c))])
            rate.append(float(np.asarray(s["detector_firing_rate"])[(tt > 0) & (tt < n)].mean()))
        axs[1].plot(rounds, rate, "o-")
        axs[1].set_xlabel("rounds"); axs[1].set_ylabel("mean firing rate, interior rounds")
        axs[1].set_title("drift check (flat = stable)")
 
        n_max = int(rounds.max())                                     # time profile inside the longest run
        coords = make_stim_circuit(distance, n_max, "Z", no_reset=True, noise=None).get_detector_coordinates()
        t = np.array([coords[i][2] for i in range(len(coords))])
        fr = np.asarray(runs[n_max]["detector_firing_rate"])
        axs[2].plot(sorted(set(t)), [fr[t == k].mean() for k in sorted(set(t))], "o-")
        axs[2].set_xlabel("round index"); axs[2].set_ylabel("mean detector firing rate")
        axs[2].set_title(f"inside the {n_max}-round run")
        fig.tight_layout()
 
    return dict(rounds=rounds, ler=ler, err=err, raw_flip=raw_flip, eps=eps, eps_err=eps_err,
                chi2=chi2, dof=dof, physical_qubit_sets=sets)
