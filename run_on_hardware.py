import numpy as np
import matplotlib.pyplot as plt
import pymatching
from scipy.optimize import curve_fit
from qiskit import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveResetInZeroState
 
from surface_code import make_stim_circuit, DEFAULT_NOISE
from stim_to_qiskit import stim_to_qiskit
from extract_syndromes import extract_syndromes
from internal_helpers import counts_to_measurement_array
from aer_noise import classical_readout


TOKEN = "p8rSyk3UBMSFgEAFnAQBMfRCwOYanmb9msOtVCqWaSkBof25/uJ3TKkrE5jtG7jk"
SERVER_URL = "https://resonance.iqm.tech"


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
    stim_circ  = make_stim_circuit(distance, rounds, no_reset=no_reset, noise=None)

    #CREATE A CIRCUIT TO BE EXECUTED ON HARDWARE
    qc = []
    qc, stim_to_dense, meas_order = stim_to_qiskit(stim_circ)
    qc = PassManager([RemoveResetInZeroState()]).run(qc) # Removes the reset operations that are not mid-circuit
    #qc = make_qiskit_circuit(args)


    # OPTIONAL: IF YOU WANT TO ASSIGN QUBIT MAP BY HAND
    # Map abstract Qiskit qubits to emerald physical qubits
    qubit_map  = build_emerald_qubit_map(stim_circ)

    # HIGHLY RECOMMEND PRINTING THIS, if you are using it.
    # print(qubit_map)
    # initial_layout: Qiskit needs a list where position i = physical qubit for logical i
    n_qubits   = qc.num_qubits
    layout     = [qubit_map.get(i, i) for i in range(n_qubits)]

    # Connect to IQM Resonance
    provider   = IQMProvider(
        SERVER_URL,
        quantum_computer=quantum_computer,
        token=token or TOKEN
    )
    backend    = provider.get_backend()

    # Transpile: Qiskit transpiler will insert SWAPs for diagonal CX pairs automatically
    # think about whether this is necessary
    qc_t       = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=0)
    used = qc_t.layout.initial_index_layout(filter_ancillas=True) # the physical qubits used for data qubits
    
    print(f"Transpiled circuit depth: {qc_t.depth()}  "
          f"(ideal: {qc.depth()}, extra depth from SWAPs)")

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
    syndromes["physical_qubits"] = used                                       # (after syndromes exists)
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

        Stim circuit → Qrisp DetectorExperiment → IQMBackend → run
        → detection events + observable flips

    The DetectorExperiment stores the info the detectors and observables itself, so
    counts_to_measurement_array() and extract_syndromes() are not needed here.

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

    os.environ["IQM_TOKEN"] = token or TOKEN

    # Build circuits, exchange for make_qrisp_experiment() if desired
    stim_circ  = make_stim_circuit(distance, rounds, no_reset=no_reset, noise=None)

    # CREATE AN EXPERIMENT TO BE EXECUTED ON HARDWARE
    experiment = []
    #experiment = stim_to_detector_experiment(stim_circ, no_reset=no_reset)
    #experiment = make_qrisp_experiment(distance, rounds, no_reset=no_reset)

    # Connect to IQM Resonance
    # Layout: think about the pass_manager. iqm.qrisp_iqm.vf2pp_layout finds an
    # exact embedding on the coupling map (zero SWAPs) or fails.
    backend    = IQMBackend(
        server_url=SERVER_URL,
        device_instance=quantum_computer,
    # further arguments
    )

    # Run the experiment on the backend and collect detection events + observable flips
    # TO BE DONE
    det_events, obs_flips = [], []

    return {"det_events": det_events, "obs_flips": obs_flips}


def decode_hardware_results(syndromes: dict, rounds: int, distance=3, noise=DEFAULT_NOISE) -> tuple[float, float]:
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

    circuit  = classical_readout(make_stim_circuit(distance, rounds, "Z", no_reset=True, noise=noise))
    model    = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)

    det, obs = syndromes["det_events"], syndromes["obs_flips"].astype(bool)

    assert det.shape[1] == model.num_detectors, (
        f"Hardware has {det.shape[1]} detectors but DEM has {model.num_detectors}")
    
    pred = matching.decode_batch(det).astype(bool)
    ler  = float(np.mean(np.any(pred != obs, axis=1)))
    err  = float(np.sqrt(ler * (1 - ler) / syndromes["num_shots"]))
    return ler, err

def logical_error_rate_hardware(rounds, distance=3, shots=5_000, token=None, noise=DEFAULT_NOISE,
                       backend=None, save_prefix="syndromes"):
    rounds = np.asarray(rounds, dtype=int)
    ler, err = [], []
    for n in rounds:
        syn = run_qiskit(distance=distance, rounds=int(n), shots=shots, token=token, backend=backend)
        if save_prefix:                                           # keep the raw data: hardware time is scarce
            np.savez_compressed(f"{save_prefix}_d{distance}_r{n}.npz", **{k: np.asarray(v) for k, v in syn.items()})
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

def logical_error_rate_stim(rounds, distance=3, shots=100_000, noise=DEFAULT_NOISE, decode_noise=None,
                            seed=None, ax=None, label="Stim"):
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
    for n in rounds:
        hw  = make_stim_circuit(distance, int(n), "Z", no_reset=True, noise=None)      # defines the detectors
        sim = classical_readout(make_stim_circuit(distance, int(n), "Z", no_reset=True, noise=noise))
        raw = sim.compile_sampler(seed=seed).sample(shots)                              # stand-in for hardware bits
        syn = extract_syndromes(raw, hw, print_summary=False)
        p, e = decode_hardware_results(syn, int(n), distance, decode_noise)
        ler.append(p); err.append(e)
        print(f"rounds {n:>3}: P_error = {p:.5f} +- {e:.5f}")
    ler, err = np.array(ler), np.maximum(np.array(err), 1.0 / shots)
 
    eps, eps_err, chi2, dof, pts = _fit_eps(rounds, ler, err, shots)
    _report_and_plot(eps, eps_err, chi2, dof, pts, ax, label)
    return eps, eps_err, ler, err
