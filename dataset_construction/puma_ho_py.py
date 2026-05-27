"""
puma_ho_py.py

Python translation of the MATLAB PUMA-HO phase unwrapping routine.
It follows the graph-cut / min-cut formulation used in puma_ho.m.

Dependency:
    pip install PyMaxflow

Usage:
    import numpy as np
    from puma_ho_py import puma_ho

    pha_wrapped = np.angle(x_crop)
    pha_unwrapped, info = puma_ho(pha_wrapped, p=1)

Notes:
    - psi must be a 2D wrapped phase image, typically in [-pi, pi].
    - Default settings match the MATLAB code as closely as possible:
        potential.quantized = 'yes'
        potential.threshold = 0
        cliques = [[1, 0], [0, 1]]
        schedule = [1]
    - The MATLAB code uses a compiled max-flow/min-cut MEX function.
      This Python version uses PyMaxflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, Any

import numpy as np


TWOPI = 2.0 * np.pi


@dataclass
class Potential:
    quantized: str = "yes"  # "yes" or "no"
    threshold: float = 0.0


def clique_energy_ho(d: np.ndarray, p: float, threshold: float = 0.0, quantized: str = "yes") -> np.ndarray:
    """
    Python version of MATLAB clique_energy_ho.m.

    Parameters
    ----------
    d : ndarray
        Clique phase difference.
    p : float
        Power-law exponent.
    threshold : float
        Quadratic region threshold. If 0, use pure d**p.
    quantized : {"yes", "no"}
        If "yes", quantize differences to multiples of 2*pi.

    Returns
    -------
    e : ndarray
        Clique energy.
    """
    if quantized.lower() == "no":
        d_abs = np.abs(d)
    elif quantized.lower() == "yes":
        d_abs = np.abs(np.round(d / TWOPI) * TWOPI)
    else:
        raise ValueError("quantized must be 'yes' or 'no'.")

    if threshold != 0:
        mask = d_abs <= threshold
        return (threshold ** (p - 2.0)) * (d_abs ** 2) * mask + (d_abs ** p) * (~mask)

    return d_abs ** p


def _make_base(m: int, n: int, maxdesl: int) -> np.ndarray:
    """Create padded base mask exactly like the MATLAB code."""
    base = np.zeros((2 * maxdesl + 2 + m, 2 * maxdesl + 2 + n), dtype=np.float64)
    s = maxdesl + 1
    base[s:s + m, s:s + n] = 1.0
    return base


def _crop_passepartout(x: np.ndarray, maxdesl: int, m: int, n: int) -> np.ndarray:
    """Remove the zero border used by the PUMA code."""
    s = maxdesl + 1
    return x[s:s + m, s:s + n, ...]


def _roll2(x: np.ndarray, shift: Tuple[int, int]) -> np.ndarray:
    """MATLAB circshift equivalent for 2D or 3D arrays along first two axes."""
    return np.roll(np.roll(x, shift[0], axis=0), shift[1], axis=1)


def energy_ho(
    kappa: np.ndarray,
    psi: np.ndarray,
    p: float,
    cliques: np.ndarray,
    disc_bar: np.ndarray,
    threshold: float = 0.0,
    quantized: str = "yes",
) -> float:
    """
    Python version of MATLAB energy_ho.m.
    Computes the phase-unwrapping energy for the current integer labeling kappa.
    """
    psi = np.asarray(psi, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)

    m, n = psi.shape
    maxdesl = int(np.max(np.abs(cliques)))
    base = _make_base(m, n, maxdesl)

    s = maxdesl + 1
    base_kappa = np.zeros_like(base)
    psi_base = np.zeros_like(base)
    base_kappa[s:s + m, s:s + n] = kappa
    psi_base[s:s + m, s:s + n] = psi

    base_disc_bar = np.zeros(base.shape + (cliques.shape[0],), dtype=np.float64)
    base_disc_bar[s:s + m, s:s + n, :] = disc_bar

    total = 0.0
    for t, (dr, dc) in enumerate(cliques.astype(int)):
        valid = base * _roll2(base, (dr, dc))

        auxili = _roll2(base_kappa, (dr, dc))
        t_dkappa = base_kappa - auxili

        auxili2 = _roll2(psi_base, (dr, dc))
        dpsi = auxili2 - psi_base

        a = (TWOPI * t_dkappa - dpsi) * valid * base_disc_bar[:, :, t]
        total += np.sum(clique_energy_ho(a, p, threshold, quantized))

    return float(total)


def _mincut_pymaxflow(source: np.ndarray, sink: np.ndarray, edges: Iterable[Tuple[int, int, float]]) -> Tuple[float, np.ndarray]:
    """
    Max-flow/min-cut using PyMaxflow.

    Parameters
    ----------
    source, sink : 1D arrays
        Terminal capacities for each node.
    edges : iterable of (i, j, capacity)
        Directed pairwise edge from i to j. Reverse capacity is 0, matching MATLAB remain(:,4)=0.

    Returns
    -------
    flow : float
    cutside : ndarray of shape (num_nodes,)
        0 means source side, 1 means sink side.
    """
    try:
        import maxflow
    except ImportError as exc:
        raise ImportError(
            "PyMaxflow is required. Install it with: pip install PyMaxflow"
        ) from exc

    source = np.asarray(source, dtype=np.float64).ravel()
    sink = np.asarray(sink, dtype=np.float64).ravel()
    num_nodes = source.size

    # Over-allocating edge count is fine for PyMaxflow.
    g = maxflow.Graph[float](num_nodes, 2 * num_nodes)
    g.add_nodes(num_nodes)

    for i in range(num_nodes):
        cs = float(source[i])
        ct = float(sink[i])
        if cs < 0 and abs(cs) < 1e-12:
            cs = 0.0
        if ct < 0 and abs(ct) < 1e-12:
            ct = 0.0
        if cs < 0 or ct < 0:
            raise ValueError("Negative terminal capacity detected.")
        g.add_tedge(i, cs, ct)

    for i, j, cap in edges:
        cap = float(cap)
        if cap < 0 and abs(cap) < 1e-12:
            cap = 0.0
        if cap < 0:
            raise ValueError("Negative edge capacity detected. The energy may be non-submodular.")
        if cap > 0:
            # MATLAB remain uses direct capacity and inverse capacity = 0.
            g.add_edge(int(i), int(j), cap, 0.0)

    flow = float(g.maxflow())
    cutside = np.fromiter((g.get_segment(i) for i in range(num_nodes)), dtype=np.int8, count=num_nodes)
    return flow, cutside


def puma_ho(
    psi: np.ndarray,
    p: float = 1.0,
    potential: Optional[Potential | Dict[str, Any]] = None,
    cliques: Optional[np.ndarray] = None,
    qualitymaps: Optional[np.ndarray] = None,
    schedule: Optional[Iterable[int]] = None,
    max_outer_iters: int = 10_000,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Graph-cut based phase unwrapping, Python version of MATLAB puma_ho.m.

    Parameters
    ----------
    psi : ndarray, shape (M, N)
        Wrapped phase image, usually in [-pi, pi].
    p : float
        Clique potential exponent. MATLAB CCTV uses p=1.
    potential : Potential or dict, optional
        Example: Potential(quantized='yes', threshold=0.0).
    cliques : ndarray, optional
        Clique displacement vectors. Default is [[1, 0], [0, 1]].
    qualitymaps : ndarray, optional
        Shape (M, N, num_cliques). Default: zeros, i.e. no discontinuity map.
    schedule : iterable of int, optional
        Jump size schedule. Default: [1].
    max_outer_iters : int
        Safety limit for while-loop iterations.
    verbose : bool
        Print energy during optimization.

    Returns
    -------
    unwph : ndarray, shape (M, N)
        Unwrapped phase.
    info : dict
        Contains kappa, iter, erglist, and settings.
    """
    psi = np.asarray(psi, dtype=np.float64)
    if psi.ndim != 2:
        raise ValueError("psi must be a 2D wrapped phase image.")
    if not np.all(np.isfinite(psi)):
        raise ValueError("psi contains NaN or Inf.")
    if p <= 0:
        raise ValueError("p must be positive.")

    if potential is None:
        pot = Potential(quantized="yes", threshold=0.0)
    elif isinstance(potential, dict):
        pot = Potential(
            quantized=str(potential.get("quantized", "yes")),
            threshold=float(potential.get("threshold", 0.0)),
        )
    elif isinstance(potential, Potential):
        pot = potential
    else:
        raise TypeError("potential must be None, dict, or Potential.")

    if cliques is None:
        cliques = np.array([[1, 0], [0, 1]], dtype=int)
    else:
        cliques = np.asarray(cliques, dtype=int)
        if cliques.ndim != 2 or cliques.shape[1] != 2:
            raise ValueError("cliques must have shape (num_cliques, 2).")

    if schedule is None:
        schedule = [1]
    else:
        schedule = list(schedule)
        if len(schedule) == 0:
            raise ValueError("schedule must not be empty.")

    m, n = psi.shape
    num_cliques = cliques.shape[0]

    if qualitymaps is None:
        qualitymaps = np.zeros((m, n, num_cliques), dtype=np.float64)
    else:
        qualitymaps = np.asarray(qualitymaps, dtype=np.float64)
        if qualitymaps.shape != (m, n, num_cliques):
            raise ValueError("qualitymaps must have shape (M, N, num_cliques).")

    disc_bar = 1.0 - qualitymaps

    maxdesl = int(np.max(np.abs(cliques)))
    base = _make_base(m, n, maxdesl)
    s = maxdesl + 1

    kappa = np.zeros((m, n), dtype=np.float64)
    kappa_aux = kappa.copy()
    erglist = []
    iter_count = 0

    for jump_size in schedule:
        possible_improvement = True
        erg_previous = energy_ho(
            kappa, psi, p, cliques, disc_bar,
            threshold=pot.threshold, quantized=pot.quantized,
        )

        while possible_improvement:
            iter_count += 1
            if iter_count > max_outer_iters:
                raise RuntimeError("Exceeded max_outer_iters. Check input or increase max_outer_iters.")

            erglist.append(float(erg_previous))

            base_kappa = np.zeros_like(base)
            psi_base = np.zeros_like(base)
            base_kappa[s:s + m, s:s + n] = kappa
            psi_base[s:s + m, s:s + n] = psi

            source_acc = np.zeros(base.shape + (num_cliques,), dtype=np.float64)
            sink_acc = np.zeros(base.shape + (num_cliques,), dtype=np.float64)
            edge_weight_acc = np.zeros(base.shape + (num_cliques,), dtype=np.float64)
            base_start_acc = np.zeros(base.shape + (num_cliques,), dtype=np.float64)
            base_end_acc = np.zeros(base.shape + (num_cliques,), dtype=np.float64)

            for t, (dr, dc) in enumerate(cliques.astype(int)):
                base_start = _roll2(base, (-dr, -dc)) * base
                base_end = _roll2(base, (dr, dc)) * base
                valid = base * _roll2(base, (dr, dc))

                auxili = _roll2(base_kappa, (dr, dc))
                t_dkappa = base_kappa - auxili

                auxili2 = _roll2(psi_base, (dr, dc))
                dpsi = auxili2 - psi_base

                a = (TWOPI * t_dkappa - dpsi) * valid

                A = clique_energy_ho(np.abs(a), p, pot.threshold, pot.quantized) * valid
                D = A
                C = clique_energy_ho(np.abs(TWOPI * jump_size + a), p, pot.threshold, pot.quantized) * valid
                B = clique_energy_ho(np.abs(-TWOPI * jump_size + a), p, pot.threshold, pot.quantized) * valid

                pos_CA = np.maximum(C - A, 0.0)
                pos_AC = np.maximum(A - C, 0.0)
                pos_DC = np.maximum(D - C, 0.0)
                pos_CD = np.maximum(C - D, 0.0)

                source = _roll2(pos_CA, (-dr, -dc)) * base_start
                sink = _roll2(pos_AC, (-dr, -dc)) * base_start
                source = source + pos_DC * base_end
                sink = sink + pos_CD * base_end

                source_acc[:, :, t] = source
                sink_acc[:, :, t] = sink
                edge_weight_acc[:, :, t] = B + C - A - D
                base_start_acc[:, :, t] = base_start
                base_end_acc[:, :, t] = base_end

            source_c = _crop_passepartout(source_acc, maxdesl, m, n)
            sink_c = _crop_passepartout(sink_acc, maxdesl, m, n)
            edge_weight_c = _crop_passepartout(edge_weight_acc, maxdesl, m, n)
            base_start_c = _crop_passepartout(base_start_acc, maxdesl, m, n)
            base_end_c = _crop_passepartout(base_end_acc, maxdesl, m, n)

            source_final = np.sum(source_c, axis=2).ravel()
            sink_final = np.sum(sink_c, axis=2).ravel()

            edges = []
            for t in range(num_cliques):
                start_idx = np.flatnonzero(base_start_c[:, :, t].ravel() != 0)
                end_idx = np.flatnonzero(base_end_c[:, :, t].ravel() != 0)
                ew = edge_weight_c[:, :, t].ravel()

                # MATLAB uses [start endd auxiliar2(endd) 0].
                # start_idx and end_idx correspond to the same ordered list of valid pair pixels.
                if start_idx.size != end_idx.size:
                    raise RuntimeError("Internal error: start/end edge counts do not match.")

                caps = ew[end_idx]
                for i, j, cap in zip(start_idx, end_idx, caps):
                    if cap > 1e-14:
                        edges.append((int(i), int(j), float(cap)))

            _, cutside = _mincut_pymaxflow(source_final, sink_final, edges)

            kappa_aux = kappa.copy().ravel()
            kappa_flat = kappa.ravel()

            # PyMaxflow returns 0 for source side and 1 for sink side.
            # MATLAB: source side => increment by jump_size; sink side => unchanged.
            kappa_aux[:] = kappa_flat + (1 - cutside.astype(np.float64)) * jump_size
            kappa_aux = kappa_aux.reshape(m, n)

            erg_actual = energy_ho(
                kappa_aux, psi, p, cliques, disc_bar,
                threshold=pot.threshold, quantized=pot.quantized,
            )

            if verbose:
                print(
                    f"jump={jump_size}, iter={iter_count}, "
                    f"E_prev={erg_previous:.6e}, E_new={erg_actual:.6e}"
                )

            if erg_actual < erg_previous:
                erg_previous = erg_actual
                kappa = kappa_aux.copy()
            else:
                possible_improvement = False

    unwph = TWOPI * kappa + psi

    info = {
        "kappa": kappa,
        "iter": iter_count,
        "erglist": np.asarray(erglist, dtype=np.float64),
        "potential": pot,
        "cliques": cliques,
        "schedule": np.asarray(schedule),
    }
    return unwph, info


if __name__ == "__main__":
    # Minimal synthetic sanity test.
    # Real use: pass wrapped phase from np.angle(x_crop).
    yy, xx = np.mgrid[0:128, 0:128]
    true_phase = 0.08 * xx + 0.05 * yy
    wrapped_phase = np.angle(np.exp(1j * true_phase))

    unwrapped, info = puma_ho(wrapped_phase, p=1, verbose=True)
    print("Finished. iter =", info["iter"])
    print("wrapped range:", wrapped_phase.min(), wrapped_phase.max())
    print("unwrapped range:", unwrapped.min(), unwrapped.max())
