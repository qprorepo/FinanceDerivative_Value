"""
QUBO Markowitz portfolio optimisation, solved by exact brute-force
diagonalisation and by a from-scratch simulated QAOA circuit (manuscript
Sec. "QUBO Hamiltonian for portfolio optimisation", Eqs. ``qubo_hamiltonian``,
``qaoa_ansatz``).
"""

from __future__ import annotations

import itertools

import numpy as np
import scipy.optimize as sopt

__all__ = [
    "qubo_energy",
    "solve_qubo_bruteforce",
    "apply_mixer",
    "qaoa_energy",
    "build_qubo_diagonal",
    "solve_qubo_qaoa",
]


def qubo_energy(
    x_bits,
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam_r: float = 1.0,
    lam_p: float = 2.5,
    lam_c: float = 3.0,
    K0: int = 3,
) -> float:
    """Manuscript Eq. ``qubo_hamiltonian``, the Markowitz portfolio-selection
    QUBO Hamiltonian:

    .. math::
        H_{\\mathrm{QUBO}}(\\bm x) = \\lambda_r\\, \\bm x^\\top \\bm\\Sigma \\bm x
        - \\lambda_p\\, \\bm\\mu^\\top \\bm x
        + \\lambda_c\\, \\Bigl(\\textstyle\\sum_i x_i - K_0\\Bigr)^2,
        \\qquad \\bm x \\in \\{0,1\\}^K.
    """
    x = np.asarray(x_bits, dtype=float)
    risk = x @ Sigma @ x
    ret = mu @ x
    card_penalty = (x.sum() - K0) ** 2
    return lam_r * risk - lam_p * ret + lam_c * card_penalty


def solve_qubo_bruteforce(mu: np.ndarray, Sigma: np.ndarray, **kwargs):
    """Exact solution of the QUBO by enumeration over all :math:`2^K`
    bitstrings (tractable for :math:`K \\lesssim 20`).

    Returns ``(best_x, best_E, energies)`` where ``energies`` is a dict
    mapping every bitstring tuple to its energy.
    """
    K = len(mu)
    best_E, best_x = np.inf, None
    energies = {}
    for bits in itertools.product([0, 1], repeat=K):
        E = qubo_energy(bits, mu, Sigma, **kwargs)
        energies[bits] = E
        if E < best_E:
            best_E, best_x = E, bits
    return best_x, best_E, energies


def build_qubo_diagonal(mu: np.ndarray, Sigma: np.ndarray, **kwargs):
    """Builds the diagonal Hamiltonian array ``H_diag`` (length :math:`2^K`)
    and the corresponding bitstring lookup table ``bit_to_x`` (shape
    :math:`(2^K, K)`) used by the QAOA statevector simulation.
    """
    K = len(mu)
    bit_to_x = np.array([[(idx >> (K - 1 - j)) & 1 for j in range(K)] for idx in range(2**K)])
    H_diag = np.array([qubo_energy(bits, mu, Sigma, **kwargs) for bits in bit_to_x])
    return H_diag, bit_to_x


def apply_mixer(state: np.ndarray, beta: float, K: int) -> np.ndarray:
    """Applies the QAOA mixer unitary
    :math:`e^{-i\\beta H_{\\mathrm{mix}}} = \\bigotimes_j e^{-i\\beta X_j}`
    to a :math:`2^K`-dimensional statevector, via repeated single-qubit
    rotations exploiting the tensor-product structure (no explicit
    :math:`2^K\\times2^K` matrix is ever built).
    """
    Rx = np.array([[np.cos(beta), -1j * np.sin(beta)], [-1j * np.sin(beta), np.cos(beta)]])
    psi = state.reshape([2] * K)
    for j in range(K):
        psi = np.moveaxis(psi, j, 0)
        shape_rest = psi.shape[1:]
        psi = (Rx @ psi.reshape(2, -1)).reshape((2,) + shape_rest)
        psi = np.moveaxis(psi, 0, j)
    return psi.reshape(-1)


def qaoa_energy(params: np.ndarray, P: int, H_diag: np.ndarray, K: int):
    """Prepares the :math:`P`-layer QAOA state
    :math:`|\\psi(\\bm\\gamma,\\bm\\beta)\\rangle = \\prod_{p=1}^{P}
    e^{-i\\beta_p H_{\\mathrm{mix}}} e^{-i\\gamma_p H_{\\mathrm{QUBO}}}
    |{+}\\rangle^{\\otimes K}` (Eq. ``qaoa_ansatz``) starting from the
    uniform superposition, and returns
    ``(expectation_value, measurement_probabilities)``.

    ``params`` has length ``2 * P``: the first ``P`` entries are
    :math:`\\bm\\gamma`, the last ``P`` are :math:`\\bm\\beta`.
    """
    gammas, betas = params[:P], params[P:]
    psi = np.full(2**K, 1.0 / np.sqrt(2**K), dtype=complex)
    for p in range(P):
        psi = np.exp(-1j * gammas[p] * H_diag) * psi
        psi = apply_mixer(psi, betas[p], K)
    probs = np.abs(psi) ** 2
    return float(np.sum(probs * H_diag)), probs


def solve_qubo_qaoa(
    mu: np.ndarray,
    Sigma: np.ndarray,
    P: int = 4,
    seed: int = 20260906 + 81,
    maxiter: int = 300,
    qubo_kwargs: dict | None = None,
):
    """End-to-end simulated-QAOA solver: builds the diagonal QUBO
    Hamiltonian, optimises the :math:`2P` variational parameters with
    COBYLA, and returns a summary dict with the exact brute-force
    reference solution for comparison.

    Returns
    -------
    dict with keys ``best_x`` / ``best_E`` (exact ground state / energy),
    ``final_E`` (QAOA expectation value at the optimum),
    ``final_probs`` (measurement distribution),
    ``best_bitstring_qaoa`` (most probable measured bitstring),
    ``history`` (cost-function trajectory over COBYLA iterations).
    """
    qubo_kwargs = qubo_kwargs or {}
    K = len(mu)
    best_x, best_E, _ = solve_qubo_bruteforce(mu, Sigma, **qubo_kwargs)
    H_diag, bit_to_x = build_qubo_diagonal(mu, Sigma, **qubo_kwargs)

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, np.pi / 4, size=2 * P)
    history: list[float] = []

    def cost(params):
        E, _ = qaoa_energy(params, P, H_diag, K)
        return E

    def callback(params):
        history.append(cost(params))

    opt_result = sopt.minimize(
        cost, x0, method="COBYLA", callback=callback, options=dict(maxiter=maxiter, rhobeg=0.3)
    )
    final_E, final_probs = qaoa_energy(opt_result.x, P, H_diag, K)
    best_bitstring_qaoa = bit_to_x[np.argmax(final_probs)]

    return dict(
        best_x=best_x,
        best_E=best_E,
        final_E=final_E,
        final_probs=final_probs,
        bit_to_x=bit_to_x,
        best_bitstring_qaoa=best_bitstring_qaoa,
        history=history,
        opt_result=opt_result,
    )
