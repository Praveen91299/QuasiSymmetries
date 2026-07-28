"""
Code to determine Fiedler vector and optimal ordering for an arbitrary state
June 9 - AI generated, verified by PJ
"""

import numpy as np
from scipy.linalg import eigh


def infer_n_qubits_from_state(psi: np.ndarray) -> int:
    """
    Infer number of qubits from a statevector.
    """
    dim = psi.size
    n = int(round(np.log2(dim)))

    if 2**n != dim:
        raise ValueError("State dimension is not a power of 2.")

    return n


def reduced_density_matrix_statevector(
    psi: np.ndarray,
    keep: list[int],
    n_qubits: int = None,
) -> np.ndarray:
    """
    Reduced density matrix for a subset of qubits from a pure statevector.

    Convention:
        The statevector is reshaped as psi.reshape([2] * n_qubits).
        Qubit 0 is the first tensor axis.

        Flat index = q_0 * 2^(n-1) + q_1 * 2^(n-2) + ... + q_{n-1}.
    """
    psi = np.asarray(psi, dtype=complex).reshape(-1)

    if n_qubits is None:
        n_qubits = infer_n_qubits_from_state(psi)

    keep = list(keep)

    if len(set(keep)) != len(keep):
        raise ValueError("Duplicate qubits in keep.")

    if any(q < 0 or q >= n_qubits for q in keep):
        raise ValueError("Qubit index out of range.")

    trace_out = [q for q in range(n_qubits) if q not in keep]

    psi_t = psi.reshape([2] * n_qubits)

    perm = keep + trace_out
    psi_perm = np.transpose(psi_t, perm)

    dim_keep = 2 ** len(keep)
    dim_trace = 2 ** len(trace_out)

    psi_mat = psi_perm.reshape(dim_keep, dim_trace)

    rho = psi_mat @ psi_mat.conj().T
    rho = 0.5 * (rho + rho.conj().T)

    return rho


def von_neumann_entropy(
    rho: np.ndarray,
    base: float = 2.0,
    tol: float = 1e-12,
) -> float:
    """
    Von Neumann entropy S(rho) = -Tr[rho log(rho)].
    """
    rho = np.asarray(rho, dtype=complex)
    rho = 0.5 * (rho + rho.conj().T)

    evals = np.linalg.eigvalsh(rho)
    evals = np.real(evals)

    # Remove tiny numerical negatives and zero eigenvalues.
    evals[np.abs(evals) < tol] = 0.0
    evals = evals[evals > tol]

    if evals.size == 0:
        return 0.0

    logs = np.log(evals)

    if base is not None:
        logs /= np.log(base)

    return float(-np.sum(evals * logs))


def qubit_mutual_information_matrix(
    psi: np.ndarray,
    n_qubits: int = None,
    base: float = 2.0,
    convention: str = "standard",
    entropy_tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute qubit mutual information matrix from a pure statevector.

    I_ij = s_i + s_j - s_ij

    Parameters
    ----------
    psi
        Approximate ground-state vector.
    n_qubits
        Number of qubits. If None, inferred from psi.
    base
        Logarithm base for entropy. base=2 gives entropy in bits.
    convention
        "standard": I_ij = s_i + s_j - s_ij.
        "half":     I_ij = 0.5 * (s_i + s_j - s_ij).
    entropy_tol
        Eigenvalue cutoff used in entropy calculation.

    Returns
    -------
    I
        Mutual information matrix.
    s1
        One-qubit entropies.
    s2
        Two-qubit entropies.
    """
    psi = np.asarray(psi, dtype=complex).reshape(-1)

    norm = np.linalg.norm(psi)
    if norm == 0:
        raise ValueError("Input state has zero norm.")
    psi = psi / norm

    if n_qubits is None:
        n_qubits = infer_n_qubits_from_state(psi)

    if convention not in {"standard", "half"}:
        raise ValueError("convention must be 'standard' or 'half'.")

    factor = 0.5 if convention == "half" else 1.0

    s1 = np.zeros(n_qubits, dtype=float)

    for i in range(n_qubits):
        rho_i = reduced_density_matrix_statevector(psi, [i], n_qubits)
        s1[i] = von_neumann_entropy(rho_i, base=base, tol=entropy_tol)

    s2 = np.zeros((n_qubits, n_qubits), dtype=float)
    I = np.zeros((n_qubits, n_qubits), dtype=float)

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            rho_ij = reduced_density_matrix_statevector(psi, [i, j], n_qubits)
            sij = von_neumann_entropy(rho_ij, base=base, tol=entropy_tol)

            s2[i, j] = s2[j, i] = sij

            mij = factor * (s1[i] + s1[j] - sij)

            # Remove tiny numerical negative values.
            if mij < 0.0 and abs(mij) < 1e-10:
                mij = 0.0

            I[i, j] = I[j, i] = max(0.0, float(mij))

    return I, s1, s2


def qubit_mutual_information_matrix_sparse_bloch(
    state,
    n_qubits: int = None,
    base: float = 2.0,
    convention: str = "standard",
    entropy_tol: float = 1e-12,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute qubit mutual information using sparse Pauli-expectation/Bloch RDMs.

    This avoids constructing reduced density matrices by tracing over a dense
    statevector.  It reconstructs the one- and two-qubit RDMs from expectation
    values of one- and two-qubit Pauli operators.
    """
    from .state_utils import SparseQubitState

    if isinstance(state, SparseQubitState):
        sparse_state = state.copy().normalize()
    else:
        sparse_state = SparseQubitState.from_dense(
            state,
            n_qubits=n_qubits,
            threshold=threshold,
        ).normalize()

    if n_qubits is not None and sparse_state.n_qubits != int(n_qubits):
        raise ValueError("n_qubits does not match sparse state.")

    return sparse_state.qubit_mutual_information_matrix(
        base=base,
        convention=convention,
        entropy_tol=entropy_tol,
    )


def sanitize_weight_matrix(W: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    """
    Symmetrize, remove diagonal, and clip tiny negative values.
    """
    W = np.asarray(W, dtype=float)

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("Weight matrix must be square.")

    W = 0.5 * (W + W.T)

    W[np.abs(W) < tol] = 0.0

    if np.min(W) < -tol:
        raise ValueError("Weight matrix has significantly negative entries.")

    W = np.maximum(W, 0.0)
    np.fill_diagonal(W, 0.0)

    return W


def connected_components_from_weights(
    W: np.ndarray,
    edge_tol: float = 1e-12,
) -> list[list[int]]:
    """
    Connected components of an undirected weighted graph.

    Edges with weight > edge_tol are considered present.
    """
    W = sanitize_weight_matrix(W)

    n = W.shape[0]
    seen = np.zeros(n, dtype=bool)
    components = []

    adjacency = [list(np.where(W[i] > edge_tol)[0]) for i in range(n)]

    for start in range(n):
        if seen[start]:
            continue

        stack = [start]
        seen[start] = True
        component = []

        while stack:
            i = stack.pop()
            component.append(i)

            for j in adjacency[i]:
                if not seen[j]:
                    seen[j] = True
                    stack.append(j)

        components.append(sorted(component))

    return components


def graph_laplacian(W: np.ndarray) -> np.ndarray:
    """
    Combinatorial graph Laplacian L = D - W.
    """
    W = sanitize_weight_matrix(W)
    degrees = np.sum(W, axis=1)
    return np.diag(degrees) - W


def fiedler_order_connected_component(
    W_sub: np.ndarray,
    nodes: list[int],
    tie_break: str = "index",
    eig_tol: float = 1e-10,
) -> dict:
    """
    Fiedler ordering for a single connected component.

    Parameters
    ----------
    W_sub
        Weight matrix restricted to one connected component.
    nodes
        Original node labels corresponding to rows/columns of W_sub.
    tie_break
        "index" or "degree".
    eig_tol
        Tolerance for diagnostics.

    Returns
    -------
    Dictionary with component ordering and diagnostics.
    """
    W_sub = sanitize_weight_matrix(W_sub)
    nodes = list(nodes)
    m = len(nodes)

    if W_sub.shape != (m, m):
        raise ValueError("W_sub shape does not match number of nodes.")

    degrees = np.sum(W_sub, axis=1)
    L = np.diag(degrees) - W_sub

    if m == 1:
        return {
            "nodes": nodes,
            "ordering": nodes.copy(),
            "reverse_ordering": nodes.copy(),
            "old_to_new_component": {nodes[0]: 0},
            "fiedler_vector": np.array([0.0]),
            "fiedler_coordinates": {nodes[0]: 0.0},
            "laplacian_eigenvalues": np.array([0.0]),
            "fiedler_eigenvalue": 0.0,
            "degrees": {nodes[0]: 0.0},
            "total_weight": 0.0,
            "is_singleton": True,
            "is_nearly_degenerate": False,
        }

    evals, evecs = eigh(L)
    evals = np.real(evals)

    # For a connected component, index 0 is the constant vector.
    # The Fiedler vector is index 1.
    fiedler_index = 1
    f = np.real(evecs[:, fiedler_index])
    fiedler_eigenvalue = float(evals[fiedler_index])

    # Diagnostics: lambda_2 approximately equal to lambda_3.
    is_nearly_degenerate = False
    if m > 2:
        gap = evals[2] - evals[1]
        is_nearly_degenerate = bool(abs(gap) < eig_tol)

    if tie_break == "index":
        local_order = sorted(range(m), key=lambda a: (f[a], nodes[a]))
    elif tie_break == "degree":
        local_order = sorted(range(m), key=lambda a: (f[a], -degrees[a], nodes[a]))
    else:
        raise ValueError("tie_break must be 'index' or 'degree'.")

    ordering = [nodes[a] for a in local_order] # NOTE - local relatively reordered in global positions
    reverse_ordering = list(reversed(ordering))

    old_to_new_component = {
        old_node: new_pos for new_pos, old_node in enumerate(ordering)
    } ### NOTE - new_pos is relative to local component

    return {
        "nodes": nodes,
        "ordering": ordering,
        "reverse_ordering": reverse_ordering,
        "old_to_new_component": old_to_new_component,
        "fiedler_vector": f,
        "fiedler_coordinates": {nodes[i]: float(f[i]) for i in range(m)},
        "laplacian_eigenvalues": evals,
        "fiedler_eigenvalue": fiedler_eigenvalue,
        "degrees": {nodes[i]: float(degrees[i]) for i in range(m)},
        "total_weight": float(np.sum(W_sub) / 2.0),
        "is_singleton": False,
        "is_nearly_degenerate": is_nearly_degenerate,
    }


def order_components(
    component_infos: list[dict],
    component_order: str = "total_weight",
) -> list[dict]:
    """
    Sort connected components before concatenating their internal orderings.
    """
    infos = list(component_infos)

    if component_order == "total_weight":
        infos.sort(
            key=lambda r: (
                -r["total_weight"],
                -len(r["nodes"]),
                min(r["nodes"]),
            )
        )
    elif component_order == "size":
        infos.sort(
            key=lambda r: (
                -len(r["nodes"]),
                -r["total_weight"],
                min(r["nodes"]),
            )
        )
    elif component_order == "index":
        infos.sort(key=lambda r: min(r["nodes"]))
    else:
        raise ValueError(
            "component_order must be 'total_weight', 'size', or 'index'."
        )

    return infos


def invert_ordering(ordering: list[int], n_total: int = None) -> list[int]:
    """
    Convert ordering from new_position -> old_qubit
    to old_qubit -> new_position.

    If n_total is given, output has length n_total and entries not present in
    ordering are None.
    ### NOTE old_to_new[i] gives new position of ith qubit
    """
    if n_total is None:
        n_total = max(ordering) + 1 if ordering else 0

    old_to_new = [None] * n_total

    for new_pos, old_qubit in enumerate(ordering):
        old_to_new[old_qubit] = new_pos

    return old_to_new 


def fiedler_order_from_weights(
    W: np.ndarray,
    nodes: list[int] = None,
    edge_tol: float = 1e-12,
    eig_tol: float = 1e-10,
    tie_break: str = "index",
    component_order: str = "total_weight",
) -> dict:
    """
    Component-aware Fiedler ordering from a weighted graph.

    This function always checks connected components first.
    Fiedler ordering is applied only within each connected component.

    Parameters
    ----------
    W
        Full symmetric nonnegative weighted adjacency matrix.
    nodes
        Optional subset of nodes to order. If None, order all nodes.
    edge_tol
        Edges with weight <= edge_tol are treated as absent.
    eig_tol
        Tolerance used for near-degeneracy diagnostics.
    tie_break
        Tie-breaking rule inside connected components:
        "index" or "degree".
    component_order
        How to concatenate disconnected components:
        "total_weight", "size", or "index".

    Returns
    -------
    result
        ordering:
            list giving new_position -> old_qubit.
        old_to_new:
            list giving old_qubit -> new_position.
    """
    W = sanitize_weight_matrix(W)

    n_total = W.shape[0]

    if nodes is None:
        nodes = list(range(n_total))
    else:
        nodes = list(nodes)

    if len(set(nodes)) != len(nodes):
        raise ValueError("Duplicate nodes.")

    if any(i < 0 or i >= n_total for i in nodes):
        raise ValueError("Node index out of range.")

    W_nodes = W[np.ix_(nodes, nodes)]

    components_local = connected_components_from_weights(
        W_nodes,
        edge_tol=edge_tol,
    )

    component_infos = []

    for comp_local in components_local:
        comp_nodes = [nodes[a] for a in comp_local]
        W_comp = W[np.ix_(comp_nodes, comp_nodes)]

        info = fiedler_order_connected_component(
            W_comp,
            nodes=comp_nodes,
            tie_break=tie_break,
            eig_tol=eig_tol,
        )

        component_infos.append(info)

    component_infos = order_components(
        component_infos,
        component_order=component_order,
    )

    ordering = []
    for info in component_infos:
        ordering.extend(info["ordering"])

    old_to_new = invert_ordering(ordering, n_total=n_total)

    return {
        "ordering": ordering,                   # new_position -> old_qubit
        "old_to_new": old_to_new,               # old_qubit -> new_position
        "reverse_ordering": list(reversed(ordering)),
        "components": component_infos,
        "nodes": nodes,
        "weight_matrix": W,
        "weight_matrix_subspace": W_nodes,
        "edge_tol": edge_tol,
        "tie_break": tie_break,
        "component_order": component_order,
    }


def fiedler_order_from_state(
    psi: np.ndarray,
    n_qubits: int  = None,
    nodes: list[int] = None,
    base: float = 2.0,
    mutual_info_convention: str = "standard",
    entropy_tol: float = 1e-12,
    edge_tol: float = 1e-12,
    eig_tol: float = 1e-10,
    tie_break: str = "index",
    component_order: str = "total_weight",
    mutual_information_method: str = "statevector",
    sparse_threshold: float = 0.0,
) -> dict:
    """
    Full pipeline:
        statevector -> qubit mutual information -> component-aware Fiedler order.

    Parameters
    ----------
    psi
        Approximate ground-state vector.
    n_qubits
        Number of qubits. If None, inferred from psi.
    nodes
        Optional subset of qubits to order.
    base
        Entropy logarithm base.
    mutual_info_convention
        "standard" or "half".
    entropy_tol
        Entropy eigenvalue cutoff.
    edge_tol
        Mutual-information values <= edge_tol are treated as absent edges.
    eig_tol
        Near-degeneracy diagnostic tolerance.
    tie_break
        "index" or "degree".
    component_order
        "total_weight", "size", or "index".

    Returns
    -------
    result
        Dictionary containing ordering and diagnostics.
    """
    if mutual_information_method not in {"statevector", "sparse_bloch"}:
        raise ValueError(
            "mutual_information_method must be 'statevector' or 'sparse_bloch'."
        )

    if mutual_information_method == "sparse_bloch":
        from .state_utils import SparseQubitState

        if isinstance(psi, SparseQubitState):
            if n_qubits is None:
                n_qubits = psi.n_qubits
        else:
            psi = np.asarray(psi, dtype=complex).reshape(-1)
            if n_qubits is None:
                n_qubits = infer_n_qubits_from_state(psi)
        I, s1, s2 = qubit_mutual_information_matrix_sparse_bloch(
            psi,
            n_qubits=n_qubits,
            base=base,
            convention=mutual_info_convention,
            entropy_tol=entropy_tol,
            threshold=sparse_threshold,
        )
    else:
        psi = np.asarray(psi, dtype=complex).reshape(-1)
        if n_qubits is None:
            n_qubits = infer_n_qubits_from_state(psi)
        I, s1, s2 = qubit_mutual_information_matrix(
            psi,
            n_qubits=n_qubits,
            base=base,
            convention=mutual_info_convention,
            entropy_tol=entropy_tol,
        )

    info = fiedler_order_from_weights(
        I,
        nodes=nodes,
        edge_tol=edge_tol,
        eig_tol=eig_tol,
        tie_break=tie_break,
        component_order=component_order,
    )

    info["mutual_information"] = I
    info["one_qubit_entropies"] = s1
    info["two_qubit_entropies"] = s2
    info["n_qubits"] = n_qubits
    info["mutual_info_convention"] = mutual_info_convention
    info["mutual_information_method"] = mutual_information_method
    info["entropy_base"] = base

    return info


def describe_ordering(ordering: list[int]) -> list[int]:
    """
    Print and return inverse ordering.

    ordering:
        new_position -> old_qubit
    old_to_new:
        old_qubit -> new_position
    """
    print("ordering: new_position -> old_qubit")
    for new_pos, old_qubit in enumerate(ordering):
        print(f"  new site {new_pos} <- old qubit {old_qubit}")


def reorder_statevector_axes(
    psi: np.ndarray,
    ordering: list[int],
    n_qubits: int = None,
) -> np.ndarray:
    """
    Reorder a statevector according to ordering = new_position -> old_qubit.

    If ordering = [1, 2, 0], the new tensor axes are [old q1, old q2, old q0].
    """
    psi = np.asarray(psi, dtype=complex).reshape(-1)

    if n_qubits is None:
        n_qubits = infer_n_qubits_from_state(psi)

    if sorted(ordering) != list(range(n_qubits)):
        raise ValueError("ordering must be a permutation of all qubits.")

    psi_tensor = psi.reshape([2] * n_qubits)
    psi_reordered = np.transpose(psi_tensor, ordering).reshape(-1)

    return psi_reordered

from .clifford_symmetry_optimized import permute_qubits_in_qubit_operator
from .metrics import get_entropies_at_cuts

def do_fiedler_reordering(
    HQ,
    psi,
    n_qubits,
    verbose=True,
    component_order="index",
    log_base=np.e,
    mutual_information_method="statevector",
    sparse_threshold=0.0,
    sparse_cut_max_dense_dim=4096,
):
    """
    Determine a Fiedler qubit ordering from ``psi`` and apply it to ``HQ``.

    Parameters
    ----------
    HQ
        OpenFermion ``QubitOperator`` Hamiltonian to permute.
    psi
        State used to build the qubit mutual-information graph.  This can be a
        dense statevector or a ``SparseQubitState``.
    n_qubits
        Number of qubits.
    verbose
        If True, print the one-qubit entropies, mutual-information matrix,
        connected components, final ordering, and reordered cut entropies.
    component_order
        How disconnected mutual-information components are stitched together;
        passed to ``fiedler_order_from_state``.  Common choices are ``"index"``,
        ``"total_weight"``, and ``"size"``.
    log_base
        Logarithm base for entropy and mutual information.  The historical
        default here is ``np.e``.
    mutual_information_method
        ``"statevector"`` uses the original dense reduced-density-matrix path.
        ``"sparse_bloch"`` uses sparse Pauli expectation values to reconstruct
        one- and two-qubit Bloch/RDM data.  For determinant-sparse states, use
        this together with a ``SparseQubitState`` input to avoid materializing
        the full ``2**n_qubits`` statevector.
    sparse_threshold
        When ``mutual_information_method="sparse_bloch"`` is requested with a
        dense input state, amplitudes with absolute value <= this threshold are
        discarded while converting to ``SparseQubitState``.
    sparse_cut_max_dense_dim
        Safety limit for the sparse reordered cut-entropy calculation.  The
        sparse path groups amplitudes by observed left/right bit patterns at
        each cut and diagonalizes the smaller Gram matrix.  If that Gram matrix
        would exceed this dimension, a ``ValueError`` is raised rather than
        accidentally allocating a very large dense matrix.  Increase this only
        intentionally.

    Feasible sparse usage
    ---------------------
    For CISD/determinant-sparse states, prefer::

        sparse_psi = SparseQubitState.from_dense(psi, n_qubits, threshold=tol)
        ent, HQ_reord, psi_reord, info = do_fiedler_reordering(
            HQ,
            sparse_psi,
            n_qubits,
            mutual_information_method="sparse_bloch",
        )

    In this mode, both the mutual-information graph and the returned reordered
    cut entropies are computed without constructing the full dense statevector.
    The returned ``psi_reord`` is also a ``SparseQubitState``.  Dense inputs keep
    the original dense return behavior unless ``"sparse_bloch"`` is explicitly
    requested.

    Returns
    -------
    ent_reord
        Entanglement entropies across contiguous cuts after reordering.
    HQ_reord
        Hamiltonian with qubits permuted according to the Fiedler ordering.
    psi_reord
        Reordered state.  Dense input returns a dense statevector; sparse input
        returns a ``SparseQubitState``.
    fiedler_info
        Diagnostics from ``fiedler_order_from_state``, including ordering,
        mutual-information matrix, components, and entropy data.
    """

    fiedler_info = fiedler_order_from_state(
        psi,
        n_qubits,
        component_order=component_order,
        base=log_base,
        mutual_information_method=mutual_information_method,
        sparse_threshold=sparse_threshold,
    )
    if verbose:

        print("One-qubit entropies:")
        print(fiedler_info["one_qubit_entropies"])

        print("\nMutual information matrix:")
        print(fiedler_info["mutual_information"])

        print("\nConnected components:")
        for comp in fiedler_info["components"]:
            print("  nodes:", comp["nodes"], "ordering:", comp["ordering"])

        print("\nFinal ordering (old qubit positions):")
        print(fiedler_info["ordering"])
        describe_ordering(fiedler_info["ordering"])

    if hasattr(psi, "reorder_qubits") and hasattr(psi, "entropies_at_cuts"):
        psi_reord = psi.reorder_qubits(fiedler_info["ordering"])
        ent_reord = psi_reord.entropies_at_cuts(
            log_base=log_base,
            max_dense_dim=sparse_cut_max_dense_dim,
        )
    elif mutual_information_method == "sparse_bloch":
        from .state_utils import SparseQubitState

        psi_reord = SparseQubitState.from_dense(
            psi,
            n_qubits=n_qubits,
            threshold=sparse_threshold,
        ).reorder_qubits(fiedler_info["ordering"])
        ent_reord = psi_reord.entropies_at_cuts(
            log_base=log_base,
            max_dense_dim=sparse_cut_max_dense_dim,
        )
    else:
        psi_reord = reorder_statevector_axes(psi, fiedler_info["ordering"], n_qubits)
        ent_reord = get_entropies_at_cuts(psi_reord, n_qubits, log_base=log_base)

    perm = invert_ordering(fiedler_info["ordering"])
    HQ_reord = permute_qubits_in_qubit_operator(HQ, perm)

    if verbose:
        print("\nReordered state entanglement:")
        for i, e in enumerate(ent_reord):
            print(i+1, i+2, e)

    return ent_reord, HQ_reord, psi_reord, fiedler_info
