
### AI code, verified
import numpy as np
from openfermion import count_qubits

def gf2_rref(A):
    """
    Compute row-reduced echelon form over GF(2).
    
    Parameters
    ----------
    A : ndarray
        Input binary matrix.
        
    Returns
    -------
    R : ndarray
        RREF of A over GF(2).
    pivots : list
        List of pivot column indices.
    """
    if len(A) == 0:
        return A, []
    R = (A.copy() & 1).astype(np.uint8)
    m, n = R.shape
    pivots = []
    r = 0

    for c in range(n):
        # Find pivot row
        pivot = None
        for rr in range(r, m):
            if R[rr, c] == 1:
                pivot = rr
                break
        if pivot is None:
            continue

        # Swap into row r
        if pivot != r:
            R[[r, pivot]] = R[[pivot, r]]

        pivots.append(c)

        # Eliminate all other 1s in column c
        for rr in range(m):
            if rr != r and R[rr, c] == 1:
                R[rr, :] ^= R[r, :]

        r += 1
        if r == m:
            break

    return R, pivots


def gf2_rank(M):
    """
    Compute rank of matrix M over GF(2).
    
    Parameters
    ----------
    M : ndarray
        Input binary matrix.
        
    Returns
    -------
    int
        Rank over GF(2).
    """
    R, piv = gf2_rref(M.astype(np.uint8))
    return len(piv)


def gf2_nullspace(A):
    """
    Compute basis for nullspace of A over GF(2): {x : Ax = 0 mod 2}.
    
    Parameters
    ----------
    A : ndarray
        Input binary matrix of shape (m, n).
        
    Returns
    -------
    basis : ndarray
        Shape (k, n) basis vectors (each row is a solution vector).
    """
    R, pivots = gf2_rref(A)
    m, n = R.shape
    pivset = set(pivots)
    free_cols = [c for c in range(n) if c not in pivset]

    basis = []
    for f in free_cols:
        x = np.zeros(n, dtype=np.uint8)
        x[f] = 1

        for i, p in enumerate(pivots):
            s = 0
            row = R[i, :]
            for j in range(n):
                if j != p and row[j] == 1 and x[j] == 1:
                    s ^= 1
            x[p] = s

        basis.append(x)

    if len(basis) == 0:
        return np.zeros((0, n), dtype=np.uint8)

    return np.stack(basis, axis=0)


def gf2_check_in_nullspace(A, S):
    """
    Check if all rows of S lie in nullspace of A over GF(2).
    
    Parameters
    ----------
    A : ndarray
        Constraint matrix of shape (m, 2n).
    S : ndarray
        Matrix of candidate vectors of shape (k, 2n).
        
    Returns
    -------
    bool
        True if A @ S^T == 0 mod 2 for all rows of S.
    """
    if S is None or S.size == 0:
        return True
    prod = (A.astype(np.uint8) @ S.T.astype(np.uint8)) & 1
    return np.all(prod == 0)


def gf2_extend_basis_additive(B_current, candidates):
    """
    Extend additive basis with linearly independent candidates.
    
    Given current additive basis B_current (k, 2n) and candidate vectors
    from the current nullspace, extend B_current by adding candidates
    that increase the GF(2) rank.
    
    Parameters
    ----------
    B_current : ndarray or None
        Current basis of shape (k, 2n).
    candidates : ndarray
        Candidate vectors of shape (c, 2n).
        
    Returns
    -------
    B_new : ndarray
        Extended basis.
    added_vectors : ndarray
        Vectors that were added.
    """
    if B_current is None or B_current.size == 0:
        B = np.zeros((0, candidates.shape[1]), dtype=np.uint8)
    else:
        B = B_current.copy().astype(np.uint8)

    added = []

    # Deterministic ordering via lexicographic sort
    cand = candidates.copy().astype(np.uint8)
    if cand.shape[0] > 0:
        order = np.lexsort(cand.T[::-1])
        cand = cand[order]

    r0 = gf2_rank(B) if B.shape[0] else 0
    for v in cand:
        if B.shape[0] == 0:
            B = v.reshape(1, -1).astype(np.uint8)
            added.append(v.copy())
            r0 = 1
            continue

        r1 = gf2_rank(np.vstack([B, v]).astype(np.uint8))
        if r1 > r0:
            B = np.vstack([B, v]).astype(np.uint8)
            added.append(v.copy())
            r0 = r1

    return B, np.array(added, dtype=np.uint8)


def pauli_term_to_ax_az(term, n):
    """
    Convert OpenFermion term to symplectic (ax, az) representation.
    
    Parameters
    ----------
    term : tuple
        OpenFermion term key, e.g. ((0,'X'), (3,'Y')) or ().
    n : int
        Number of qubits.
        
    Returns
    -------
    ax : ndarray
        X-component binary vector of length n.
    az : ndarray
        Z-component binary vector of length n.
    """
    ax = np.zeros(n, dtype=np.uint8)
    az = np.zeros(n, dtype=np.uint8)

    for q, p in term:
        if p == 'X':
            ax[q] = 1
        elif p == 'Z':
            az[q] = 1
        elif p == 'Y':
            # Y = iXZ, both bits set (phase ignored for commutation)
            ax[q] = 1
            az[q] = 1
        else:
            raise ValueError(f"Unknown Pauli {p} in term {term}")

    return ax, az

def qubitop_to_G_matrix(qubit_op, n=None):
    """
    Build symplectic matrix G = (Gx | Gz) encoding Pauli terms.
    
    Parameters
    ----------
    qubit_op : QubitOperator
        Input qubit operator.
    n : int, optional
        Number of qubits. If None, inferred from operator.
        
    Returns
    -------
    G : ndarray
        Shape (m, 2n) symplectic matrix.
    coeffs : ndarray
        Complex coefficients of each term.
    labels : list
        String labels for each term.
    n : int
        Number of qubits used.
    """
    if n is None:
        n = count_qubits(qubit_op)

    items = list(qubit_op.terms.items())
    m = len(items)

    G = np.zeros((m, 2 * n), dtype=np.uint8)
    coeffs = np.zeros(m, dtype=np.complex128)
    labels = []

    for i, (term, coeff) in enumerate(items):
        ax, az = pauli_term_to_ax_az(term, n)
        G[i, :n] = ax
        G[i, n:] = az
        coeffs[i] = coeff

        if len(term) == 0:
            labels.append("I")
        else:
            labels.append(" ".join([f"{p}{q}" for (q, p) in term]))

    return G, coeffs, labels, n


def symplectic_to_pauli_string(s, n):
    """
    Convert symplectic vector to Pauli string representation.
    
    Parameters
    ----------
    s : ndarray
        Symplectic vector of length 2n (s_x | s_z).
    n : int
        Number of qubits.
        
    Returns
    -------
    str
        Pauli string like "Z0 Z2 X5" or "I" for identity.
    """
    sx = s[:n]
    sz = s[n:]

    ops = []
    for q in range(n):
        x = int(sx[q])
        z = int(sz[q])
        if x == 0 and z == 0:
            continue
        elif x == 1 and z == 0:
            ops.append(f"X{q}")
        elif x == 0 and z == 1:
            ops.append(f"Z{q}")
        else:  # x==1 and z==1
            ops.append(f"Y{q}")
    return " ".join(ops) if ops else "I"

def exchange_Gx_Gz(G, n):
    """
    Build constraint matrix A for commutation conditions.
    
    For each row g_i in G, the constraint row is (g_z,i | g_x,i) so that:
        (g_z,i | g_x,i) · (s_x | s_z) = 0 (mod 2)
    
    Parameters
    ----------
    G : ndarray
        Symplectic matrix (Gx | Gz) of shape (m, 2n).
    n : int
        Number of qubits.
        
    Returns
    -------
    A : ndarray
        Constraint matrix of shape (m, 2n).
    """
    assert np.shape(G)[1] == 2*n, "Matrix G of invalid shape {} for n = {}".format(np.shape(G), n)
    Gx = G[:, :n]
    Gz = G[:, n:]
    A = np.concatenate([Gz, Gx], axis=1).astype(np.uint8)
    return A

def gf2_find_commuting_basis(G, n_qubits):
    return gf2_nullspace(exchange_Gx_Gz(G, n_qubits))

def gf2_check_commuting(A, B, n_qubits):
    return gf2_check_in_nullspace(exchange_Gx_Gz(A, n_qubits), B)

#new
def gf2_symp_nullspace(G, n_qubits, verify=True):
    """
    Returns symplectic nullspace of G
    
    """
    H = gf2_nullspace(exchange_Gx_Gz(G, n_qubits))
    if verify:
        gf2_check_commuting(G, H, n_qubits)

    return H

def concatenate_matrices(A, B):
    """
    Stack matrices
    [ A ]
    [ - ]
    [ B ]

    """
    assert np.shape(A)[1] == np.shape(B)[1], "Incompatible matrix dimensions {} and {} for stacking! ".format(np.shape(A), np.shape(B))
    
    return np.vstack((A, B)).astype(np.int8)

def gf2_get_basis(A):
    if gf2_rank(A) < len(A):
        print("Get Basis: Matrix not generating set, reducing to a minimal basis...")
    return gf2_nullspace(gf2_nullspace(A))

def gf2_intersection(A, B, n_qubits, verify=True):
    """
    Finds generating set for common space spanned by rows of A and B
    
    """

    #reduce to basis
    A = gf2_get_basis(A)
    B = gf2_get_basis(B)

    AB = np.transpose(concatenate_matrices(A, B)) #columns are the basis
    n_A  =len(A)
    H =  np.transpose(gf2_nullspace(AB))

    x = H[:n_A]

    y = np.transpose(x) @ A % 2 # solutions in rows

    # TODO reduce y to basis
    y  = gf2_get_basis(y)

    if verify:
        assert gf2_rank(A) == gf2_rank(concatenate_matrices(A, y)) and gf2_rank(B) == gf2_rank(concatenate_matrices(B, y)), "Invalid Intersection."

    return y

def gf2_complement(A, B, verify=True):
    """
    Returns generating set for span(A)\span(B)
    
    Notes:
    NOT the same as (A \int NULL(B)) since (B \int NULL(B) \neq null) for GF2
    A\B may have linearly independent generators that can represent some elements of B, for example:
        g1, g2 \in A\B, but g1+g2 \in B

    """
    A = gf2_get_basis(A)
    B = gf2_get_basis(B)

    C = []
    for a in A:
        if gf2_rank(concatenate_matrices(B, [a])) > gf2_rank(B):
            C.append(a)
    C = np.array(C, dtype=np.uint8)

    if verify:
        
        for c in C:
            #in A
            assert gf2_rank(concatenate_matrices(A, [c])) == gf2_rank(A), "Basis not in A"
            #not in B
            assert gf2_rank(concatenate_matrices(B, [c])) > gf2_rank(B), "Basis not in A"

    return C