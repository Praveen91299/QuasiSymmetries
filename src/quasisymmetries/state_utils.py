# HF, CISD, FCI stuff

import numpy as np
from scipy.sparse import csr_matrix
#from pyscf.fci import cistring
import openfermion as of
from openfermion import MolecularData
from scipy.linalg import eigh
import scipy as sp
import math
from dataclasses import dataclass
#from pyscf import fci


def _parity_array(values):
    """Return parity of integer bit counts as a boolean NumPy array."""
    values = np.asarray(values, dtype=np.uint64).reshape(-1).copy()
    # Fold every 64-bit word down to four bits, then use the hexadecimal
    # parity lookup 0x6996.  Keeping this operation in NumPy avoids one Python
    # ``bin(...).count`` call per determinant during repeated Pauli actions.
    values ^= values >> np.uint64(32)
    values ^= values >> np.uint64(16)
    values ^= values >> np.uint64(8)
    values ^= values >> np.uint64(4)
    return ((np.uint64(0x6996) >> (values & np.uint64(0xF))) & 1).astype(bool)

def to_str(occ_list):
    st = ''
    for occ in occ_list:
        st += str(occ)
    
    return st

def get_hf_occ(n_electrons, n_orbitals, spin_ord = 'udud', remove_qubit_loc = [], as_str=False):
    '''
    List slater determinant of HF
    '''
    hf = [1]*n_electrons + [0]*(2*n_orbitals - n_electrons)
    if spin_ord == 'uudd':
        hf = hf[::2] + hf[1::2]
    
    hf_f = []
    for i, a in enumerate(hf):
        if i not in remove_qubit_loc:
            hf_f.append(a)
    
    if as_str:
        return to_str(occ_list=hf_f)
    else:
        return hf_f

def get_hf_wfn(occ):
    wfn = [1.0]
    for i in occ:
        if i == 1:
            wfn = np.kron(wfn, [0, 1])
        else:
            wfn = np.kron(wfn, [1, 0])
    return wfn

### CISD
### Choi's code
#
# def get_gs(mol, op):
#     values, vectors = eigh(op.toarray())

#     order = np.argsort(values)
#     values = values[order]
#     vectors = vectors[:, order]
#     print(values)
#     if mol == 'ch2':
#         eigenvalue = values[3]
#         eigenstate = vectors[:, 3]
#     elif mol == 'h2ost2':
#         eigenvalue = values[5]
#         eigenstate = vectors[:, 5]
#     else:
#         eigenvalue = values[0]
#         eigenstate = vectors[:, 0]

#     return eigenvalue, eigenstate.T

def get_gs(op):
    """
    Returns gs energy and state of a given matrix

    """
    values, vectors = eigh(op.toarray())

    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    #print(values)
    
    eigenvalue = values[0]
    eigenstate = vectors[:, 0]

    return eigenvalue, eigenstate.T


def partial_order(x, y):
    """
    As described in arXiv:quant-ph/0003137 pg.10, computes the if x <= y where <= is a partial order and x and y are binary strings (but inputted as integers).
    Args:
        x, y (int): Integers that will be converted to binary to then check x <= y.

    Returns:
        partial_order(bool): Whether x <= y

    """
    if x > y:
        return False

    else:
        x_b, y_b = format(x, 'b'), format(y, 'b')

        if len(x_b) != len(y_b):
            while len(x_b) != len(y_b):
                x_b = '0' + x_b

        length = len(x_b)

        partial_order = False
        for l0 in range(length):
            if x_b[0:l0] == y_b[0:l0] and y_b[l0:length] == (length - l0)*'1':
                partial_order = True
                break

        return partial_order

def get_bk_tf_matrix(n_qubits):
    """
    Implementation from arXiv:quant-ph/0003137 and https://doi.org/10.1021/acs.jctc.8b00450. Given some reference occupation no's in the fermionic space, find the corresponding BK basis state in the qubit space.
    Args:
        n_qubits (int): No. of qubits
    Returns:
        tf_mat (np.array): Transformation matrix that converts fermionic occupation numbers to BK transformed basis vectors.
    """

    tf_mat = np.zeros((n_qubits, n_qubits))

    for i in range(n_qubits):
        if np.mod(i, 2) == 0:
            tf_mat[i, i] = 1
        elif np.mod(math.log(i+1, 2), 1) == 0:
            for j in range(i+1):
                tf_mat[i, j] = 1
        else:
            for j in range(n_qubits):
                if partial_order(j, i) == True:
                    tf_mat[i, j] = 1

    return tf_mat

def get_bk_basis_states(occ_no, n_qubits):
    """
    Implementation from arXiv:quant-ph/0003137 and https://doi.org/10.1021/acs.jctc.8b00450. Given some reference occupation no's in the fermionic space, find the corresponding BK basis state in the qubit space.
    Args:
        occ_no_list (List[str]): List of occupation number vectors. Occ no. vectors ordered from left to right going from 0 -> n-1 in terms of orbitals.
    Returns:
        basis_state (np.array): Basis vector in (BK transformed) qubit space corresponding to occ_no_state.
    """

    tf_mat = get_bk_tf_matrix(n_qubits)

    occ_no_vec = np.array(list(occ_no), dtype = int)
    qubit_state = np.mod(np.matmul(tf_mat, occ_no_vec), 2)

    return qubit_state

def get_jw_basis_states(occ_no_list, n_qubits):
    """
    Implementation from arXiv:quant-ph/0003137 and https://doi.org/10.1021/acs.jctc.8b00450. Given some reference occupation no's in the fermionic space, find the corresponding BK basis state in the qubit space.
    Args:
        occ_no_list (List[str]): List of occupation number vectors. Occ no. vectors ordered from left to right going from 0 -> n-1 in terms of orbitals.
    Returns:
        basis_state (np.array): Basis vector in (JW transformed) qubit space corresponding to occ_no_state.
    """

    jw_list = []
    for occ_no in occ_no_list:
        qubit_state = np.array(list(occ_no), dtype = int)
        jw_list.append(qubit_state)

    return jw_list


def find_index(basis_state):
    """
    Given some qubit/fermionic basis state, find the index of the a wavefunction that corresponds to that array.
    Args:
        basis_state (str or list/np.array): Occupation number vector/ Qubit basis state. If str, ordered from left to right going from 0 -> n-1 in terms of orbitals/qubits.
    Returns:
        index (int): Index of the basis in total Qubit space.
    """
    index = 0
    n_qubits = len(basis_state)
    for j in range(n_qubits):
        index += int(basis_state[j])*2**(n_qubits - j - 1)

    return index

def significant_determinants(wavefunction, threshold=1e-8):
    """
    Extract computational-basis determinants with significant amplitudes.

    Parameters
    ----------
    wavefunction : numpy.ndarray, scipy.sparse matrix, or sparse array
        State vector with shape ``(2**n_qubits,)``, ``(2**n_qubits, 1)``,
        or ``(1, 2**n_qubits)``.
    threshold : float, optional
        Keep determinants whose coefficient magnitude is strictly greater
        than this value.

    Returns
    -------
    list[tuple[str, complex]]
        ``(determinant, coefficient)`` pairs sorted by decreasing coefficient
        magnitude. Determinants are occupation bitstrings in OpenFermion
        ordering: qubit 0 is the leftmost (most-significant) bit.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    if sp.sparse.issparse(wavefunction):
        if wavefunction.ndim != 2 or 1 not in wavefunction.shape:
            raise ValueError(
                "wavefunction must be a row or column vector; "
                f"got shape {wavefunction.shape}."
            )
        dimension = max(wavefunction.shape)
        state = wavefunction.tocoo(copy=True)
        state.sum_duplicates()
        indices = state.row if wavefunction.shape[1] == 1 else state.col
        coefficients = state.data
    elif isinstance(wavefunction, np.ndarray):
        if wavefunction.ndim == 1:
            state = wavefunction
        elif wavefunction.ndim == 2 and 1 in wavefunction.shape:
            state = wavefunction.reshape(-1)
        else:
            raise ValueError(
                "wavefunction must be a 1-D, row, or column vector; "
                f"got shape {wavefunction.shape}."
            )
        dimension = state.size
        indices = np.flatnonzero(np.abs(state) > threshold)
        coefficients = state[indices]
    else:
        raise TypeError(
            "wavefunction must be a NumPy array or a SciPy sparse matrix/array."
        )

    if dimension < 1 or dimension & (dimension - 1):
        raise ValueError(
            "wavefunction length must be a positive power of two; "
            f"got {dimension}."
        )

    n_qubits = dimension.bit_length() - 1

    significant = [
        (format(int(index), f"0{n_qubits}b"), coefficient)
        for index, coefficient in zip(indices, coefficients)
        if abs(coefficient) > threshold
    ]
    significant.sort(key=lambda item: abs(item[1]), reverse=True)
    return significant


@dataclass(frozen=True)
class PauliActionMask:
    """
    State-action bit-mask representation of one Pauli product.

    Acting on computational basis index ``b`` gives

        coeff * (-1)**parity(b & sign_mask) |b xor flip_mask>

    The bit convention here follows OpenFermion/Jordan--Wigner statevectors:
    qubit 0 is the most-significant bit of the basis-state integer.  This is
    intentionally different from ``quasisymmetries.bs.utils.PauliMask``, where
    qubit q is stored as bit ``1 << q`` for GF(2)/symplectic algebra.

    Use ``from_pauli_mask`` and ``to_pauli_mask`` to convert across that
    convention boundary.  Since ``PauliMask`` stores only X/Z supports and no
    coefficient, ``from_pauli_mask`` reconstructs the Hermitian Pauli string by
    default, with Y sites carrying the usual matrix phase through the action
    rule rather than through a non-Hermitian prefactor.
    """

    coeff: complex
    flip_mask: int
    sign_mask: int

    @classmethod
    def from_pauli_term(cls, term, coeff=1.0, n_qubits=None):
        flip_mask = 0
        sign_mask = 0
        n_y = 0

        for q, pauli in term:
            if n_qubits is None:
                bit_mask = 1 << int(q)
            else:
                bit_mask = 1 << (int(n_qubits) - 1 - int(q))

            if pauli == "X":
                flip_mask ^= bit_mask
            elif pauli == "Y":
                flip_mask ^= bit_mask
                sign_mask ^= bit_mask
                n_y += 1
            elif pauli == "Z":
                sign_mask ^= bit_mask
            else:
                raise ValueError(f"Unknown Pauli operator {pauli!r}.")

        return cls(complex(coeff) * (1.0j ** n_y), flip_mask, sign_mask)

    @classmethod
    def from_pauli_mask(cls, mask, n_qubits, coeff=1.0):
        """
        Convert a beam-search ``PauliMask = (x_mask, z_mask)`` to an action mask.

        The input mask uses bit ``1 << q`` for qubit q.  The returned action
        mask uses statevector-index bits ``1 << (n_qubits - 1 - q)``.

        If no coefficient is supplied, the result represents the Hermitian
        Pauli operator produced by ``bs.utils.mask_to_term``.
        """
        try:
            from .bs.utils import masks_to_term
        except ImportError:  # pragma: no cover - defensive for direct execution
            from quasisymmetries.bs.utils import masks_to_term

        term = masks_to_term(mask, n_qubits)
        return cls.from_pauli_term(term, coeff=coeff, n_qubits=n_qubits)

    def to_pauli_mask(self, n_qubits):
        """
        Convert this action mask to a beam-search ``PauliMask = (x_mask, z_mask)``.

        The complex coefficient/action phase is intentionally discarded, matching
        the algebraic purpose of ``PauliMask``.
        """
        x_mask = 0
        z_mask = 0
        for q in range(n_qubits):
            state_bit = 1 << (int(n_qubits) - 1 - q)
            if self.flip_mask & state_bit:
                x_mask |= 1 << q
            if self.sign_mask & state_bit:
                z_mask |= 1 << q
        return x_mask, z_mask

    def phases(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        phases = np.full(indices.shape, self.coeff, dtype=np.complex128)
        if self.sign_mask:
            phases[_parity_array(indices & self.sign_mask)] *= -1.0
        return phases


def pauli_mask_to_action_mask(mask, n_qubits, coeff=1.0):
    """Convert a beam-search PauliMask into a Hermitian PauliActionMask by default."""
    return PauliActionMask.from_pauli_mask(mask, n_qubits=n_qubits, coeff=coeff)


def action_mask_to_pauli_mask(action_mask, n_qubits):
    """Convert a PauliActionMask into a beam-search PauliMask, dropping coeff/phase."""
    return action_mask.to_pauli_mask(n_qubits=n_qubits)


def qubit_operator_to_pauli_action_masks(op, n_qubits):
    """Convert an OpenFermion QubitOperator into PauliActionMask objects."""
    return [
        PauliActionMask.from_pauli_term(term, coeff, n_qubits=n_qubits)
        for term, coeff in op.terms.items()
    ]


class SparseQubitState:
    """
    Sparse computational-basis state represented by integer indices and amplitudes.

    The basis-index convention matches OpenFermion statevectors:
    qubit 0 is the leftmost/most-significant bit in the binary index.
    """

    _PAULI_MATRICES = {
        "I": np.eye(2, dtype=np.complex128),
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }

    def __init__(self, indices, coeffs, n_qubits=None, *, copy=True, drop_tol=0.0):
        indices = np.asarray(indices, dtype=np.int64)
        coeffs = np.asarray(coeffs, dtype=np.complex128)

        if indices.ndim != 1 or coeffs.ndim != 1:
            raise ValueError("indices and coeffs must be one-dimensional.")
        if len(indices) != len(coeffs):
            raise ValueError("indices and coeffs must have the same length.")
        if np.any(indices < 0):
            raise ValueError("Basis indices must be non-negative.")

        if n_qubits is None:
            max_index = int(indices.max()) if len(indices) else 0
            n_qubits = max(1, max_index.bit_length())
        self.n_qubits = int(n_qubits)
        self.dimension = 1 << self.n_qubits

        if np.any(indices >= self.dimension):
            raise ValueError("Basis index exceeds 2**n_qubits.")

        self.indices, self.coeffs = self._coalesce(indices, coeffs, drop_tol=drop_tol)
        if copy:
            self.indices = self.indices.copy()
            self.coeffs = self.coeffs.copy()
        self._refresh_lookup()

    @staticmethod
    def _coalesce(indices, coeffs, drop_tol=0.0):
        if len(indices) == 0:
            return indices.astype(np.int64), coeffs.astype(np.complex128)

        unique, inverse = np.unique(indices, return_inverse=True)
        summed = np.zeros(len(unique), dtype=np.complex128)
        np.add.at(summed, inverse, coeffs)

        if drop_tol > 0:
            keep = np.abs(summed) > drop_tol
            unique = unique[keep]
            summed = summed[keep]
        return unique.astype(np.int64), summed.astype(np.complex128)

    def _refresh_lookup(self):
        self._amp = {
            int(index): complex(coeff)
            for index, coeff in zip(self.indices, self.coeffs)
            if coeff != 0
        }

    @property
    def nnz(self):
        return len(self.indices)

    @classmethod
    def from_dense(cls, state, n_qubits=None, threshold=0.0):
        state = np.asarray(state, dtype=np.complex128).reshape(-1)
        dim = state.size
        if dim < 1 or dim & (dim - 1):
            raise ValueError("Dense state length must be a positive power of two.")

        inferred = dim.bit_length() - 1
        if n_qubits is None:
            n_qubits = inferred
        elif int(n_qubits) != inferred:
            raise ValueError("n_qubits does not match dense state dimension.")

        indices = np.flatnonzero(np.abs(state) > threshold)
        return cls(indices, state[indices], n_qubits=n_qubits)

    @classmethod
    def from_scipy_sparse(cls, state, n_qubits=None, threshold=0.0):
        if not sp.sparse.issparse(state):
            raise TypeError("state must be a SciPy sparse vector.")
        if state.ndim != 2 or 1 not in state.shape:
            raise ValueError("state must be a row or column sparse vector.")

        dim = max(state.shape)
        if dim < 1 or dim & (dim - 1):
            raise ValueError("Sparse state length must be a positive power of two.")

        inferred = dim.bit_length() - 1
        if n_qubits is None:
            n_qubits = inferred
        elif int(n_qubits) != inferred:
            raise ValueError("n_qubits does not match sparse state dimension.")

        coo = state.tocoo(copy=True)
        coo.sum_duplicates()
        indices = coo.row if state.shape[1] == 1 else coo.col
        coeffs = coo.data
        keep = np.abs(coeffs) > threshold
        return cls(indices[keep], coeffs[keep], n_qubits=n_qubits)

    @classmethod
    def from_dict(cls, amplitudes, n_qubits=None, threshold=0.0):
        items = [
            (int(index), complex(coeff))
            for index, coeff in amplitudes.items()
            if abs(coeff) > threshold
        ]
        if not items:
            return cls([], [], n_qubits=n_qubits or 1)
        indices, coeffs = zip(*items)
        return cls(indices, coeffs, n_qubits=n_qubits)

    @classmethod
    def from_determinants(cls, determinants, n_qubits=None, threshold=0.0):
        indices = []
        coeffs = []
        for determinant, coeff in determinants:
            if abs(coeff) <= threshold:
                continue
            if isinstance(determinant, str):
                bits = determinant
            else:
                bits = "".join(str(int(bit)) for bit in determinant)
            if any(bit not in "01" for bit in bits):
                raise ValueError(f"Invalid determinant bitstring {bits!r}.")
            if n_qubits is None:
                n_qubits = len(bits)
            elif len(bits) != int(n_qubits):
                raise ValueError("All determinants must have length n_qubits.")
            indices.append(int(bits, 2))
            coeffs.append(coeff)
        return cls(indices, coeffs, n_qubits=n_qubits or 1)

    @classmethod
    def basis_state(cls, determinant, n_qubits=None, coeff=1.0):
        return cls.from_determinants([(determinant, coeff)], n_qubits=n_qubits)

    def copy(self):
        return SparseQubitState(self.indices, self.coeffs, self.n_qubits)

    def to_dense(self):
        state = np.zeros(self.dimension, dtype=np.complex128)
        state[self.indices] = self.coeffs
        return state

    def to_scipy_sparse(self, column=True):
        if column:
            rows = self.indices
            cols = np.zeros(self.nnz, dtype=np.int64)
            shape = (self.dimension, 1)
        else:
            rows = np.zeros(self.nnz, dtype=np.int64)
            cols = self.indices
            shape = (1, self.dimension)
        return csr_matrix((self.coeffs, (rows, cols)), shape=shape)

    def to_dict(self):
        return dict(self._amp)

    def significant_determinants(self, threshold=1e-8):
        out = [
            (format(int(index), f"0{self.n_qubits}b"), coeff)
            for index, coeff in zip(self.indices, self.coeffs)
            if abs(coeff) > threshold
        ]
        out.sort(key=lambda item: abs(item[1]), reverse=True)
        return out

    def norm_squared(self):
        return float(np.real(np.vdot(self.coeffs, self.coeffs)))

    def norm(self):
        return float(np.sqrt(self.norm_squared()))

    def normalize(self):
        norm = self.norm()
        if norm == 0:
            raise ValueError("Cannot normalize a zero state.")
        self.coeffs = self.coeffs / norm
        self._refresh_lookup()
        return self

    def normalized(self):
        return self.copy().normalize()

    def expectation_mask(self, mask):
        targets = self.indices ^ int(mask.flip_mask)
        target_coeffs = np.fromiter(
            (self._amp.get(int(target), 0.0j) for target in targets),
            dtype=np.complex128,
            count=len(targets),
        )
        phases = mask.phases(self.indices)
        return np.sum(np.conjugate(target_coeffs) * phases * self.coeffs)

    def expectation_masks(self, masks):
        return sum((self.expectation_mask(mask) for mask in masks), 0.0j)

    def expectation_pauli_term(self, term, coeff=1.0):
        mask = PauliActionMask.from_pauli_term(term, coeff, n_qubits=self.n_qubits)
        return self.expectation_mask(mask)

    def expectation_qubit_operator(self, op):
        return self.expectation_masks(
            qubit_operator_to_pauli_action_masks(op, self.n_qubits)
        )

    def expectation_sparse_operator(self, operator):
        vec = self.to_scipy_sparse(column=True)
        out = operator @ vec
        return (vec.conjugate().T @ out)[0, 0]

    def apply_mask(self, mask, drop_tol=0.0):
        targets = self.indices ^ int(mask.flip_mask)
        coeffs = mask.phases(self.indices) * self.coeffs
        return SparseQubitState(
            targets,
            coeffs,
            n_qubits=self.n_qubits,
            drop_tol=drop_tol,
        )

    def apply_masks(self, masks, drop_tol=0.0):
        targets = []
        coeffs = []
        for mask in masks:
            targets.append(self.indices ^ int(mask.flip_mask))
            coeffs.append(mask.phases(self.indices) * self.coeffs)
        if not targets:
            return SparseQubitState([], [], n_qubits=self.n_qubits)
        return SparseQubitState(
            np.concatenate(targets),
            np.concatenate(coeffs),
            n_qubits=self.n_qubits,
            drop_tol=drop_tol,
        )

    def apply_qubit_operator(self, op, drop_tol=0.0):
        masks = qubit_operator_to_pauli_action_masks(op, self.n_qubits)
        return self.apply_masks(masks, drop_tol=drop_tol)

    def apply_sparse_operator(self, operator, drop_tol=0.0):
        out = operator @ self.to_scipy_sparse(column=True)
        return SparseQubitState.from_scipy_sparse(
            out,
            n_qubits=self.n_qubits,
            threshold=drop_tol,
        )

    def variance_qubit_operator(self, op):
        exp_val = self.expectation_qubit_operator(op)
        op_state = self.apply_qubit_operator(op)
        return np.real_if_close(op_state.norm_squared() - abs(exp_val) ** 2)

    def variance_sparse_operator(self, operator):
        exp_val = self.expectation_sparse_operator(operator)
        op_state = self.apply_sparse_operator(operator)
        return np.real_if_close(op_state.norm_squared() - abs(exp_val) ** 2)

    @staticmethod
    def _entropy_from_probabilities(probs, base=2.0, tol=1e-12):
        probs = np.asarray(probs, dtype=float).reshape(-1)
        probs[np.abs(probs) < tol] = 0.0
        probs = probs[probs > tol]
        if probs.size == 0:
            return 0.0
        logs = np.log(probs)
        if base is not None:
            logs /= np.log(base)
        return float(-np.sum(probs * logs))

    def cut_entropy(
        self,
        cut,
        base=2.0,
        tol=1e-12,
        max_dense_dim=4096,
    ):
        """
        Exact bipartite entropy across a contiguous cut of the sparse state.

        The cut is between qubits ``cut - 1`` and ``cut`` in the current qubit
        ordering.  Instead of materializing the full ``2**cut`` by
        ``2**(n_qubits-cut)`` Schmidt matrix, this routine keeps only left and
        right bit patterns that occur in the sparse support, then diagonalizes
        the smaller Gram matrix.
        """
        cut = int(cut)
        if cut <= 0 or cut >= self.n_qubits:
            raise ValueError("cut must satisfy 0 < cut < n_qubits.")

        norm = self.norm()
        if norm == 0:
            raise ValueError("Cannot compute entropy of a zero state.")

        n_right = self.n_qubits - cut
        right_mask = (1 << n_right) - 1
        left_bits = self.indices >> n_right
        right_bits = self.indices & right_mask

        _, left_inv = np.unique(left_bits, return_inverse=True)
        _, right_inv = np.unique(right_bits, return_inverse=True)

        n_left = int(left_inv.max()) + 1 if left_inv.size else 0
        n_right_patterns = int(right_inv.max()) + 1 if right_inv.size else 0
        gram_dim = min(n_left, n_right_patterns)

        if gram_dim == 0:
            return 0.0
        if max_dense_dim is not None and gram_dim > int(max_dense_dim):
            raise ValueError(
                "Sparse cut entropy would require diagonalizing a "
                f"{gram_dim}x{gram_dim} dense Gram matrix. Increase "
                "max_dense_dim if this is intentional."
            )

        schmidt_matrix = sp.sparse.coo_matrix(
            (self.coeffs / norm, (left_inv, right_inv)),
            shape=(n_left, n_right_patterns),
        ).tocsr()

        if n_left <= n_right_patterns:
            rho = (schmidt_matrix @ schmidt_matrix.getH()).toarray()
        else:
            rho = (schmidt_matrix.getH() @ schmidt_matrix).toarray()
        rho = 0.5 * (rho + rho.conjugate().T)
        evals = np.real(np.linalg.eigvalsh(rho))
        evals[np.abs(evals) < tol] = 0.0
        evals = np.maximum(evals, 0.0)
        return self._entropy_from_probabilities(evals, base=base, tol=tol)

    def entropies_at_cuts(
        self,
        log_base=2.0,
        tol=1e-12,
        max_dense_dim=4096,
    ):
        return [
            self.cut_entropy(
                cut,
                base=log_base,
                tol=tol,
                max_dense_dim=max_dense_dim,
            )
            for cut in range(1, self.n_qubits)
        ]

    def one_qubit_pauli_expectations(self, qubit):
        return {
            pauli: self.expectation_pauli_term(((qubit, pauli),))
            for pauli in ("X", "Y", "Z")
        }

    def two_qubit_pauli_expectations(self, qubit_i, qubit_j):
        if qubit_i == qubit_j:
            raise ValueError("qubit_i and qubit_j must be distinct.")
        return {
            (pauli_i, pauli_j): self.expectation_pauli_term(
                ((qubit_i, pauli_i), (qubit_j, pauli_j))
            )
            for pauli_i in ("X", "Y", "Z")
            for pauli_j in ("X", "Y", "Z")
        }

    def one_qubit_rdm(self, qubit):
        rho = self._PAULI_MATRICES["I"].copy()
        for pauli, value in self.one_qubit_pauli_expectations(qubit).items():
            rho += value * self._PAULI_MATRICES[pauli]
        rho *= 0.5
        return 0.5 * (rho + rho.conjugate().T)

    def two_qubit_rdm(self, qubit_i, qubit_j):
        if qubit_i == qubit_j:
            raise ValueError("qubit_i and qubit_j must be distinct.")
        labels = ("I", "X", "Y", "Z")
        rho = np.zeros((4, 4), dtype=np.complex128)
        one_i = self.one_qubit_pauli_expectations(qubit_i)
        one_j = self.one_qubit_pauli_expectations(qubit_j)
        two = self.two_qubit_pauli_expectations(qubit_i, qubit_j)

        for a in labels:
            for b in labels:
                if a == "I" and b == "I":
                    value = 1.0
                elif a == "I":
                    value = one_j[b]
                elif b == "I":
                    value = one_i[a]
                else:
                    value = two[(a, b)]
                rho += value * np.kron(self._PAULI_MATRICES[a], self._PAULI_MATRICES[b])
        rho *= 0.25
        return 0.5 * (rho + rho.conjugate().T)

    @staticmethod
    def von_neumann_entropy(rho, base=2.0, tol=1e-12):
        rho = np.asarray(rho, dtype=np.complex128)
        rho = 0.5 * (rho + rho.conjugate().T)
        evals = np.real(np.linalg.eigvalsh(rho))
        evals[np.abs(evals) < tol] = 0.0
        evals = evals[evals > tol]
        if evals.size == 0:
            return 0.0
        logs = np.log(evals)
        if base is not None:
            logs /= np.log(base)
        return float(-np.sum(evals * logs))

    def qubit_mutual_information_matrix(
        self,
        base=2.0,
        convention="standard",
        entropy_tol=1e-12,
    ):
        if convention not in {"standard", "half"}:
            raise ValueError("convention must be 'standard' or 'half'.")
        factor = 0.5 if convention == "half" else 1.0
        s1 = np.zeros(self.n_qubits, dtype=float)
        for i in range(self.n_qubits):
            s1[i] = self.von_neumann_entropy(
                self.one_qubit_rdm(i),
                base=base,
                tol=entropy_tol,
            )

        s2 = np.zeros((self.n_qubits, self.n_qubits), dtype=float)
        mi = np.zeros((self.n_qubits, self.n_qubits), dtype=float)
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                sij = self.von_neumann_entropy(
                    self.two_qubit_rdm(i, j),
                    base=base,
                    tol=entropy_tol,
                )
                s2[i, j] = s2[j, i] = sij
                mij = factor * (s1[i] + s1[j] - sij)
                if mij < 0.0 and abs(mij) < 1e-10:
                    mij = 0.0
                mi[i, j] = mi[j, i] = max(0.0, float(mij))
        return mi, s1, s2

    def reorder_qubits(self, ordering):
        if sorted(ordering) != list(range(self.n_qubits)):
            raise ValueError("ordering must be a permutation of all qubits.")
        new_indices = np.zeros_like(self.indices)
        for new_pos, old_qubit in enumerate(ordering):
            old_mask = 1 << (self.n_qubits - 1 - int(old_qubit))
            new_mask = 1 << (self.n_qubits - 1 - int(new_pos))
            new_indices[(self.indices & old_mask) != 0] |= new_mask
        return SparseQubitState(new_indices, self.coeffs, n_qubits=self.n_qubits)

def get_reference_state(occ_no_state, tf = 'bk', gs_format = 'dm'):
    """
    Given some occupation numebr vector, make the density matrix that corresponds to that state.
    Args:
        occ_no_state (str or list/np.array): Occupation number vector. If str, ordered from left to right going from 0 -> n-1 in terms of orbitals.
    Returns:
        dm (sp.sparse.coo_matrix): Density matrix (sparse for efficiency) of the reference state in qubit space.
        or wfs (np.array): wavefunction of the CISD state in qubit space.
    """
    n_qubits = len(occ_no_state)
    bk_basis_state = get_bk_basis_states(occ_no_state, n_qubits)
    index = find_index(bk_basis_state[0])

    if gs_format == 'wfs':
        wfs = np.zeros(2**n_qubits)
        wfs[index] = 1

        return wfs


    if gs_format == 'dm':

        dm = sp.sparse.coo_matrix(([1], ([index], [index])), shape = (2**n_qubits, 2**n_qubits))

        return dm

def get_occ_no(mol, n_qubits):
    """
    Given some molecule, find the reference occupation number state.
    Args:
        mol (str): H2, LiH, BeH2, H2O, NH3
    Returns:
        occ_no (str): Occupation no. vector.
    """
    n_electrons = {'h2': 2, 'lih': 4, 'beh2': 6, 'h2o': 10, 'nh3': 10, 'n2': 14, 'hf':10, 'ch4':10, 'co':14, 'h4':4, 'ch2':8, 'heh':2, 'h6':6, 'nh':8, 'h3':2, 'h4sq':4, 'h2ost':10, 'beh2st':6, 'h2ost2':10, 'beh2st2':6}
    occ_no = '1'*n_electrons[mol] + '0'*(n_qubits - n_electrons[mol])

    return occ_no

def get_jw_cisd_basis_states_wrap(ref_occ_nos, n_qubits):
    """
    Given some occupation number, find the all other occupation numbers that are achieved by single and double excitations.
    Args:
        ref_occ_nos (str): Reference (likely HF) occupation number ordered from left to right going from 0 -> n-1 in terms of orbitals.
    Returns:
        cisd_basis_states (List[str]): List of all occupation number achieved by singles and doubles from reference occupation number.
    """

    indices = [find_index(get_jw_basis_states(ref_occ_nos, n_qubits))]
    for occidx, occ_orbitals in enumerate(ref_occ_nos):
        if occ_orbitals == '1':
            annihilated_state = list(ref_occ_nos)
            annihilated_state[occidx] = '0'

            #Singles
            for virtidx, virtual_orbs in enumerate(ref_occ_nos):
                if virtual_orbs == '0':
                    new_state = annihilated_state[:]
                    new_state[virtidx] = '1'
                    indices.append(find_index(get_jw_basis_states(''.join(new_state), n_qubits)))

                    #Doubles
                    for occ2idx in range(occidx +1, n_qubits):
                        if ref_occ_nos[occ2idx] == '1':
                            annihilated_state_double = new_state[:]
                            annihilated_state_double[occ2idx] = '0'

                            for virt2idx in range(virtidx +1, n_qubits):
                                if ref_occ_nos[virt2idx] == '0':
                                    new_state_double = annihilated_state_double[:]
                                    new_state_double[virt2idx] = '1'
                                    indices.append(find_index(get_jw_basis_states(''.join(new_state_double), n_qubits)))
    return indices

def get_bk_cisd_basis_states_wrap(ref_occ_nos, n_qubits):
    """
    Given some occupation number, find the all other occupation numbers that are achieved by single and double excitations.
    Args:
        ref_occ_nos (str): Reference (likely HF) occupation number ordered from left to right going from 0 -> n-1 in terms of orbitals.
    Returns:
        cisd_basis_states (List[str]): List of all occupation number achieved by singles and doubles from reference occupation number.
    """

    indices = [find_index(get_bk_basis_states(ref_occ_nos, n_qubits))]
    for occidx, occ_orbitals in enumerate(ref_occ_nos):
        if occ_orbitals == '1':
            annihilated_state = list(ref_occ_nos)
            annihilated_state[occidx] = '0'

            #Singles
            for virtidx, virtual_orbs in enumerate(ref_occ_nos):
                if virtual_orbs == '0':
                    new_state = annihilated_state[:]
                    new_state[virtidx] = '1'
                    indices.append(find_index(get_bk_basis_states(''.join(new_state), n_qubits)))

                    #Doubles
                    for occ2idx in range(occidx +1, n_qubits):
                        if ref_occ_nos[occ2idx] == '1':
                            annihilated_state_double = new_state[:]
                            annihilated_state_double[occ2idx] = '0'

                            for virt2idx in range(virtidx +1, n_qubits):
                                if ref_occ_nos[virt2idx] == '0':
                                    new_state_double = annihilated_state_double[:]
                                    new_state_double[virt2idx] = '1'
                                    indices.append(find_index(get_bk_basis_states(''.join(new_state_double), n_qubits)))
    return indices

def get_bk_cisd_basis_states(mol, n_qubits):
    """
    Given some molecule, find the all BK basis vectors that correspond to occupation numbers that are achieved by single and double excitations.
    Args:
        mol (str): H2, LiH, BeH2, H2O, NH3
        n_qubits (int): No. of qubits
    Returns:
        bk_basis_states (List[array]): List of all BK basis states corresponding to occupation numbers achieved by singles and doubles from reference occupation number.
    """

    ref_occ_nos = get_occ_no(mol, n_qubits)
    indices = get_bk_cisd_basis_states_wrap(ref_occ_nos, n_qubits)
    return indices


def get_jw_cisd_basis_states(mol, n_qubits):
    """
    Given some molecule, find the all BK basis vectors that correspond to occupation numbers that are achieved by single and double excitations.
    Args:
        mol (str): H2, LiH, BeH2, H2O, NH3
        n_qubits (int): No. of qubits
    Returns:
        jw_basis_states (List[array]): List of all JW basis states corresponding to occupation numbers achieved by singles and doubles from reference occupation number.
    """

    ref_occ_nos = get_occ_no(mol, n_qubits)
    indices = get_jw_cisd_basis_states_wrap(ref_occ_nos, n_qubits)
    return indices

def create_hamiltonian_in_subspace(indices, Hq, n_qubits):
    """
    Given some basis states, create the Hamiltonian within the span of those basis states.
    Args:
        qubit_basis_states(List[array] or List[str]): List of basis vectors to create hamiltonian within
        Hq (QubitOperator): Qubit hamiltonian
        n_qubits (int): Number of qubits.
    Returns:
        H_mat_sub (sp.sparse.coo_matrix): Hamiltonian matrix defined in subspace.
        indices (List[int]): Gives the index in the 2**n dimensional space of the ith qubit_basis_state.
    """

    subspace_dim = len(indices)

    row_idx = []
    col_idx = []
    H_mat_elements = []

    #print(len(Hq.terms))
    elements_sum = np.zeros((len(indices),len(indices)), dtype =complex)
    op_sum = of.QubitOperator.zero()
    for prog, op in enumerate(Hq):
        op_sum += op
        if (prog + 1)%350 == 0 or prog == len(Hq.terms) - 1:
            #print(prog)
            opspar = of.get_sparse_operator(op_sum, n_qubits)
            op_sum = of.QubitOperator.zero()
            for iidx, iindx in enumerate(indices):
                for jidx, jindx in enumerate(indices):
                    elements_sum[iidx, jidx] += opspar[iindx, jindx]
                 
    for iidx, iindx in enumerate(indices):
        for jidx, jindx in enumerate(indices):
            row_idx.append(iidx)
            col_idx.append(jidx)
            H_mat_elements.append(elements_sum[iidx, jidx])

    H_mat_sub = sp.sparse.coo_matrix((H_mat_elements, (row_idx, col_idx)), shape = (subspace_dim, subspace_dim))

    return H_mat_sub

def get_cisd_gs(occ_str, Hq, n_qubits, gs_format = 'dm', reduce_determinants = False, tf = 'bk'):
    """
    Finds the CISD wavefunction/density matrix in qubit space.
    Args:
        mol (str): H2, LiH, BeH2, H2O, NH3
        Hq (QubitOperator): Qubit hamiltonian
        n_qubits (int): No. of qubits
    Returns:
        dm (sp.sparse.coo_matrix): Density matrix (sparse for efficiency) of the CISD state in qubit space.
        or wfs (np.array): wavefunction of the CISD state in qubit space.
    """


    if tf == 'bk':
        indices = get_bk_cisd_basis_states_wrap(occ_str, n_qubits)
    elif tf == 'jw':
        indices = get_jw_cisd_basis_states_wrap(occ_str, n_qubits)
    else:
        return('Transformation Not Valid.')
    H_mat_cisd = create_hamiltonian_in_subspace(indices, Hq, n_qubits)

    #energy, gs = get_gs(mol, H_mat_cisd)
    energy, gs = get_gs(H_mat_cisd)

    if reduce_determinants == True:
        while np.linalg.norm(gs) > 0.99:
            min_index = np.argmin(np.abs(gs))
            gs[min_index] = 0

        gs = gs/np.linalg.norm(gs) #Renormalisation


    if gs_format == 'wfs': # TODO make to sparse or use SDState

        wfs = np.zeros(2**n_qubits)

        for iidx, iindx in enumerate(indices):
            wfs[iindx] = gs[iidx]

        wfs = wfs/np.linalg.norm(wfs)

        return energy, wfs

    if gs_format == 'dm':

        row_idx = []
        col_idx = []
        dm_vals = []

        for iidx, iindx in enumerate(indices):
            for jidx, jindx in enumerate(indices):
                row_idx.append(iindx)
                col_idx.append(jindx)
                dm_vals.append(gs[iidx]*np.conj(gs[jidx]))

        dm = sp.sparse.coo_matrix((dm_vals, (row_idx, col_idx)), shape = (2**n_qubits, 2**n_qubits))
        dm = dm / dm.diagonal().sum()

        return energy, dm

### FCI
# AI code, seems to work
def get_fci_state_openfermion(molecule: MolecularData, threshold=1e-12):
    """
    Return (energy, state, info) where state is a sparse column vector
    compatible with OpenFermion/get_ground_state conventions.

    Parameters
    ----------
    molecule
        MolecularData / PyscfMolecularData object already processed by run_pyscf.
    threshold : float
        Drop coefficients with |c| <= threshold.

    Returns
    -------
    energy : float
    state : scipy.sparse.csr_matrix
        Shape (2**n_qubits, 1)
    info : dict
    """
    if not hasattr(molecule, "_pyscf_data") or molecule._pyscf_data is None:
        raise ValueError("molecule._pyscf_data missing; use a molecule returned by run_pyscf.")

    pyscf_data = molecule._pyscf_data
    pyscf_mol = pyscf_data["mol"]
    pyscf_scf = pyscf_data["scf"]

    # Reuse stored FCI solver if present, else build one.
    solver = pyscf_data.get("fci", None)
    if solver is None:
        solver = fci.FCI(pyscf_mol, pyscf_scf.mo_coeff)
        solver.verbose = 0

    energy, ci = solver.kernel()
    ci = np.asarray(ci)

    norb = int(pyscf_scf.mo_coeff.shape[1])   # spatial orbitals
    n_qubits = 2 * norb
    dim = 1 << n_qubits

    nelec = int(pyscf_mol.nelectron)
    spin = int(pyscf_mol.spin)   # = n_alpha - n_beta
    n_alpha = (nelec + spin) // 2
    n_beta = nelec - n_alpha

    alpha_strings = np.asarray(cistring.make_strings(range(norb), n_alpha), dtype=np.int64)
    beta_strings = np.asarray(cistring.make_strings(range(norb), n_beta), dtype=np.int64)

    expected_shape = (len(alpha_strings), len(beta_strings))
    if ci.shape != expected_shape:
        raise ValueError(
            f"Unexpected CI tensor shape {ci.shape}; expected {expected_shape}."
        )

    def occupied_orbitals(det, norb):
        return [p for p in range(norb) if (det >> p) & 1]

    def pyscf_det_to_openfermion_index_and_phase(alpha_det, beta_det, norb):
        """
        PySCF CI basis is organized by separate alpha/beta strings.
        We embed into OpenFermion spin-orbital order:
            [a0, b0, a1, b1, ..., a_{norb-1}, b_{norb-1}]
        and build the basis index using qubit 0 as the leftmost tensor factor.

        Returns
        -------
        basis_index : int
        phase : +1 or -1
        """
        occ_alpha = occupied_orbitals(alpha_det, norb)
        occ_beta = occupied_orbitals(beta_det, norb)

        # Fermionic sign from reordering:
        # starting from all-alpha then all-beta ordering
        # into interleaved spin-orbital ordering.
        inversions = 0
        for p in occ_alpha:
            for q in occ_beta:
                if q < p:
                    inversions += 1
        phase = -1.0 if (inversions % 2) else 1.0

        # Build OpenFermion computational-basis index.
        # qubit 0 is the leftmost tensor factor, so it contributes to the
        # most-significant bit of the basis index.
        idx = 0
        for p in occ_alpha:
            q = 2 * p
            idx |= (1 << (n_qubits - 1 - q))
        for p in occ_beta:
            q = 2 * p + 1
            idx |= (1 << (n_qubits - 1 - q))

        return idx, phase

    rows = []
    data = []

    for ia, alpha_det in enumerate(alpha_strings):
        for ib, beta_det in enumerate(beta_strings):
            coeff = ci[ia, ib]
            if abs(coeff) > threshold:
                idx, phase = pyscf_det_to_openfermion_index_and_phase(
                    int(alpha_det), int(beta_det), norb
                )
                rows.append(idx)
                data.append(phase * coeff)

    if rows:
        cols = np.zeros(len(rows), dtype=np.int64)
        state = csr_matrix(
            (np.asarray(data, dtype=np.complex128), (np.asarray(rows), cols)),
            shape=(dim, 1),
            dtype=np.complex128,
        )
    else:
        state = csr_matrix((dim, 1), dtype=np.complex128)

    info = {
        "norb": norb,
        "n_qubits": n_qubits,
        "nelec": nelec,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
        "nnz": state.nnz,
    }
    return float(energy), state, info
