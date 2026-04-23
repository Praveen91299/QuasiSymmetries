import numpy as np
from scipy.linalg import expm, logm
from src.ferm_utils import get_U
from src.mat_utils import is_antihermitian, pad_2d_to_square, is_unitary, is_skewsymmetric, skew_log_orthogonal, is_orthogonal
from qiskit import QuantumCircuit
from copy import deepcopy

### decomposition of orbital rotation into individual rotations
def depth_eff_order_mf(N):
    """
    Returns index ordering for linear depth circuit

    For example N = 6 gives elimination order
    [ 0.  0.  0.  0.  0.  0.]
    [ 7.  0.  0.  0.  0.  0.]
    [ 5. 10.  0.  0.  0.  0.]
    [ 3.  8. 12.  0.  0.  0.]
    [ 2.  6. 11. 14.  0.  0.]
    [ 1.  4.  9. 13. 15.  0.]
    """
    l = []
    for c in range(0, N - 1):
        for r in range(1, N):
            if r - c > 0:
                l.append([r, c, 2 * c - r + N])
    l.sort(key=lambda x: x[2])
    return [(a[0], a[1]) for a in l]

def givens(i, j, theta, N):
    """
    returns NxN real givens matrix exp[\theta M_ij] where M_ij is the elementary matrix, |i><j| - |j><i|, i neq j

    """

    U = np.identity(N, complex)
    
    if i != j:
        U[i, i], U[j, j], U[i, j], U[j, i] = np.cos(theta), np.cos(theta), np.sin(theta), np.sin(-theta) 
    else:
        U[i, i] = np.exp(theta) # expecting theta to be imaginary

    return U

def decompose_SO(U, verify=True, tol=1e-05):
    """
    Depth efficient Given' decomposition of real unitary, SO(N).
    Linear depth circuit

    Returns list[(i, j, theta)] # final diagonal should be 1
    
    """
    N = len(U)

    U_curr = deepcopy(U)
    assert is_orthogonal(U_curr, tol), "Matrix not orthogonal!"
    depth_eff_ord = depth_eff_order_mf(N)
    rotations = []
    
    for i,j in depth_eff_ord:

        #eliminating (i, j) entry with (i-1, j) entry using givens(i-1, i)
        if abs(U_curr[i, j]) > tol:
            theta = np.arctan2(float(U_curr[i, j]), float(U_curr[i-1, j]))
            ug = givens(i-1, i, theta, N) # storing -theta as the sequence is inverted in the end

            U_curr = ug @ U_curr

            if abs(theta % np.pi) > tol: # to check away from multiples of pi
                rotations.append((i-1, i, - theta))
    
    #diagonal
    for i in range(N):
        if np.isclose(U_curr[i, i], -1):
            rotations.append((i, i, np.pi * 1.j)) #whats happening here??
    
    rotations.reverse()
    #will need to invert in the end

    if verify:
        U_test = np.identity(N)
        for i, j, theta in rotations:
            U_test = givens(i, j, theta, N) @ U_test
        
        assert np.isclose(np.sum(np.abs(U_test.T @ U - np.identity(N))), 0, tol), "Unitary not equivalent!"
    return rotations

def append_givens_circuit(qc:QuantumCircuit, i, j, theta):
    qc.cx(j, i)
    qc.cry(-2*theta, control_qubit=i, target_qubit=j)
    qc.cx(j, i)

### objects
class OrbitalRotation:
    """
    General orbital rotation object defined by generator_mat

    """
    def __init__(self, n_qubits, generator_mat):
        self.n_qubits = n_qubits
        self.generator_mat = generator_mat
    
    @property
    def n_qubits(self):
        return self._n_qubits
    
    @n_qubits.setter
    def n_qubits(self, value):
        assert isinstance(value, int) and value > 0, "Number of qubits {} invalid".format(value)
        self._n_qubits = value

    @property
    def generator_mat(self):
        return self._generator_mat
    
    @generator_mat.setter
    def generator_mat(self, value):

        ### checks square shape of n_qubits x n_qubits, anti hermiticity

        assert (np.shape(value) == (self.n_qubits, self.n_qubits)), "Generator matrix not of correct dimensions!"
        assert is_antihermitian(value), "Generator matrix not antihermitian!"

        self._generator_mat = value
    
    def get_mat_rep(self):
        """
        Returns N x N matrix representation of unitary

        """

        return expm(self.generator_mat)
    
    def get_exp_rep(self):
        """
        Returns 2^N x 2^N (sparse) matrix representation of unitary
        
        """
        return get_U(self.get_mat_rep(), self.n_qubits)
    
    @classmethod
    def num_params(cls, n_qubits):
        return 0

    def get_num_params(self):
        return 0


class ParameterizedOrbitalRotation(OrbitalRotation):
    """
    Parameterized orbital rotation - angles stored in params - cannot be directly used
    
    """
    def __init__(self, params):
        print("DO NOT INITIALIZE PARAMETERIZED ORBITAL ROTATION DIRECTLY")

    @classmethod
    def build_param_mat(self, params, n_qubits):
        pass

    @classmethod
    def num_params(cls, n_qubits):
        pass
    
    def get_num_params(self):
        return self.num_params(self.n_qubits)

    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, value):
        assert len(value) == self.get_num_params(), "Incorrect number of params passed!"
        self._params = value
    
    @property
    def generator_mat(self):
        return self.build_param_mat(self.params, self.n_qubits)
    
    def freeze_params(self):
        """
        Returns unparameterized version

        """

        return OrbitalRotation(self.n_qubits, self.generator_mat)

class RealOrbitalRotation(ParameterizedOrbitalRotation):
    """
    Real orbital rotation

    """
    def __init__(self, n_qubits, params):
        self.n_qubits = n_qubits
        self.params = params
    
    @classmethod
    def num_params(cls, n_qubits):
        return n_qubits*(n_qubits-1)//2
    
    @classmethod
    def build_param_mat(cls, params, n_qubits):
        """
        Real orbital rotations, N(N-1)/2 parameters
        """
        N = cls.num_params(n_qubits)
        assert len(params) == N, "Number of parameters provided don't match!"
        theta = params

        param_mat = np.zeros((n_qubits, n_qubits), complex)

        idx = 0
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                param_mat[i, j] =   theta[idx]
                param_mat[j, i] = - theta[idx]

                idx += 1

        return param_mat


class ImaginaryOrbitalRotation(ParameterizedOrbitalRotation):
    """
    Imaginary orbital rotation
    
    """
    def __init__(self, n_qubits, params):
        self.n_qubits = n_qubits
        self.params = params

    @classmethod
    def num_params(cls, n_qubits):
        return n_qubits*(n_qubits-1)//2
    
    @classmethod
    def build_param_mat(cls, params, n_qubits):
        """
        Imaginary rotations, N(N-1)/2 parameters 
        """
        N = cls.num_params(n_qubits)
        assert len(params) == N, "Number of parameters provided don't match!"
        phi = params

        param_mat = np.zeros((n_qubits, n_qubits), complex)

        idx = 0
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                param_mat[i, j] = 1.j * phi[idx] 
                param_mat[j, i] = 1.j * phi[idx]
                
                idx += 1
        
        return param_mat

class FullOrbitalRotation(ParameterizedOrbitalRotation):
    """
    Full orbital rotation
    
    """
    def __init__(self, n_qubits, params):
        self.n_qubits = n_qubits
        self.params = params

    @classmethod
    def num_params(cls, n_qubits):
        return n_qubits*(n_qubits-1)
    
    @classmethod
    def build_param_mat(cls, params, n_qubits):
        """
        Full U_mf, N(N-1) parameters
        """

        ## get anti hermitian matrix, transform polynomial, and convert to Sparse

        N = cls.num_params(n_qubits)
        assert len(params) == N, "Number of parameters provided don't match!"
        phi = params[:N//2]
        theta = params[N//2:]

        param_mat = np.zeros((n_qubits, n_qubits), complex)

        idx = 0
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                param_mat[i, j] =   theta[idx] + 1.j * phi[idx] 
                param_mat[j, i] = - theta[idx] + 1.j * phi[idx]

                idx += 1
        
        return param_mat

class RestrictedOrbitalRotation(ParameterizedOrbitalRotation):
    """
    Restricted orbital rotation - to qubit pairs

    """
    def __init__(self, n_qubits, params, qubit_pairs):
        self.n_qubits = n_qubits
        self.qubit_pairs = qubit_pairs
        self.params = params
    
    @classmethod
    def num_params(cls, qubit_pairs):
        return len(qubit_pairs)
    
    def get_num_params(self):
        return self.num_params(self.qubit_pairs)

    @classmethod
    def build_param_mat(cls, params, n_qubits, qubit_pairs):
        """
        Excitations restricted to subset of qubit pairs provided (i, j, r/i)
        
        """

        assert len(params) == len(qubit_pairs), "Number of parameters provided don't match!"

        param_mat = np.zeros((n_qubits, n_qubits), complex)

        idx = 0
        for t in qubit_pairs:
            i, j, kind = t

            if kind == "real":
                param_mat[i, j] += params[idx]
                param_mat[j, i] += -params[idx]
            
            if kind == "imag":
                param_mat[i, j] += 1.j * params[idx]
                param_mat[j, i] += 1.j * params[idx]
            
            idx += 1

        return param_mat
    
    @property
    def generator_mat(self):
        return self.build_param_mat(self.params, self.n_qubits, self.qubit_pairs)

class SpinRestrictedRealOrbitalRotation(ParameterizedOrbitalRotation):
    """
    Real orbital rotation, spin restricted, to order spin_ord

    """
    def __init__(self, n_qubits, params, spin_ord='udud'):
        assert n_qubits % 2 == 0, "Odd number of qubits: {}".format(n_qubits)
        self.n_qubits = n_qubits
        self.n_orbs = n_qubits//2
        self.params = params
        self.spin_ord = spin_ord
    
    @classmethod
    def num_params(cls, n_qubits):
        assert n_qubits % 2 == 0, "Number of qubits is not even!"
        n_orbs = n_qubits//2
        return n_orbs*(n_orbs-1)//2
    
    @property
    def generator_mat(self):
        return self.build_param_mat(self.params, self.n_qubits, self.spin_ord)
    
    @classmethod
    def from_spinresU(cls, U, spin_ord='udud', verify=False):
        """
        Initialize from NxN U matrix
        
        """
        def extract_params(M):
            assert is_skewsymmetric(M), "matrix is not skew symmetrc"

            n_orb = len(M)
            params = []
            for i in range(n_orb):
                for j in range(i+1, n_orb):
                    params.append(M[i, j])
            
            return params
        #obtain generator matrix

        n_qubits = 2*len(U)

        assert np.isclose(abs(np.linalg.det(U)), 1), "SpinResU: Error, |det(U)| is not 1!"

        if np.isclose(np.linalg.det(U), -1):
            print("SpinResU: Determinant -1, multiplying first column by -1.")
            n = len(U)
            corr = np.ones(n)
            corr[0] = -1
            U= np.array([corr]) * U
        M = skew_log_orthogonal(U)

        #obtain parameter list
        params = extract_params(M)
        assert len(params) == cls.num_params(n_qubits)

        #initialize
        obj =  cls(n_qubits, params, spin_ord)
        if verify:
            if spin_ord == 'udud':
                Ufull = np.kron(U, np.identity(2))
            else:
                Ufull = np.kron(np.identity(2), U)
            
            assert np.allclose(obj.get_mat_rep().T.conjugate() @ Ufull, np.identity(n_qubits))
        
        return obj
    
    @classmethod
    def build_param_mat(cls, params, n_qubits, spin_ord='udud'):
        """
        Real orbital rotations, N(N-1)/2 parameters
        """

        spinres_param_mat = cls.build_spinres_param_mat(params, n_qubits)
        if spin_ord == 'udud':
            return np.kron(spinres_param_mat, np.identity(2))
        else:
            return np.kron(np.identity(2), spinres_param_mat)
    
    @classmethod
    def build_spinres_param_mat(cls, params, n_qubits):
        """
        Submatrix over a single spin sector only

        """
        N = cls.num_params(n_qubits)
        assert len(params) == N, "Number of parameters provided don't match!"
        theta = params

        param_mat = np.zeros((n_qubits//2, n_qubits//2), complex)

        idx = 0
        for i in range(n_qubits//2):
            for j in range(i+1, n_qubits//2):
                param_mat[i, j] =   theta[idx]
                param_mat[j, i] = - theta[idx]

                idx += 1

        return param_mat
    
    def get_spinres_mat_rep(self):
        """
        Returns N_orb x N_orb matrix representation of unitary

        """
        gen_mat_spinres = self.build_spinres_param_mat(self.params, self.n_qubits)
        return expm(gen_mat_spinres)
    
    def get_spinres_exp_rep(self):
        """
        Returns 2^N_orb x 2^N_orb (sparse) matrix representation of unitary
        
        """
        return get_U(self.get_spinres_mat_rep(), self.n_orbs//2)
    
    def append_qiskit_circuit(self, qc: QuantumCircuit, tol=1e-5):
        """
        appends qiskit 

        """

        if self.spin_ord == 'udud':
            assert False, "Code not developed yet"
        else:
            alpha_qubits, beta_qubits = list(range(self.n_orbs)), list(range(self.n_orbs, 2*self.n_orbs))

            spinres_U = self.get_spinres_mat_rep()
            decomp = decompose_SO(spinres_U, True, tol)

            for i, j, theta in decomp:
                if i == j:
                    #diagonal parts, append Z for -1 phase
                    assert np.isclose(1.j*np.pi, theta, tol), "Incorrect diagonal entry"

                    qc.z([alpha_qubits[i], beta_qubits[i]])
                else:
                    append_givens_circuit(qc, alpha_qubits[i], alpha_qubits[j], theta)
                    append_givens_circuit(qc, beta_qubits[i], beta_qubits[j], theta)

### functions

def combine_orbital_rotations(orbital_rotation_list: list[OrbitalRotation]):
    """
    Create single orbital rotation obj from a list of orbital rotations

    Returns OrbitalRotation of largest qubit size of the list

    """

    #combined_u = orbital_rotation_list[0].get_mat_rep()
    n_qubits_max = int(np.max([orb.n_qubits for orb in orbital_rotation_list]))
    combined_u = np.eye(n_qubits_max, dtype=complex)

    for orb in orbital_rotation_list:
        
        gen = pad_2d_to_square(orb.generator_mat, n_qubits_max)
        combined_u = combined_u @ expm(gen)
    
    combined_generator_matrix = logm(combined_u)

    assert is_unitary(combined_u)
    assert is_antihermitian(combined_generator_matrix)

    return OrbitalRotation(n_qubits=n_qubits_max, generator_mat=combined_generator_matrix)