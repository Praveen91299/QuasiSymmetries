## In search of greater ~purpose~ Pauli Quasi Symmetries...  

See `hct_bs_sample.py` for example script to find symmetries and test various metrics.

Notes:  
- HCT_mod should give the same symmetries as found in the HCT paper, but the diagonalizing Clifford is not unique, hence need not match.  
- BS-HCT has been observed to not improve much upon HCT, hence redundant.  
- Beam search (with HCT symmetries added) currently performs best (lowest entanglement/bond dimension for DMRG convergence).  

### Requirements  
numpy, openfermion, openfermionpyscf, pyscf, block2, scipy and other standard libraries.