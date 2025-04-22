import sys, os
num_threads = 3
os.environ["OMP_NUM_THREADS"] = f"{num_threads}"    # For libraries using OpenMP
os.environ["MKL_NUM_THREADS"] = f"{num_threads}"    # For Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = f"{num_threads}"  # For OpenBLAS
os.environ["BLIS_NUM_THREADS"] = f"{num_threads}"   # For BLIS

import numpy as np
from  methods import nucleosome_free_energy, nucleosome_groundstate, read_nucleosome_triads, GenStiffness, calculate_midstep_triads
from binding_model import binding_model_free_energy, binding_model_free_energy_old



genstiff = GenStiffness(method='hybrid')   # alternatively you can use the 'crystal' method for the Olson data
# genstiff = GenStiffness(method='md')   # alternatively you can use the 'crystal' method for the Olson data
seq601 = "ATCGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCGAT"
# seq601 = "GTAGCCCCGATCGATCGATCGCGCGATCTAGCTATATAAAAATCGCGGGCGATCTATTTTAGAGATCCTCTAAACCCGCATTCGCTCGCGCGCGCGCGATCTTATCTAGCTAGTACGATCGAAACTATCTAGCGACGATCATAAACG"
# seq601 = "GTAGCCCCGATCGATCGATCGCGCGATCTAGCTATATAAAAAGCGCGCGCGCGATCTTATCTAGCTAGTACGATCGAAACTATCTAGCGACGATCATAAACGTCGCGGGCGATCTATTTTAGAGATCCTCTAAACCCGCATTCGCTC"
# seq601 = "AAAAGCGCGCGCGCGATCTTATCTAGCTAGTACGATCGAAACTATCTAGCGACGATCATAAACGTCGCGGGCGATCTATTTTAGAGATCCTCTAAACCCGCATTCGCTCGTAGCCCCGATCGATCGATCGCGCGATCTAGCTATATA"

seq601  = "CTGGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGT"
# seq601 = 'TTCCACATGGATAATACAAGAGATTCATCGACGTGCTCATTTGGCATTAGGGCATCATCCTAATGAGATTCGGTGGCGCTAACAACTTCGCTGAAAGATCAGTGGAGCGAACTGCCCTACTGTTAATTGGGTACCAGACCTCCTCAC'

# seq601 = 'ATTTGGCCTTAAAAAAACTTCCCCCTTCGCTATACAAGAGATTCATCGGAAAGATCAGTGGAGCGAACTGCCCTACATCATCCTAATGAGATTCGGTGCTGTTAATTGGGTACCAGACTTCCACGCGAAAAAATCGCGGGGGCACGA'
    

seq = seq601
    

stiffmat,groundstate = genstiff.gen_params(seq,use_group=True)

# from methods.PolyCG import polycg
# groundstate,stiffmat = polycg.cgnaplus_bps_params(seq,group_split=True)


triadfn = 'methods/State/Nucleosome.state'
nuctriads = read_nucleosome_triads(triadfn)

midstep_constraint_locations = [
    2, 6, 14, 17, 24, 29, 
    34, 38, 45, 49, 55, 59, 
    65, 69, 76, 80, 86, 90, 
    96, 100, 107, 111, 116, 121, 
    128, 131, 139, 143
]

# FOR NOW WE USE THE FIXED MIDSTEP TRIADS AS MU_0
# Find midstep triads in fixed framework for comparison
nuc_mu0 = calculate_midstep_triads(
    midstep_constraint_locations,
    nuctriads
)
        
Kentries = np.array([1,1,1,10,10,10])*1

diags = np.concatenate([Kentries]*len(nuc_mu0))
K1_10 = np.diag(diags)

left_open = 0
right_open = 0

print('##################################')
print('K_1-10')
nucout = binding_model_free_energy(
    groundstate,
    stiffmat,    
    nuc_mu0,
    K1_10,
    left_open=left_open,
    right_open=right_open,
    use_correction=True,
)
for key in nucout:
    if key in ['gs','alphas']:
        continue
    print(f'{key} = {nucout[key]}')
    
print(f'F_diff = {nucout["F"]-nucout["F_freedna"]}')


Kmd_comb     = np.load('MDParams/nuc_K_comb.npy')
Kmd_raw     = np.load('MDParams/nuc_K.npy')
Kmd_pos = np.load('MDParams/nuc_K_pos.npy')
Kmd_pos_resc = np.load('MDParams/nuc_K_posresc.npy')
def free_energy(K):
    sgn, slogdet = np.linalg.slogdet(K)
    return 0.5*slogdet

print(free_energy(Kmd_pos))
print(free_energy(Kmd_pos_resc))

print('##################################')
print('K_raw')

# Kmd     = np.load('MDParams/nuc_K_nsym.npy')
# Kmd_pos = np.load('MDParams/nuc_K_nsym_pos.npy')

nucout = binding_model_free_energy(
    groundstate,
    stiffmat,    
    nuc_mu0,
    Kmd_raw,
    left_open=left_open,
    right_open=right_open,
    use_correction=True,
)
for key in nucout:
    if key in ['gs','alphas']:
        continue
    print(f'{key} = {nucout[key]}')
print(f'F_diff = {nucout["F"]-nucout["F_freedna"]}')
    
    
print('##################################')
print('K_comb')
nucout = binding_model_free_energy(
    groundstate,
    stiffmat,    
    nuc_mu0,
    Kmd_comb,
    left_open=left_open,
    right_open=right_open,
    use_correction=True,
)
for key in nucout:
    if key in ['gs','alphas']:
        continue
    print(f'{key} = {nucout[key]}')
print(f'F_diff = {nucout["F"]-nucout["F_freedna"]}')

print('##################################')
print('K_pos')
nucout = binding_model_free_energy(
    groundstate,
    stiffmat,    
    nuc_mu0,
    Kmd_pos,
    left_open=left_open,
    right_open=right_open,
    use_correction=True,
)
for key in nucout:
    if key in ['gs','alphas']:
        continue
    print(f'{key} = {nucout[key]}')
print(f'F_diff = {nucout["F"]-nucout["F_freedna"]}')

print('##################################')
print('K_pos_resc')
nucout = binding_model_free_energy(
    groundstate,
    stiffmat,    
    nuc_mu0,
    Kmd_pos_resc,
    left_open=left_open,
    right_open=right_open,
    use_correction=True,
)
for key in nucout:
    if key in ['gs','alphas']:
        continue
    print(f'{key} = {nucout[key]}')
print(f'F_diff = {nucout["F"]-nucout["F_freedna"]}')


print('##################################')
print('K_pos_resc')

Kinf = np.copy(Kmd_comb)*1000000000000

nucout = binding_model_free_energy(
    groundstate,
    stiffmat,    
    nuc_mu0,
    Kinf,
    left_open=left_open,
    right_open=right_open,
    use_correction=True,
)
for key in nucout:
    if key in ['gs','alphas']:
        continue
    print(f'{key} = {nucout[key]}')
print(f'F_diff = {nucout["F"]-nucout["F_freedna"]}')


