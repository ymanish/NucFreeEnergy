import numpy as np 
from  methods import nucleosome_free_energy, nucleosome_groundstate, read_nucleosome_triads, GenStiffness

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
triadfn = 'methods/State/Nucleosome.state'
nuctriads = read_nucleosome_triads(triadfn)

midstep_constraint_locations = [
    2, 6, 14, 17, 24, 29, 
    34, 38, 45, 49, 55, 59, 
    65, 69, 76, 80, 86, 90, 
    96, 100, 107, 111, 116, 121, 
    128, 131, 139, 143
]

Fdict = nucleosome_free_energy(groundstate,stiffmat,midstep_constraint_locations,nuctriads)
print(Fdict)

gs = nucleosome_groundstate(groundstate,stiffmat,midstep_constraint_locations,nuctriads)
gs = gs.reshape(len(gs)//6,6)
print(10*'\n')
# print(gs)
print('Analytical gs')
for i in range(len(gs)):
    pstr = f'{i} {gs[i]}'
    if i in midstep_constraint_locations:
        pstr += ' mid'
    print(pstr)
    
gs_num = np.load('gs_601.npy')

print(10*'\n')
print('Numerical gs')
# print(gs_num)
for i in range(len(gs)):
    pstr = f'{i} {gs_num[i]}'
    if i in midstep_constraint_locations:
        pstr += ' mid'
    print(pstr)
    
    
print(10*'\n')
print('difference')
# print(gs_num)
for i in range(len(gs)):
    pstr = f'{i} {gs[i]-gs_num[i]}'
    if i in midstep_constraint_locations:
        pstr += ' mid'
    print(pstr)
    
print(10*'\n')
print('difference norms')
# print(gs_num)
for i in range(len(gs)):
    diff = gs[i]-gs_num[i]
    dr = np.linalg.norm(diff[:3])
    dt = np.linalg.norm(diff[3:])
    pstr = '%d %.3f %.3f'%(i,dr,dt)
    if dr > 0.05 or dt > 0.05:
        pstr += ' <--'
    if i in midstep_constraint_locations:
        pstr += ' mid'
    print(pstr)
    
