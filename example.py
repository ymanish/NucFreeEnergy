import numpy as np 
from  methods import nucleosome_free_energy, read_nucleosome_triads, GenStiffness

genstiff = GenStiffness(method='MD')   # alternatively you can use the 'crystal' method for the Olson data
seq601 = "ATCGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCGAT"

stiffmat,groundstate = genstiff.gen_params(seq601,use_group=True)
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