import sys, os
import numpy as np
import scipy as sp
from typing import List, Tuple, Callable, Any, Dict

from .PolyCG.polycg.SO3 import so3
from .PolyCG.polycg.Transforms.transform_SO3 import euler2rotmat_so3
from .PolyCG.polycg.Transforms.transform_marginals import send_to_back_permutation
from .midstep_composites import midstep_composition_transformation, midstep_se3_groundstate
from .read_nuc_data import read_nucleosome_triads, GenStiffness

from .PolyCG.polycg.cgnaplus import cgnaplus_bps_params


def trap_calculation_simple(M: np.ndarray, C: np.ndarray, mu: float) -> Tuple[float,float, float]:
    
    N  = len(M)
    Nz = len(C) 
    Ny = N-Nz
    
    My  = M[:Ny,:Ny]
    Mz  = M[Ny:,Ny:]
    Myz = M[Ny:,:Ny]
    
    M_nuc = Mz*mu
    
    M_chi_nuc = np.copy(M)
    M_chi_nuc[Ny:,Ny:] += M_nuc
    
    b = np.zeros(N)
    b[Ny:] = -M_nuc @ C
    
    c1 = 0.5 * C.T @ M_nuc @ C
    c2 = -0.5 * b.T @ np.linalg.inv(M_chi_nuc) @ b
    
    betaF_entropy = 0.5 * np.linalg.slogdet(M_chi_nuc)[1]
    betaF_enthalpy = c1 + c2
    
    betaF = betaF_entropy + betaF_enthalpy
    return betaF, betaF_entropy, betaF_enthalpy


def trap_calculation(M: np.ndarray, C: np.ndarray, mu: float) -> Tuple[float,float, float]:
    
    N  = len(M)
    Nz = len(C) 
    Ny = N-Nz
    
    My  = M[:Ny,:Ny]
    Mz  = M[Ny:,Ny:]
    Mzy = M[Ny:,:Ny]
    Myz = M[:Ny,Ny:]
    
    M_nuc = Mz*mu
    # M_nuc = np.eye(len(Mz)) * mu
    
    M_chi_nuc = np.copy(M)
    M_chi_nuc[Ny:,Ny:] += M_nuc
    
    Dy = - np.linalg.inv(M_chi_nuc[:Ny,:Ny]) @ M_chi_nuc[:Ny,Ny:] @ C
    
    C0 =  np.linalg.inv(M_nuc.T) @ M_chi_nuc[Ny:,:Ny] @ Dy + np.linalg.inv(M_nuc.T) @ M_chi_nuc[Ny:,Ny:] @ C
    
    # print(np.linalg.inv(M_nuc.T) @ M_chi_nuc[Ny:,Ny:])
    
    b = np.zeros(N)
    b[Ny:] = -M_nuc @ C0
    
    # D = -np.linalg.inv(M_chi_nuc) @ b
    # print(D[Ny:]-C)
    
    c1 = 0.5 * C0.T @ M_nuc @ C0
    c2 = -0.5 * b.T @ np.linalg.inv(M_chi_nuc) @ b
    
    betaF_entropy = 0.5 * np.linalg.slogdet(M_chi_nuc)[1]
    betaF_enthalpy = c1 + c2
    
    betaF = betaF_entropy + betaF_enthalpy
    return betaF, betaF_entropy, betaF_enthalpy
  
    

def nucleosome_free_energy(
    groundstate: np.ndarray,
    stiffmat: np.ndarray,
    midstep_constraint_locations: List[int],  # index of the lower (left-hand) triad neighboring the constraint midstep-triad
    nucleosome_triads: np.ndarray
) -> np.ndarray:
    
    if len(midstep_constraint_locations) == 0:
        n = len(stiffmat)
        F_pi = -0.5*n * np.log(2*np.pi)
        # matrix term
        logdet_sign, logdet = np.linalg.slogdet(stiffmat)
        F_mat = 0.5*logdet
        F = F_mat + F_pi  
        return F, F, 0, 0
    
    
    midstep_constraint_locations = sorted(list(set(midstep_constraint_locations)))

    midstep_triads = calculate_midstep_triads(
        midstep_constraint_locations,
        nucleosome_triads
    )
    
    # find contraint excess values
    excess_vals = midstep_excess_vals(
        groundstate,
        midstep_constraint_locations,
        midstep_triads
    )
    C = excess_vals.flatten()
        
    # find composite transformation
    transform, replaced_ids = midstep_composition_transformation(
        groundstate,
        midstep_constraint_locations
    )
    
    # transform stiffness matrix
    inv_transform = np.linalg.inv(transform)
    stiffmat_transformed = inv_transform.T @ stiffmat @ inv_transform
    
    # rearrange stiffness matrix
    full_replaced_ids = list()
    for i in range(len(replaced_ids)):
        full_replaced_ids += [6*replaced_ids[i]+j for j in range(6)]
     
    P = send_to_back_permutation(len(stiffmat),full_replaced_ids)
    stiffmat_rearranged = P @ stiffmat_transformed @ P.T

    # select fluctuating, constraint and coupling part of matrix
    N  = len(stiffmat)
    NC = len(full_replaced_ids)
    NF = N-NC
    
    MF = stiffmat_rearranged[:NF,:NF]
    MC = stiffmat_rearranged[NF:,NF:]
    MM = stiffmat_rearranged[NF:,:NF]
    
    MFi = np.linalg.inv(MF)
    b = MM.T @ C
    
    # constant energies
    F_const_C =  0.5 * C.T @ MC @ C
    F_const_b = -0.5 * b.T @ MFi @ b
    
    # entropy term
    n = len(MF)
    logdet_sign, logdet = np.linalg.slogdet(MF)
    F_pi = -0.5*n * np.log(2*np.pi)
    # matrix term
    F_mat = 0.5*logdet
    F_entropy = F_pi + F_mat
    F_jacob = np.log(np.linalg.det(transform))
    
    Fdict = {
        'F': F_entropy + F_jacob + F_const_C + F_const_b,
        'F_entropy' : F_entropy,
        'F_const'   : F_const_C + F_const_b,
        'F_jacob'   : F_jacob
    }
    return Fdict
    


def calculate_midstep_triads(
    triad_ids: List[int],  # index of the lower (left-hand) triad neighboring the constraint midstep-triad
    nucleosome_triads: np.ndarray
) -> np.ndarray:
    midstep_triads = np.zeros((len(triad_ids),4,4))
    for i,id in enumerate(triad_ids):
        T1 = nucleosome_triads[id]
        T2 = nucleosome_triads[id+1]
        midstep_triads[i,:3,:3] = T1[:3,:3] @ so3.euler2rotmat(0.5*so3.rotmat2euler(T1[:3,:3].T @ T2[:3,:3]))
        midstep_triads[i,:3,3]  = 0.5* (T1[:3,3]+T2[:3,3])
        midstep_triads[i,3,3]   = 1
    return midstep_triads
    

def midstep_excess_vals(
    groundstate: np.ndarray,
    midstep_constraint_locations: List[int],
    midstep_triads: np.ndarray  
):
    
    num = len(midstep_constraint_locations)-1
    excess_vals = np.zeros((num,6))
    for i in range(num):
        id1 = midstep_constraint_locations[i]
        id2 = midstep_constraint_locations[i+1]
        triad1 = midstep_triads[i]
        triad2 = midstep_triads[i+1]
        partial_gs = groundstate[id1:id2+1] 
        excess_vals[i] = midstep_composition_excess(partial_gs,triad1,triad2) 
    return excess_vals
    
    


def midstep_composition_excess(
    groundstate: np.ndarray,
    triad1: np.ndarray,
    triad2: np.ndarray
) -> np.ndarray:
    g_ij = np.linalg.inv(triad1) @ triad2
    Smats = midstep_se3_groundstate(groundstate)
    s_ij = np.eye(4)
    for Smat in Smats:
        s_ij = s_ij @ Smat
    d_ij = np.linalg.inv(s_ij) @ g_ij
    X = so3.se3_rotmat2euler(d_ij)
    return X



if __name__ == '__main__':
    
    np.set_printoptions(linewidth=250,precision=3,suppress=True)
    
    genstiff = GenStiffness(method='MD')
    
    seq = ''.join(['ATCG'[np.random.randint(4)] for i in range(147)])
    seq601 = "ATCGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCGAT"
    randseq = 'TTCCACATGGATAATACAAGAGATTCATCGACGTGCTCATTTGGCATTAGGGCATCATCCTAATGAGATTCGGTGGCGCTAACAACTTCGCTGAAAGATCAGTGGAGCGAACTGCCCTACTGTTAATTGGGTACCAGACCTCCTCACATCGTTGGTAGCTCCGTTCCTCGCGGACCGCAAGGGCAAACGTCTTACGCGACATCTGTGAATCATAACTCAGTACTTTAAAGCTAGGGCGTATTATGCA'
    
    # seq = randseq
    seq = seq601
    
    beta = 1./4.114
    
    stiff,gs = genstiff.gen_params(seq)
    
    triadfn = os.path.join(os.path.dirname(__file__), 'State/Nucleosome.state')
    nuctriads = read_nucleosome_triads(triadfn)

    midstep_constraint_locations = [
        2, 6, 14, 17, 24, 29, 
        34, 38, 45, 49, 55, 59, 
        65, 69, 76, 80, 86, 90, 
        96, 100, 107, 111, 116, 121, 
        128, 131, 139, 143
    ]
        
    
    extended_601 = seq601 + seq601[:100]
    Fdicts = []

    sweepseq = randseq
    probs_filename = 'randseq.probs'
    sweepseq = extended_601
    probs_filename = '601.probs'


    for i in range(len(sweepseq)-146):
        
        print(i)
        seq = sweepseq[i:i+147]
        
        stiff, gs = genstiff.gen_params(seq)
        
        # gs,stiff = cgnaplus_bps_params(seq,euler_definition=True,group_split=True)
        
        Fdict  = nucleosome_free_energy(
            gs,
            stiff,
            midstep_constraint_locations, 
            nuctriads
        )
        Fdicts.append(Fdict)

    
    probs = np.loadtxt(probs_filename)
    betaE = -np.log(probs)
    betaE -= np.mean(betaE)
    pos   = np.arange(len(betaE))
        
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8.6/2.54,10./2.54))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    ax1.plot(pos,betaE,lw=1.4,color='black',zorder=0)
    
    Ftots = np.array([Fdict['F'] for Fdict in Fdicts])
    Fcnst = np.array([Fdict['F_const'] for Fdict in Fdicts])
    Fentr = np.array([Fdict['F_entropy'] for Fdict in Fdicts])
    
    Ftots_rel = Ftots - np.mean(Ftots)
    Fcnst_rel = Fcnst - np.mean(Fcnst)
    Fentr_rel = Fentr - np.mean(Fentr)
        
    Epos   = np.arange(len(Ftots_rel))
    ax1.plot(Epos,Ftots_rel,lw=1,color='blue',zorder=2)
    ax2.plot(Epos,Fentr_rel,lw=1,color='green')
    ax3.plot(Epos,Fcnst_rel,lw=1,color='red')
    
    ax1.plot(Epos,Fcnst_rel,lw=1,color='red',alpha=0.7,zorder=1)
    
    tick_pad            = 2
    axlinewidth         = 0.9
    axtick_major_width  = 0.6
    axtick_major_length = 1.6
    tick_labelsize      = 6
    label_fontsize      = 7
    
    ax1.set_xlabel('Nucleosome Position',size = label_fontsize,labelpad=1)
    ax1.set_ylabel(r'$\beta E$',size = label_fontsize,labelpad=1)
    ax2.set_xlabel('Nucleosome Position',size = label_fontsize,labelpad=1)
    ax2.set_ylabel(r'$\beta E_{\mathrm{entropic}}$',size = label_fontsize,labelpad=1)
    ax3.set_xlabel('Nucleosome Position',size = label_fontsize,labelpad=1)
    ax3.set_ylabel(r'$\beta E_{\mathrm{enthalpic}}$',size = label_fontsize,labelpad=1)
    
    ax1.tick_params(axis="both",which='major',direction="in",width=axtick_major_width,length=axtick_major_length,labelsize=tick_labelsize,pad=tick_pad)
    ax2.tick_params(axis="both",which='major',direction="in",width=axtick_major_width,length=axtick_major_length,labelsize=tick_labelsize,pad=tick_pad)
    ax3.tick_params(axis="both",which='major',direction="in",width=axtick_major_width,length=axtick_major_length,labelsize=tick_labelsize,pad=tick_pad)
        
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.7)
        ax2.spines[axis].set_linewidth(0.7)
        ax3.spines[axis].set_linewidth(0.7)
        

    
    plt.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.06,
        top=0.98,
        wspace=0.2,
        hspace=0.26)
    
    plt.savefig(f'Figs/{probs_filename.split(".")[0]}.png',dpi=300,facecolor='white')
    plt.close()
    
    