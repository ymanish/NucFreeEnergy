import sys, os
import numpy as np
import scipy as sp
from typing import List, Tuple, Callable, Any, Dict

from .PolyCG.polycg.SO3 import so3
from .PolyCG.polycg.transform_SO3 import euler2rotmat_so3
from .PolyCG.polycg.transform_marginals import send_to_back_permutation
from .midstep_composites import midstep_composition_transformation, midstep_se3_groundstate
from .read_nuc_data import read_nucleosome_triads, GenStiffness



def nucleosome_free_energy(
    groundstate: np.ndarray,
    stiffmat: np.ndarray,
    midstep_constraint_locations: List[int],  # index of the lower (left-hand) triad neighboring the constraint midstep-triad
    nucleosome_triads: np.ndarray
) -> np.ndarray:
    
    midstep_constraint_locations = sorted(midstep_constraint_locations)
    
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
    const_1 = 0.5 * C.T @ MC @ C
    
    MMTC = MM.T @ C
    const_2 = -0.5 * MMTC.T @ MFi @ MMTC
    
    F_const = const_1+const_2
    
    # entropy term
    n = len(MF)
    F_pi = -0.5*n * np.log(2*np.pi)
    # matrix term
    logdet_sign, logdet = np.linalg.slogdet(MF)
    F_mat = 0.5*logdet
    
    # add up contributions
    F = F_mat + F_pi + F_const
    
    
    # print(f'F_pi    = {F_pi}')
    # print(f'F_const = {F_const}')
    # print(f'F_mat   = {F_mat}')
    # print(f'F       = {F}')
    
    # n = len(stiffmat)
    # F_pi = -0.5*n * np.log(2*np.pi)
    # logdet_sign, logdet = np.linalg.slogdet(stiffmat)
    # F_free = 0.5*logdet + F_pi
    # print(f'F_free  = {F_free}')
    

    return F, F_mat + F_pi, F_const
    

    
    
    


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

    # print('##########')
    # print(g_ij)
    # print(s_ij)    
    # print(d_ij)
    # print(X)
    # ex = np.copy(X)
    # ex[:3] *= 180./np.pi
    # print(ex)
    
    return X



if __name__ == '__main__':
    
    np.set_printoptions(linewidth=250,precision=3,suppress=True)
    
    genstiff = GenStiffness(method='MD')
    
    seq = ''.join(['ATCG'[np.random.randint(4)] for i in range(147)])
    seq601 = "ATCGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCGAT"
    seq = seq601
    
    randseq = 'TTCCACATGGATAATACAAGAGATTCATCGACGTGCTCATTTGGCATTAGGGCATCATCCTAATGAGATTCGGTGGCGCTAACAACTTCGCTGAAAGATCAGTGGAGCGAACTGCCCTACTGTTAATTGGGTACCAGACCTCCTCACATCGTTGGTAGCTCCGTTCCTCGCGGACCGCAAGGGCAAACGTCTTACGCGACATCTGTGAATCATAACTCAGTACTTTAAAGCTAGGGCGTATTATGCA'

    
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
    # midstep_constraint_locations = [
    #     55, 59, 
    #     65, 69, 76, 80, 86, 90, 
    #     96, 100, 107, 111, 116, 121, 
    #     128, 131, 139, 143
    # ]
    
    print(len(midstep_constraint_locations))
    
    
    F601,F_entrop, F_entalap = nucleosome_free_energy(
        gs,
        stiff,
        midstep_constraint_locations, 
        nuctriads
    )
    
    
    extended_601 = seq601 + seq601[:100]
    energies = []

    sweepseq = randseq
    probs_filename = 'randseq.probs'
    sweepseq = extended_601
    probs_filename = '601.probs'


    for i in range(len(sweepseq)-146):
        
        print(i)
        seq = sweepseq[i:i+147]
        
        stiff, gs = genstiff.gen_params(seq)
        F, F_entrop, F_entalap  = nucleosome_free_energy(
            gs,
            stiff,
            midstep_constraint_locations, 
            nuctriads
        )

        energies.append([F,F_entrop, F_entalap])
        print(F)
        
    
    
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
    
    energies = np.array(energies)
    
    Fparts = energies - np.mean(energies,axis=0)
    
    Epos   = np.arange(len(Fparts))
    ax1.plot(Epos,Fparts[:,0],lw=1,color='blue',zorder=2)
    ax2.plot(Epos,Fparts[:,1],lw=1,color='green')
    ax3.plot(Epos,Fparts[:,2],lw=1,color='red')
    
    ax1.plot(Epos,Fparts[:,2],lw=1,color='red',alpha=0.7,zorder=1)
    
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
    
    