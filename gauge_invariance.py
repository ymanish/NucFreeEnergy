import sys, os
num_threads = 4
os.environ["OMP_NUM_THREADS"] = f"{num_threads}"    # For libraries using OpenMP
os.environ["MKL_NUM_THREADS"] = f"{num_threads}"    # For Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = f"{num_threads}"  # For OpenBLAS
os.environ["BLIS_NUM_THREADS"] = f"{num_threads}"   # For BLIS

import numpy as np
import random
import matplotlib.pyplot as plt
from  methods import nucleosome_free_energy, nucleosome_groundstate, read_nucleosome_triads, GenStiffness, calculate_midstep_triads
from binding_model import binding_model_free_energy, binding_model_free_energy_old

def plot_dF(seqsdata,refs,gccont,seqs,savefn):

    def cm_to_inch(cm: float) -> float:
        return cm/2.54

    fig_width = 8.6
    fig_height = 10

    axlinewidth  = 0.8
    axtick_major_width  = 0.8
    axtick_major_length = 2.4
    axtick_minor_width  = 0.4
    axtick_minor_length = 1.6

    tick_pad        = 2
    tick_labelsize  = 5
    label_fontsize  = 6
    legend_fontsize = 6

    panel_label_fontsize = 8
    label_fontweight= 'bold'
    panel_label_fontweight= 'bold'

    fig = plt.figure(figsize=(cm_to_inch(fig_width), cm_to_inch(fig_height)), dpi=300,facecolor='w',edgecolor='k') 
    axes = []
    axes.append(fig.add_subplot(311))
    axes.append(fig.add_subplot(312))
    axes.append(fig.add_subplot(313))

    ##############################################
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]

    markersize = 5
    linewidth  = 0.6

    for i,K in enumerate(Kmats):
        
        dFs = [seqdata[i]['F']-refs[i]['F'] for seqdata in seqsdata]
        dFentrs = [seqdata[i]['F_entropy']-refs[i]['F_entropy'] for seqdata in seqsdata]
        dFenths = [seqdata[i]['F_enthalpy']-refs[i]['F_enthalpy'] for seqdata in seqsdata]
        
        ax1.plot(gccont,dFs,lw=linewidth,zorder=1,alpha=0.5)
        ax1.scatter(gccont,dFs,s=markersize,edgecolors='white',linewidth=0.5*linewidth,marker='o',zorder=2,alpha=0.75,label=f'K{i+1}')
        
        ax2.plot(gccont,dFentrs,lw=linewidth,zorder=1,alpha=0.5)
        ax2.scatter(gccont,dFentrs,s=markersize,edgecolors='white',linewidth=0.5*linewidth,marker='o',zorder=2,alpha=0.75,label=f'K{i+1}')
        
        ax3.plot(gccont,dFenths,lw=linewidth,zorder=1,alpha=0.5)
        ax3.scatter(gccont,dFenths,s=markersize,edgecolors='white',linewidth=0.5*linewidth,marker='o',zorder=2,alpha=0.75,label=f'K{i+1}')


    ax = axes[0]
    ax.set_xlabel('GC Content',fontsize=label_fontsize,fontweight=label_fontweight)
    ax.set_ylabel(r'$\mathbf{\Delta F}$',fontsize=label_fontsize,fontweight=label_fontweight,rotation=90)
    ax = axes[1]
    ax.set_xlabel('GC Content',fontsize=label_fontsize,fontweight=label_fontweight)
    ax.set_ylabel(r'$\mathbf{\Delta F_{\mathrm{entropy}}}$',fontsize=label_fontsize,fontweight=label_fontweight,rotation=90)
    ax = axes[2]
    ax.set_xlabel('GC Content',fontsize=label_fontsize,fontweight=label_fontweight)
    ax.set_ylabel(r'$\mathbf{\Delta F_{\mathrm{enthalpy}}}$',fontsize=label_fontsize,fontweight=label_fontweight,rotation=90)

    for ax in axes:
        ax.xaxis.set_label_coords(0.5,-0.1)
        ax.yaxis.set_label_coords(-0.055,0.5)
        ax.set_xlim([0,1])

    # ax1.legend(fontsize=legend_fontsize)

    ##############################################
    # Axes configs
    for ax in axes:
        ###############################
        # set major and minor ticks
        ax.tick_params(axis="both",which='major',direction="in",width=axtick_major_width,length=axtick_major_length,labelsize=tick_labelsize,pad=tick_pad,color='#cccccc')
        ax.tick_params(axis='both',which='minor',direction="in",width=axtick_minor_width,length=axtick_minor_length,color='#cccccc')

        ###############################
        ax.xaxis.set_ticks_position('both')
        # set ticks right and top
        ax.yaxis.set_ticks_position('both')
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(axlinewidth)
            ax.spines[axis].set_color('grey')
            ax.spines[axis].set_alpha(0.7)
            
    plt.subplots_adjust(left=0.09,
                        right=0.98,
                        bottom=0.05,
                        top=0.99,
                        wspace=6,
                        hspace=0.25)
            
    fig.savefig(savefn+'.pdf',dpi=300,transparent=True)
    fig.savefig(savefn+'.svg',dpi=300,transparent=True)
    fig.savefig(savefn+'.png',dpi=300,transparent=False)








def random_seq(N,gc=0.5):
    NGC = int(N*gc)
    NAT = N-NGC
    seqlist = ['AT'[np.random.randint(2)] for i in range(NAT)] + ['CG'[np.random.randint(2)] for i in range(NGC)]
    random.shuffle(seqlist)
    seq = ''.join(seqlist)
    return seq


genstiff = GenStiffness(method='hybrid')   # alternatively you can use the 'crystal' method for the Olson data
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

Kentries = np.array([1,1,1,10,10,10])

diags = np.concatenate([Kentries]*len(nuc_mu0))
Kbase = np.diag(diags)

Kmats = []
Kmats.append(Kbase*10)
Kmats.append(Kbase*1)
Kmats.append(Kbase*0.1)
Kmats.append(Kbase*0.01)
Kmats.append(Kbase*0.001)


######################
# ref vals
seq601  = "CTGGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGT"
refseq = seq601

Nseqs = 100
seqs = [''.join(['ATCG'[np.random.randint(4)] for i in range(147)]) for j in range(Nseqs)]

Nseqs = 101
# Nseqs = 5
seqs = []
gccont = []
for i in range(Nseqs):
    gc = i/(Nseqs-1)
    seq = random_seq(147,gc)
    seqs.append(seq)
    gccont.append(gc)


for nopen in [0,4,8,12]:
    
    left_open  = nopen
    right_open = nopen

    stiffmat,groundstate = genstiff.gen_params(refseq,use_group=True)
    refs = []
    for K in Kmats:
        nucout = binding_model_free_energy(
            groundstate,
            stiffmat,    
            nuc_mu0,
            K,
            left_open=left_open,
            right_open=right_open,
            use_correction=True,
        )
        refs.append(nucout)
        
    seqsdata = []
    for i,seq in enumerate(seqs):
        print(i)
        stiffmat,groundstate = genstiff.gen_params(seq,use_group=True)
        seqdata = []
        for K in Kmats:
            nucout = binding_model_free_energy(
                groundstate,
                stiffmat,    
                nuc_mu0,
                K,
                left_open=left_open,
                right_open=right_open,
                use_correction=True,
            )
            seqdata.append(nucout)
        seqsdata.append(seqdata)
    
    savefn = f'Figs/GaugeInvariance_l{right_open}_r{right_open}'
    plot_dF(seqsdata,refs,gccont,seqs,savefn)


