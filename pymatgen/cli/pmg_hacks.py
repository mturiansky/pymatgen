from __future__ import division, unicode_literals
from matplotlib import pyplot as plt
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.io.vasp.outputs import Vasprun, Procar
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.electronic_structure.core import Spin
from matplotlib.collections import LineCollection
from itertools import repeat
import pymatgen as pmg
import numpy as np
import os

'''
a convenience script for doing things I do all the damn time
'''


__author__ = "Mark Turiansky"
__copyright__ = "Copyright 2018, Mark Turiansky"
__version__ = "4.0"
__maintainer__ = "Shyue Ping Ong"
__email__ = "ongsp@ucsd.edu"
__date__ = "May 08 2018"


########################################################################################
# Plot the projected band structure
########################################################################################

def plot_sp_proj(fold_name, sym_kpt,ax,xshift=0.0,bg=None, orbs='spd'):
    #fold_name="/home/lik/Documents/Data/comet/WBG_ord/bulk_calcs/AlN_wz/hse/band_struct"
    print (fold_name)
    bands = Vasprun(fold_name+"/vasprun.xml").get_band_structure(sym_kpt, line_mode=True)
    plotter=BSPlotter(bands)
    bs_data=plotter.bs_plot_data(zero_to_efermi=True)
    #bands.apply_scissor(6.18)
    kdist=np.empty(0)
    eng=np.empty(0)
    # bandstructure data
    bs_dist=bs_data['distances']
    bs_en=bs_data['energy']

    dl = np.sum([len(x) for x in bs_dist])

    # projected bands
    #obj=Procar("./data/hse_band_BN/PROCAR")
    pro_file=fold_name+"/PROCAR"
    bs_procar = Procar(pro_file).data[Spin.up][-dl:]
    # s_orbital
    s_orb=bs_procar[:,:,:,0]
    s_orb_sum=np.sum(s_orb,axis=2).T

    p_orb=np.sum(bs_procar[:,:,:,1:4],axis=3)
    p_orb_sum=np.sum(p_orb,axis=2).T


    d_orb=np.sum(bs_procar[:,:,:,4:9],axis=3)
    d_orb_sum=np.sum(d_orb,axis=2).T

    tot_orb=np.sum(bs_procar,axis=3)
    tot_orb_sum=np.sum(tot_orb,axis=2).T


    for ibranch in enumerate(bs_dist):
        branch_dist=np.array(ibranch[1])
        bs_en[ibranch[0]]
        bs_en_up=np.array(bs_en[ibranch[0]]['1'])
        #print ' dist', dist.shape
        #print 'kdist', kdist.shape
        kdist=np.hstack([kdist, branch_dist]) if kdist.size else branch_dist
        kdist=kdist+xshift
        eng=np.hstack([eng, bs_en_up]) if eng.size else bs_en_up

    # spline interpolating the each band
    from scipy.interpolate import interp1d
    xx=np.linspace(min(kdist),max(kdist),200)

    mask=(kdist[:-1]-kdist[1:]!=0)
    mask=np.hstack([mask,True])

    bounds = np.linspace(-1,1,50)

    sval = []
    pval = []
    dval = []
    for iband in enumerate(eng):
        itr = iband[0]
        Sorb_funcs = interp1d(kdist[mask], s_orb_sum[itr][mask], kind='cubic')
        Porb_funcs = interp1d(kdist[mask], p_orb_sum[itr][mask], kind='cubic')
        Dorb_funcs = interp1d(kdist[mask], d_orb_sum[itr][mask], kind='cubic')
        Torb_funcs = interp1d(kdist[mask], tot_orb_sum[itr][mask], kind='cubic')
        sval.extend(Sorb_funcs(xx[:-1])/Torb_funcs(xx[:-1]))
        pval.extend(Porb_funcs(xx[:-1])/Torb_funcs(xx[:-1]))
        dval.extend(Dorb_funcs(xx[:-1])/Torb_funcs(xx[:-1]))
    svalmax = np.max(sval)
    svalmin = np.min(sval)
    pvalmax = np.max(pval)
    pvalmin = np.min(pval)
    dvalmax = np.max(dval)
    dvalmin = np.min(dval)

    for iband in enumerate(eng):
        itr=iband[0]
        band_func = interp1d(kdist[mask], iband[1][mask], kind='cubic')
        Sorb_funcs = interp1d(kdist[mask], s_orb_sum[itr][mask], kind='cubic')
        Porb_funcs = interp1d(kdist[mask], p_orb_sum[itr][mask], kind='cubic')
        Dorb_funcs = interp1d(kdist[mask], d_orb_sum[itr][mask], kind='cubic')

        Torb_funcs = interp1d(kdist[mask], tot_orb_sum[itr][mask], kind='cubic')
        #
        t_orb_val=Torb_funcs(xx[:-1])
        s_orb_val=Sorb_funcs(xx[:-1])
        p_orb_val=Porb_funcs(xx[:-1])
        d_orb_val=Dorb_funcs(xx[:-1])

        # normals = (s_orb_val - np.min(s_orb_val)) /\
        #     np.max(s_orb_val - np.min(s_orb_val))
        # normalp = (p_orb_val - np.min(p_orb_val)) /\
        #     np.max(p_orb_val - np.min(p_orb_val))
        # normals = (s_orb_val - np.min(s_orb_val)) / np.max(s_orb_sum)
        # normalp = (p_orb_val - np.min(p_orb_val)) / np.max(p_orb_sum)
        normals = (s_orb_val/t_orb_val - svalmin) / (svalmax - svalmin)
        normalp = (p_orb_val/t_orb_val - pvalmin) / (pvalmax - pvalmin)
        normald = (d_orb_val/t_orb_val - dvalmin) / (dvalmax - dvalmin)
        if orbs == 'spd':
            colors = list(zip(normals, normald, normalp, repeat(1.)))
        elif orbs == 'sp':
            colors = list(zip(normals, repeat(0.), normalp, repeat(1.)))
        elif orbs == 'sd':
            colors = list(zip(normals, normald, repeat(0.), repeat(1.)))
        elif orbs == 'pd':
            colors = list(zip(repeat(0.), normald, normalp, repeat(1.)))
        else:
            colors = list(zip(repeat(0.), repeat(0.), repeat(0.), repeat(1.)))

        lwidths=2.
        points = np.array([xx, band_func(xx)]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-2], points[2:]], axis=1)
        lc = LineCollection(segments, linewidths=lwidths, colors=colors)
        ax.add_collection(lc)
        #ax.plot(xx,band_func(xx),'k-')
        #ax.plot(kdist[mask], iband[1][mask], 'ko', ms=2)
    tix_dict=bs_data['ticks']


    print (tix_dict)
    ax.set_xticks(np.array(tix_dict['distance'])+xshift)
    ax.set_xlim(tix_dict['distance'][0]+xshift,tix_dict['distance'][-1]+xshift)
    ax.set_xticklabels(tix_dict['label'])
    for xtpt in tix_dict[u'distance']:
        ax.axvline(xtpt+xshift,-100,100,color='k',lw=1.,zorder=0)
    return tix_dict[u'distance'][-1]+xshift

def get_bs_plot(args):
    if not os.path.isfile(os.path.join(args.bs_dir, 'KPOINTS')):
        print('[-] no KPOINTS in', args.bs_dir)
    if not os.path.isfile(os.path.join(args.bs_dir, 'PROCAR')):
        print('[-] no PROCAR in', args.bs_dir)
    if not os.path.isfile(os.path.join(args.bs_dir, 'vasprun.xml')):
        print('[-] no vasprun.xml in', args.bs_dir)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_sp_proj(args.bs_dir, os.path.join(args.bs_dir, 'KPOINTS'), ax)
    if args.ylim:
        ax.set_ylim(args.ylim)
    else:
        ax.set_ylim([-5, 15])
    return plt


def make_bs_kpoint(ifile, ikpath, line_density, updatef):
    if ifile.split('/')[-1] == 'CONTCAR' or ifile.split('/')[-1] == 'POSCAR':
        s = pmg.Structure.from_file(ifile)
    if ifile.split('/')[-1] == 'vasprun.xml':
        s = Vasprun(ifile).structures[-1]

    hsk = HighSymmKpath(s)
    if ikpath == '':
        print('[+] kpoints:', list(hsk.kpath['kpoints'].keys()))
        print('[+] usual path: \'',
              ';'.join([','.join(x) for x in hsk.kpath['path']]),
              '\'', sep='')
        return
    elif ikpath.lower() == 'generate':
        kpath = hsk.kpath['path']
    else:
        kpath = [x.split(',') for x in ikpath.split(';')]
        for x in kpath:
            for y in x:
                if y not in list(hsk.kpath['kpoints'].keys()):
                    print('[-] bad input, valid high symmetry points to choose'
                          ' from:', list(hsk.kpath['kpoints'].keys()))
                    return
    kpts = []
    labels = []
    for kp in kpath:
        for k0, k1 in zip(kp[:-1], kp[1:]):
            k0c = hsk.kpath['kpoints'][k0]
            k1c = hsk.kpath['kpoints'][k1]
            dist = np.linalg.norm(hsk._prim_rec.get_cartesian_coords(k0c)
                                  - hsk._prim_rec.get_cartesian_coords(k1c))
            nb = int(np.ceil(dist * line_density))
            kpts.extend([k0c*(1-x/nb) + k1c*x/nb for x in range(nb + 1)])
            labels.extend([k0] + ['']*(nb - 1) + [k1])

    wt = 0
    if updatef and os.path.isfile(updatef):
        print('write to file')
        os.rename(updatef, updatef + '.bak')
        with open(updatef, 'w') as out:
            with open(updatef + '.bak', 'r') as inp:
                for i, line in enumerate(inp):
                    if i == 1:
                        num = int(line)
                        print(num)
                        out.write('\t' + str(num + len(kpts)) + '\n')
                    else:
                        out.write(line)
            for kpt, label in zip(kpts, labels):
                out.write(f'    {kpt[0]:16.014f}    {kpt[1]:16.014f}    '
                          f'{kpt[2]:16.014f}{wt: >14}  {label}\n')
    else:
        for kpt, label in zip(kpts, labels):
            print(f'    {kpt[0]:16.014f}    {kpt[1]:16.014f}    '
                  f'{kpt[2]:16.014f}{wt: >14}  {label}')


def kpoint(args):
    if args.bs_file:
        make_bs_kpoint(args.bs_file, args.path, args.line_density, args.update)
