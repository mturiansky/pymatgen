from __future__ import division, unicode_literals
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.symmetry.bandstructure import HighSymmKpath
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


def make_bs_kpoint(ifile, ikpath, line_density, print_hs_path, updatef):
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
        make_bs_kpoint(args.bs_file, args.path, args.line_density,
                       args.highsymmetry, args.update)
