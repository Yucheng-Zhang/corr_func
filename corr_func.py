'''
Compute 2D correlation function.
'''
import numpy as np
import pandas as pd
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
from cosmo import cosmo_LCDM
import argparse


def save_pc(fn, pc):
    header = 'smin   smax   savg   mumax   npairs   weightavg'
    data = np.array([[p['smin'], p['smax'], p['savg'],
                      p['mumax'], p['npairs'], p['weightavg']]for p in pc])
    np.savetxt(fn, data, header=header)


def load_data_w_pd(fn):
    tb = pd.read_table(fn, delim_whitespace=True, comment='#', header=None)
    return tb.values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute 2D correlation function.')
    parser.add_argument('-data', default='',
                        help='Input data file: RA, DEC, Z, weight')
    parser.add_argument('-rand', default='',
                        help='Input random file: RA, DEC, Z, weight')
    parser.add_argument('-H0', type=float, default=70.0,
                        help='H0 for fiducial cosmology')
    parser.add_argument('-Om0', type=float, default=0.25,
                        help='OmegaM_0 for fiducial cosmology')
    parser.add_argument('-ncpu', type=int, default=1, help='Number of CPUs')

    parser.add_argument('-mu_max', type=float, default=1.0, help='Maximum mu')
    parser.add_argument('-n_mu_bins', type=int,
                        default=100, help='Number of linear mu bins')

    parser.add_argument('-s_min', type=float, default=1.0,
                        help='Minimum s in [Mpc/h]')
    parser.add_argument('-s_max', type=float, default=160.0,
                        help='Maximum s in [Mpc/h]')
    parser.add_argument('-n_s_bins', type=int, default=32,
                        help='Number of s bins')
    args = parser.parse_args()

    print('>> Loading data file: {}'.format(args.data))
    data = load_data_w_pd(args.data)
    print('>> Loading random file: {}'.format(args.rand))
    rand = load_data_w_pd(args.rand)

    print('>> Getting comoving distance from redshift...')
    print('>> fiducial cosmology: H0 = {0:f}, OmegaM_0 = {1:f}'.format(
        args.H0, args.Om0))
    cosmo = cosmo_LCDM(args.H0, args.Om0, 1. - args.Om0)
    data[:, 2] = cosmo.z2chi(data[:, 2])
    rand[:, 2] = cosmo.z2chi(rand[:, 2])

    print('>> Getting bin file for s...')
    binfile = np.linspace(args.s_min, args.s_max, args.n_s_bins + 1)

    print('>> Computing DD pair count...')
    DD = DDsmu_mocks(autocorr=1, cosmology=1, nthreads=args.ncpu,
                     mu_max=1.0, nmu_bins=args.n_mu_bins, binfile=binfile,
                     RA1=data[:, 0], DEC1=data[:, 1],
                     CZ1=data[:, 2], weights1=data[:, 3],
                     weight_type='pair_product', is_comoving_dist=True,
                     output_savg=True, verbose=True)
    save_pc('DD_test.dat', DD)

    print('>> Computing RR pair count...')
    RR = DDsmu_mocks(autocorr=1, cosmology=1, nthreads=args.ncpu,
                     mu_max=1.0, nmu_bins=args.n_mu_bins, binfile=binfile,
                     RA1=rand[:, 0], DEC1=rand[:, 1],
                     CZ1=rand[:, 2], weights1=rand[:, 3],
                     weight_type='pair_product', is_comoving_dist=True,
                     output_savg=True, verbose=True)
    save_pc('RR_test.dat', RR)

    print('>> Computing DR pair count...')
    DR = DDsmu_mocks(autocorr=0, cosmology=1, nthreads=args.ncpu,
                     mu_max=1.0, nmu_bins=args.n_mu_bins, binfile=binfile,
                     RA1=data[:, 0], DEC1=data[:, 1],
                     CZ1=data[:, 2], weights1=data[:, 3],
                     RA2=rand[:, 0], DEC2=rand[:, 1],
                     CZ2=rand[:, 2], weights2=rand[:, 3],
                     weight_type='pair_product', is_comoving_dist=True,
                     output_savg=True, verbose=True)
    save_pc('DR_test.dat', DR)

    print('>> Converting pair count to correlation function...')
