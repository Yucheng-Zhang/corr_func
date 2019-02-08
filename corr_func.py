'''
Compute 2D correlation function.
'''
import numpy as np
import pandas as pd
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
from cosmo import cosmo_LCDM
import argparse

if __name__ == '__main__':
    '''Input arguments.'''
    parser = argparse.ArgumentParser(
        description='Compute 2D correlation function.')
    parser.add_argument('-data', default='',
                        help='Input data file: RA, DEC, Z, weight')
    parser.add_argument('-rand', default='',
                        help='Input random file: RA, DEC, Z, weight')
    parser.add_argument('-H0', type=float, default=70.0,
                        help='H0 for fiducial cosmology')
    parser.add_argument('-Om0', type=float, default=0.3,
                        help='OmegaM_0 for fiducial cosmology')
    parser.add_argument('-ncpu', type=int, default=1, help='Number of CPUs')

    parser.add_argument('-mu_max', type=float, default=1.0, help='Maximum mu')
    parser.add_argument('-n_mu_bins', type=int,
                        default=100, help='Number of linear mu bins')

    parser.add_argument('-s_min', type=float, default=0.1,
                        help='Minimum s in [Mpc/h]')
    parser.add_argument('-s_max', type=float, default=160.1,
                        help='Maximum s in [Mpc/h]')
    parser.add_argument('-n_s_bins', type=int, default=32,
                        help='Number of s bins')
    args = parser.parse_args()


def save_pc(fn, pc):
    '''Save the result of DDsmu_mocks.'''
    header = 'smin   smax   savg   mumax   npairs   weightavg'
    data = np.array([[p['smin'], p['smax'], p['savg'],
                      p['mumax'], p['npairs'], p['weightavg']]for p in pc])
    np.savetxt(fn, data, header=header)


def save_smu_arr(fn, smu_arr, s_bins, mu_bins):
    '''Save s mu 2D array.'''
    header = 'xi(s, mu) with same s for each row\n'
    header += '{0:d} s bins, with edges:\n'.format(
        len(s_bins) - 1) + np.array_str(s_bins) + '\n'
    header += '{0:d} mu bins, with edges:\n'.format(
        len(mu_bins) - 1) + np.array_str(mu_bins)
    np.savetxt(fn, smu_arr, header=header)


def load_data_w_pd(fn):
    '''Read input data with pandas.'''
    tb = pd.read_table(fn, delim_whitespace=True, comment='#', header=None)
    return tb.values


if __name__ == '__main__':

    print('>> Loading data file: {}'.format(args.data))
    data = load_data_w_pd(args.data)
    print('>> Loading random file: {}'.format(args.rand))
    rand = load_data_w_pd(args.rand)

    print('>> Getting comoving distance from redshift')
    print('>> fiducial cosmology: H0 = {0:f}, OmegaM_0 = {1:f}'.format(
        args.H0, args.Om0))
    cosmo = cosmo_LCDM(args.H0, args.Om0, 1. - args.Om0)
    data[:, 2] = cosmo.z2chi(data[:, 2])
    rand[:, 2] = cosmo.z2chi(rand[:, 2])

    print('>> Getting bin file for s')
    binfile = np.linspace(args.s_min, args.s_max, args.n_s_bins + 1)

    data[:, 3] = np.ones(len(data[:, 3]))
    rand[:, 3] = np.ones(len(rand[:, 3]))

    print('>> Computing DD pair count')
    DD = DDsmu_mocks(autocorr=1, cosmology=1, nthreads=args.ncpu,
                     mu_max=1.0, nmu_bins=args.n_mu_bins, binfile=binfile,
                     RA1=data[:, 0], DEC1=data[:, 1],
                     CZ1=data[:, 2], weights1=data[:, 3],
                     weight_type=None, is_comoving_dist=True,
                     output_savg=True, verbose=True)

    print('>> Computing RR pair count')
    RR = DDsmu_mocks(autocorr=1, cosmology=1, nthreads=args.ncpu,
                     mu_max=1.0, nmu_bins=args.n_mu_bins, binfile=binfile,
                     RA1=rand[:, 0], DEC1=rand[:, 1],
                     CZ1=rand[:, 2], weights1=rand[:, 3],
                     weight_type=None, is_comoving_dist=True,
                     output_savg=True, verbose=True)

    print('>> Computing DR pair count')
    DR = DDsmu_mocks(autocorr=0, cosmology=1, nthreads=args.ncpu,
                     mu_max=1.0, nmu_bins=args.n_mu_bins, binfile=binfile,
                     RA1=data[:, 0], DEC1=data[:, 1],
                     CZ1=data[:, 2], weights1=data[:, 3],
                     RA2=rand[:, 0], DEC2=rand[:, 1],
                     CZ2=rand[:, 2], weights2=rand[:, 3],
                     weight_type=None, is_comoving_dist=True,
                     output_savg=True, verbose=True)

    del data
    del rand

    print('>> Extracting data')

    def re_shape(arr):
        '''reshape to same s bin for each row'''
        return np.reshape(arr, (args.n_s_bins, args.n_mu_bins))
    # DD, DR, RR have the same smin, smax and mumax, so use any one
    # the data are given in the same s bin, different mu bins order
    s_min = re_shape(np.array([p['smin'] for p in DD]))[:, 0]
    s_max = re_shape(np.array([p['smax'] for p in DD]))[:, 0]
    mu_max = re_shape(np.array([p['mumax'] for p in DD]))[0, :]

    # average s in each bin, different for DD, DR and RR
    s_ave_DD = re_shape(np.array([p['savg'] for p in DD]))
    s_ave_DR = re_shape(np.array([p['savg'] for p in DR]))
    s_ave_RR = re_shape(np.array([p['savg'] for p in RR]))
    # pair counts
    n_DD = re_shape(np.array([p['npairs'] for p in DD]))
    n_DR = re_shape(np.array([p['npairs'] for p in DR]))
    n_RR = re_shape(np.array([p['npairs'] for p in RR]))
    # average weight
    w_ave_DD = re_shape(np.array([p['weightavg'] for p in DD]))
    w_ave_DR = re_shape(np.array([p['weightavg'] for p in DR]))
    w_ave_RR = re_shape(np.array([p['weightavg'] for p in RR]))

    print('>> Converting pair count to correlation function with Landy & Szalay method')
    xi2d = (n_DD - 2. * n_DR + n_RR) / n_RR

    s_mid = (s_min + s_max) / 2.
    mu_bw = (args.mu_max - 0.) / args.n_mu_bins
    mu_mid = np.array(
        [(i - 0.5) * mu_bw for i in range(1, args.n_mu_bins + 1)])

    print('>> Computing spherically averaged monopole')
    xi_0 = 0.5 * np.sum(xi2d * mu_mid, axis=1) / args.n_mu_bins
    print(xi_0)

    s_bins = np.concatenate((np.array([s_min[0]]), s_max))
    mu_bins = np.concatenate((np.array([0.]), mu_max))
    save_smu_arr('xi2d_test.dat', xi2d, s_bins, mu_bins)
    save_smu_arr('n_DD_test.dat', n_DD, s_bins, mu_bins)
    save_smu_arr('n_DR_test.dat', n_DR, s_bins, mu_bins)
    save_smu_arr('n_RR_test.dat', n_RR, s_bins, mu_bins)
