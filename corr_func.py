'''
Compute 2D correlation function.
'''
import numpy as np
import pandas as pd
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
from cosmo import cosmo_LCDM
import argparse
import sys

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

    parser.add_argument('-out_xi_smu', default='out_xi_smu.dat',
                        help='Ouput xi(s,mu) file')
    parser.add_argument('-out_xi_s', default='out_xi_s.dat',
                        help='Output xi(s) file.')
    parser.add_argument('-out_xi_0', default='out_xi_0.dat',
                        help='Output monopole file')
    parser.add_argument('-out_xi_2', default='out_xi_2.dat',
                        help='Output quadrupole file')

    parser.add_argument('-save_pc', type=int, default=1,
                        help='Save pair count to file.')
    parser.add_argument('-verbose', type=int, default=0,
                        help='Verbose for Corrfunc, 0 or 1')

    args = parser.parse_args()


def save_pc(fn, pc):
    '''Save the result of DDsmu_mocks.'''
    header = 'smin   smax   savg   mumax   npairs   weightavg'
    data = np.array([[p['smin'], p['smax'], p['savg'],
                      p['mumax'], p['npairs'], p['weightavg']]for p in pc])
    np.savetxt(fn, data, header=header)


def save_smu_arr(fn, smu_arr, s_bins, mu_bins, header=''):
    '''Save s mu 2D array.'''
    header += 'Same s for each row\n'
    header += '{0:d} s bins, with edges:\n'.format(
        len(s_bins) - 1) + np.array_str(s_bins) + '\n'
    header += '{0:d} mu bins, with edges:\n'.format(
        len(mu_bins) - 1) + np.array_str(mu_bins)
    np.savetxt(fn, smu_arr, header=header)


def save_s_arr(fn, s_arr, s_bins, s_eff, h='value', s_err=None):
    '''Save s 1D array.'''
    header = '{0:d} s bins, with edges:\n'.format(
        len(s_bins) - 1) + np.array_str(s_bins) + '\n'
    header += '   s   {}'.format(h)
    data = np.column_stack((s_eff, s_arr))
    # error bar provided
    if s_err != None:
        header += '   error-bar'
        data = np.column_stack((data, s_err))

    np.savetxt(fn, data, header=header)


def load_data_w_pd(fn):
    '''Read input data with pandas.'''
    tb = pd.read_table(fn, delim_whitespace=True, comment='#', header=None)
    return tb.values


def pc_radecz_smu(data, rand, args, binfile):
    '''>> Pair count in (s, mu) with RA, DEC, comoving radial distance input.'''

    print('>> Computing DD pair count')
    DD = DDsmu_mocks(autocorr=1, cosmology=1, nthreads=args.ncpu,
                     mu_max=1.0, nmu_bins=args.n_mu_bins, binfile=binfile,
                     RA1=data[:, 0], DEC1=data[:, 1],
                     CZ1=data[:, 2], weights1=data[:, 3],
                     weight_type='pair_product', is_comoving_dist=True,
                     output_savg=True, verbose=args.verbose)

    print('>> Computing RR pair count')
    RR = DDsmu_mocks(autocorr=1, cosmology=1, nthreads=args.ncpu,
                     mu_max=1.0, nmu_bins=args.n_mu_bins, binfile=binfile,
                     RA1=rand[:, 0], DEC1=rand[:, 1],
                     CZ1=rand[:, 2], weights1=rand[:, 3],
                     weight_type='pair_product', is_comoving_dist=True,
                     output_savg=True, verbose=args.verbose)

    print('>> Computing DR pair count')
    DR = DDsmu_mocks(autocorr=0, cosmology=1, nthreads=args.ncpu,
                     mu_max=1.0, nmu_bins=args.n_mu_bins, binfile=binfile,
                     RA1=data[:, 0], DEC1=data[:, 1],
                     CZ1=data[:, 2], weights1=data[:, 3],
                     RA2=rand[:, 0], DEC2=rand[:, 1],
                     CZ2=rand[:, 2], weights2=rand[:, 3],
                     weight_type='pair_product', is_comoving_dist=True,
                     output_savg=True, verbose=args.verbose)

    return DD, DR, RR


def extract_pc(DD, args):
    '''Extract results from pair count.'''

    def re_shape(arr):
        '''reshape to same s bin for each row'''
        return np.reshape(arr, (args.n_s_bins, args.n_mu_bins))

    # the data are given in the same s bin, different mu bins order
    s_min = re_shape(np.array([p['smin'] for p in DD]))[:, 0]
    s_max = re_shape(np.array([p['smax'] for p in DD]))[:, 0]
    mu_max = re_shape(np.array([p['mumax'] for p in DD]))[0, :]
    # bin edges
    s_bins = np.concatenate((np.array([s_min[0]]), s_max))
    mu_bins = np.concatenate((np.array([0.]), mu_max))
    # average s in each bin
    s_ave = re_shape(np.array([p['savg'] for p in DD]))
    # pair counts
    n_pc = re_shape(np.array([p['npairs'] for p in DD]))
    # average weight
    w_ave = re_shape(np.array([p['weightavg'] for p in DD]))
    # weighted pair counts
    n_w = n_pc * w_ave
    # effective bin value
    s_eff = np.average(s_ave, axis=1)
    mu_bw = (args.mu_max - 0.) / args.n_mu_bins
    mu_mid = np.array(
        [(i - 0.5) * mu_bw for i in range(1, args.n_mu_bins + 1)])

    # combine in dictionary
    DD = {'s_bins': s_bins, 's_eff': s_eff, 'mu_bins': mu_bins,
          'mu_eff': mu_mid, 'n_pc': n_pc, 'w_ave': w_ave, 'n_w': n_w}

    return DD


def calc_xi2d(DD, DR, RR, w_sum_d, w_sum_r, method='LS'):
    '''Estimator for correlation function with pair count.'''
    if method == 'LS':
        # weight ratio
        f = w_sum_d / w_sum_r
        print('>> Converting pair count to correlation function with Landy & Szalay method')
        xi2d = (DD - 2. * f * DR + f**2 * RR) / (f**2 * RR)
    else:
        sys.exit('Wrong method.')

    return xi2d


def calc_xi_s(xi2d, method='spher_ave'):
    '''Get xi(s) from xi(s,mu).'''
    if method == 'spher_ave':
        print('>> Computing spherically averaged xi(s)')
        xi_s = np.average(xi2d, axis=1)
    else:
        sys.exit('Wrong method.')

    return xi_s


def calc_xi_pole(xi2d, mu, ell, method='spher_ave'):
    '''Get multipole from xi(s,mu).'''
    if method == 'spher_ave':
        print('>> Computing spherically averaged monopole')
        if ell == 0:
            L_mu = np.ones(len(mu))
        elif ell == 2:
            L_mu = (3. * np.power(mu, 2) - 1.) / 2.
        else:
            sys.exit('Only monopole and quadrupole are supported.')
        factor = 2. * (2. * ell + 1.) / 2.
        tmp_arr = xi2d * L_mu
        # average of all mu bins
        xi_ell = factor * np.average(tmp_arr, axis=1)
        # standard deviation of all mu bins
        xi_std = factor * np.std(tmp_arr, axis=1)
    else:
        sys.exit('Wrong method.')

    return xi_ell, xi_std


if __name__ == '__main__':

    print('>> Loading data file: {}'.format(args.data))
    data = load_data_w_pd(args.data)
    n_data = len(data[:, 0])
    print('>> {0:d} data points'.format(n_data))
    w_sum_d = np.sum(data[:, 3])
    print('>> Sum of data weights: {0:f}'.format(w_sum_d))

    print('>> Loading random file: {}'.format(args.rand))
    rand = load_data_w_pd(args.rand)
    n_rand = len(rand[:, 0])
    print('>> {0:d} random points'.format(n_rand))
    w_sum_r = np.sum(rand[:, 3])
    print('>> Sum of random weights: {0:f}'.format(w_sum_r))

    print('>> Getting comoving distance from redshift')
    print('>> fiducial cosmology: H0 = {0:f}, OmegaM_0 = {1:f}'.format(
        args.H0, args.Om0))
    cosmo = cosmo_LCDM(args.H0, args.Om0, 1. - args.Om0)
    data[:, 2] = cosmo.z2chi(data[:, 2])
    rand[:, 2] = cosmo.z2chi(rand[:, 2])

    print('>> Getting bin file for s')
    binfile = np.linspace(args.s_min, args.s_max, args.n_s_bins + 1)

    DD, DR, RR = pc_radecz_smu(data, rand, args, binfile)

    del data
    del rand

    print('>> Extracting data')
    DD = extract_pc(DD, args)
    DR = extract_pc(DR, args)
    RR = extract_pc(RR, args)

    if args.save_pc:
        save_smu_arr('DD_pc.dat', DD['n_w'], DD['s_bins'], DD['mu_bins'])
        save_smu_arr('DR_pc.dat', DR['n_w'], DR['s_bins'], DR['mu_bins'])
        save_smu_arr('RR_pc.dat', RR['n_w'], RR['s_bins'], RR['mu_bins'])

    xi2d = calc_xi2d(DD['n_w'], DR['n_w'], RR['n_w'], w_sum_d, w_sum_r)
    save_smu_arr(args.out_xi_smu, xi2d, DD['s_bins'], DD['mu_bins'])

    xi_s = calc_xi_s(xi2d)
    save_s_arr(args.out_xi_s, xi_s, DD['s_bins'], DD['s_eff'], h='xi(s)')

    xi_0, xi_0_err = calc_xi_pole(xi2d, DD['mu_eff'], 0)
    save_s_arr(args.out_xi_0, xi_0, DD['s_bins'],
               DD['s_eff'], h='xi_0', s_err=xi_0_err)

    xi_2, xi_2_err = calc_xi_pole(xi2d, DD['mu_eff'], 2)
    save_s_arr(args.out_xi_2, xi_2, DD['s_bins'],
               DD['s_eff'], h='xi_2', s_err=xi_2_err)
