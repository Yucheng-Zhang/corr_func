import numpy as np
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
import multiprocessing
from .utils import load_data_pd
import sys


class corr:

    def __init__(self):

        self.h = None
        self.cosmo = None

        self.mu_max = None
        self.n_mu_bins = None
        self.n_s_bins = None
        self.binfile = None

        self.s = None
        self.mu = None

        self.data = None
        self.data_w_sum = None
        self.rand = None
        self.rand_w_sum = None

        self.ncpu = multiprocessing.cpu_count() - 2

        self.DD, self.DR, self.RR = None, None, None
        self.xi2d = None
        self.xi0, self.xi2 = None, None

    def set_cosmo(self, H0=67.66, Om0=0.3111):
        '''flat LCDM'''
        self.h = H0 / 100.
        self.cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

    def set_bins(self, s_max=200, n_s_bins=40, mu_max=1., n_mu_bins=100):
        '''set s and mu bins'''
        self.mu_max = mu_max
        self.n_mu_bins = n_mu_bins
        self.n_s_bins = n_s_bins
        bs = np.linspace(0, s_max, n_s_bins+1)
        bs[0] = 0.1
        self.binfile = bs

        self.s = (bs[1:] + bs[:-1]) / 2  # median s values

        mu_bw = (mu_max - 0.) / n_mu_bins
        self.mu = np.array(
            [(i - 0.5) * mu_bw for i in range(1, n_mu_bins + 1)])

    def read_data(self, fn, ftype='fits'):
        '''load data file'''
        if ftype == 'fits':
            data = fits.open(fn)[1].data
            ra, dec = data['RA'], data['DEC']
            weight = data['WEIGHT_FKP']
            z_dist = self.cosmo.comoving_distance(data['Z']).value * self.h
        elif ftype == 'txt':
            data = load_data_pd(fn)
            ra, dec = data[:, 0], data[:, 1]
            weight = data[:, 3]
            z_dist = self.cosmo.comoving_distance(data[:, 2]).value * self.h

        self.data = np.column_stack((ra, dec, z_dist, weight))
        self.data_w_sum = np.sum(weight)

    def read_rand(self, fn, ftype='fits'):
        '''load random file'''
        if ftype == 'fits':
            data = fits.open(fn)[1].data
            ra, dec = data['RA'], data['DEC']
            weight = data['WEIGHT_FKP']
            z_dist = self.cosmo.comoving_distance(data['Z']).value * self.h
        elif ftype == 'txt':
            data = load_data_pd(fn)
            ra, dec = data[:, 0], data[:, 1]
            weight = data[:, 3]
            z_dist = self.cosmo.comoving_distance(data[:, 2]).value * self.h

        self.rand = np.column_stack((ra, dec, z_dist, weight))
        self.rand_w_sum = np.sum(weight)

    def read_rdzw(self, rdzw, dat_or_ran):
        '''load ra, dec, redshift, weight'''
        rdzw[:, 2] = self.cosmo.comoving_distance(rdzw[:, 2].value * self.h)
        if dat_or_ran == 'data':
            self.data = rdzw
            self.data_w_sum = np.sum(rdzw[:, 3])
        elif dat_or_ran == 'rand':
            self.rand = rdzw
            self.rand_w_sum = np.sum(rdzw[:, 3])
        else:
            sys.exit('data or rand')

    def extract_pc(self, PC):
        '''extract results from Corrfunc pair count'''

        def re_shape(arr):
            '''reshape to same s bin for each row'''
            return np.reshape(arr, (self.n_s_bins, self.n_mu_bins))

        # the data are given in the same s bin, different mu bins order
        # s_min = re_shape(np.array([p['smin'] for p in PC]))[:, 0]
        # s_max = re_shape(np.array([p['smax'] for p in PC]))[:, 0]
        # mu_max = re_shape(np.array([p['mumax'] for p in PC]))[0, :]
        # bin edges
        # s_bins = np.concatenate((np.array([s_min[0]]), s_max))
        # mu_bins = np.concatenate((np.array([0.]), mu_max))
        # average s in each bin
        # s_ave = re_shape(np.array([p['savg'] for p in PC]))
        # pair counts
        n_pc = re_shape(np.array([p['npairs'] for p in PC]))
        # average weight
        w_ave = re_shape(np.array([p['weightavg'] for p in PC]))
        # weighted pair counts
        n_w = n_pc * w_ave
        # effective bin value
        # s_eff = np.average(s_ave, axis=1, weights=n_w)

        # mu_bw = (self.mu_max - 0.) / self.n_mu_bins
        # mu_mid = np.array(
        # [(i - 0.5) * mu_bw for i in range(1, self.n_mu_bins + 1)])

        # combine in dictionary
        # PC = {'s_bins': s_bins, 's_eff': s_eff, 'mu_bins': mu_bins,
        #   'mu_eff': mu_mid, 'n_pc': n_pc, 'w_ave': w_ave, 'n_w': n_w}

        PC = n_w

        return PC

    def pc_auto(self, data):
        '''auto pair count'''
        DD = DDsmu_mocks(autocorr=True, cosmology=2, nthreads=self.ncpu,
                         mu_max=1., nmu_bins=self.n_mu_bins, binfile=self.binfile,
                         RA1=data[:, 0], DEC1=data[:, 1],
                         CZ1=data[:, 2], weights1=data[:, 3],
                         weight_type='pair_product', is_comoving_dist=True,
                         output_savg=True, verbose=False)

        return self.extract_pc(DD)

    def pc_cross(self, data, rand):
        '''cross pair count'''
        DR = DDsmu_mocks(autocorr=False, cosmology=2, nthreads=self.ncpu,
                         mu_max=1., nmu_bins=self.n_mu_bins, binfile=self.binfile,
                         RA1=data[:, 0], DEC1=data[:, 1],
                         CZ1=data[:, 2], weights1=data[:, 3],
                         RA2=rand[:, 0], DEC2=rand[:, 1],
                         CZ2=rand[:, 2], weights2=rand[:, 3],
                         weight_type='pair_product', is_comoving_dist=True,
                         output_savg=True, verbose=False)

        return self.extract_pc(DR)

    def pc_DD(self):
        '''DD pair count'''
        self.DD = self.pc_auto(self.data)

    def pc_DR(self):
        '''DR pair count'''
        self.DR = self.pc_cross(self.data, self.rand)

    def pc_RR(self):
        '''RR pair count'''
        self.RR = self.pc_auto(self.rand)

    def get_xi2d(self):
        '''LS estimator for correlation function with pair count'''
        f = self.data_w_sum / self.rand_w_sum
        self.xi2d = (self.DD - 2. * f * self.DR +
                     f**2 * self.RR) / (f**2 * self.RR)

    def get_xi02(self):
        '''get xi_0 and xi_2, w/ spherical average'''
        # xi_0
        L_mu = np.ones(self.n_mu_bins)
        factor = 1.
        self.xi0 = factor * np.average(self.xi2d*L_mu, axis=1)
        # xi_2
        L_mu = (3. * np.power(self.mu, 2) - 1.) / 2.
        factor = 5.
        self.xi2 = factor * np.average(self.xi2d*L_mu, axis=1)
