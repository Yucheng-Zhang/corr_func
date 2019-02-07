'''
Cosmology class.
'''
from astropy.cosmology import LambdaCDM
import numpy as np
from scipy import interpolate


class cosmo_LCDM:
    '''LambdaCDM cosmology.'''

    def __init__(self, H0, Om0, Ode0):
        self.H0 = H0
        self.cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)
        self.h = self.H0 / 100.

    def z2chi(self, z):
        '''Get comoving distance in [Mpc/h] from redshift'''
        return self.cosmo.comoving_distance(z).value * self.h

    def rdz2xyz(self, ra, dec, redshift):
        '''Convert RA, DEC, redshift to X, Y, Z'''
        co_dis = self.z2chi(redshift)
        theta = np.pi * (90. - dec) / 180.
        phi = np.pi * ra / 180.

        ngal = ra.size
        xyz = np.zeros(ngal * 3).reshape(ngal, 3)
        xyz[:, 0] = co_dis * np.sin(theta) * np.cos(phi)
        xyz[:, 1] = co_dis * np.sin(theta) * np.sin(phi)
        xyz[:, 2] = co_dis * np.cos(theta)

        return xyz

    def H_z(self, z):
        '''Get Hubble parameter at redshift z, i.e. H(z)'''
        return self.H0 * self.cosmo.efunc(z)

    def inv_H_z(self, z):
        '''Inverse of H(z), i.e. 1/H(z)'''
        return self.cosmo.inv_efunc(z) / self.H0

    def chi2z(self, chi, z_min, z_max, n_interp=1000):
        '''Get redshift from comoving distance with interpolation,
        make sure that all chi are included in the redshift bin set by z_min, z_max'''
        z_samp = np.linspace(z_min, z_max, n_interp)
        chi_samp = self.z2chi(z_samp)
        interp = interpolate.splrep(chi_samp, z_samp, s=0, k=1)

        return interpolate.splev(chi, interp, der=0)


if __name__ == '__main__':
    '''For test.'''
    cosmo = cosmo_LCDM(H0=70.0, Om0=0.3, Ode0=0.7)
    z = [0.0, 0.5, 0.6]
    inv_H = cosmo.inv_H_z(z)
    print(inv_H)
