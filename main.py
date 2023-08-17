## Implement SGS
## Author: Honggeun Jo
## Date: 2023.08.01

import numpy as np
import pandas as pd
import os
import shutil
from geostatspy import geostats
from bisect import bisect

from gslib_wrapper import sgsim

class Gen_TwoGeoStats():
    version = '0.0.1'

    Vario = None
    data = None 
    num_real = 1              # how many model to create
    seed = 77777
    grid_mn = [10, 10, 1000]  # location of [0, 0, 0] grid
    grid_size = [20, 20, 1]   # grid size in X-, Y-, and Z-directions, respectively
    grid_dim  = [128, 128, 1] # grid dimension

    def load_variogram(self, variogram):
        self.Vario = variogram
        
    def load_well_data(self, data):
        assert all([i in data.columns for i in ['X', 'Y', 'Z']]), \
               'X, Y, Z columns are required in data' 
        self.data = data

    def rasterize_grid(self):
        # rasterize spatial map into grid system
        self.data ['X_i'] = np.round((self.data ['X'] - self.grid_mn[0]) / self.grid_size[0]) + 1/2
        self.data ['Y_j'] = np.round((self.data ['Y'] - self.grid_mn[1]) / self.grid_size[1]) + 1/2
        self.data ['Z_k'] = np.round((self.data ['Z'] - self.grid_mn[2]) / self.grid_size[2]) + 1/2
        

    def create_sgs_model(self, val_name, val_range, 
                         horizon_grid_size = 1, 
                         vertical_grid_size = 1, 
                         grid_mn=[0.5, 0.5, 0.5]):
        variogram = geostats.make_variogram_3D(**self.Vario)
        return {
                "nreal": self.num_real,
                "df_": self.data,
                "Val_range": val_range,
                "xcol": "X_i",
                "ycol": "Y_j",
                "zcol": "Z_k",
                "vcol": val_name,
                "nx_cells": self.grid_dim[0],
                "ny_cells": self.grid_dim[1],
                "nz":  self.grid_dim[2],
                "hsiz": horizon_grid_size,
                "vsiz": vertical_grid_size,
                "hmn_max": grid_mn[0],
                "hmn_med": grid_mn[1],
                "zmn_ver": grid_mn[2],
                "seed": self.seed,
                "var": variogram,
                "output_file": "sgsim.out"
            }
            
    
    def sgsim_realizations(self, val_name, val_range):
        sim = sgsim(**self.create_sgs_model(val_name, val_range))
        return sim
        
    def __version__(self):
        print(f'The current version is {self.__version__}')

    def parallel_sgs(self, parameter_file, data_file,seeds):
        for seed in seeds:
            os.mkdir(f'seed_{seed}')
            for file in ['sgsim.exe', parameter_file, data_file]:
                shutil.copyfile(file,f'seed_{seed}/{file}')

        # make training image file
        os.chdir(f'seed_{seed}')
        os.system(f"snesim.exe < snesim.par")       
        np.save('mps_reals.npy', np.loadtxt(f"snesim.out",skiprows = 4)[:,0].reshape(self.num_real, self.grid_dim[-1], self.grid_dim[-2], self.grid_dim[-3]))

        os.remove(f"snesim.par")
        os.remove(f"snesim.out")
        os.remove(f"snesim.exe")
        os.remove(f"data_temp.dat")
        os.remove(f"train.dat")
        
    def nscore( self,
        df, vcol, wcol=None, ismooth=False, dfsmooth=None, smcol=0, smwcol=0
        ):
        """GSLIB's NSCORE program (Deutsch and Journel, 1998) converted from the
        original Fortran to Python by Michael Pyrcz, the University of Texas at
        Austin (Jan, 2019).
        :param df: pandas DataFrame with the spatial data
        :param vcol: name of the variable column
        :param wcol: name of the weight column, if None assumes equal weighting
        :param ismooth: if True then use a reference distribution
        :param dfsmooth: pandas DataFrame required if reference distribution is used
        :param smcol: reference distribution property (required if reference
                    distribution is used)
        :param smwcol: reference distribution weight (required if reference
                    distribution is used)
        :return: TODO
        """
        # Set constants
        np.random.seed(90102)
        pwr = 1.0  # interpolation power, hard coded to 1.0 in GSLIB
        EPSILON = 1.0e-20

        # Decide which file to use for establishing the transformation table
        if ismooth:
            nd = len(dfsmooth)
            vr = dfsmooth[smcol].values
            wt_ns = np.ones(nd)
            if smwcol != 0:
                wt_ns = dfsmooth[smwcol].values
        else:
            nd = len(df)
            vr = df[vcol].values
            wt_ns = np.ones(nd)
            if wcol is not None:
                wt_ns = df[wcol].values
        twt = np.sum(wt_ns)

        # Sort data by value
        istart = 0
        iend = nd
        vr, wt_ns = self.dsortem(istart, iend, vr, 2, wt_ns)

        # Compute the cumulative probabilities and write transformation table
        wtfac = 1.0 / twt
        oldcp = 0.0
        cp = 0.0
        for j in range(istart, iend):
            w = wtfac * wt_ns[j]
            cp = cp + w
            wt_ns[j] = (cp + oldcp) / 2.0
            vrrg = self.gauinv(wt_ns[j])
            vrg = float(vrrg)
            oldcp = cp

            # Now, reset the weight to the normal scores value
            wt_ns[j] = vrg

        # Normal scores transform
        nd_trans = len(df)
        ns = np.zeros(nd_trans)
        val = df[vcol].values
        for i in range(0, nd_trans):
            vrr = val[i] + np.random.rand() * EPSILON

            # Now, get the normal scores value for "vrr"
            j = self.dlocate(vr, 1, nd, vrr)
            j = min(max(1, j), (nd - 1))
            ns[i] = self.dpowint(vr[j], vr[j + 1], wt_ns[j], wt_ns[j + 1], vrr, pwr)

        return ns, vr, wt_ns

    def inverse_nscore(self, df, wt_ns, wt_vr, vcol=None):

        # Set constants
        np.random.seed(90102)
        pwr = 1.0  # interpolation power, hard coded to 1.0 in GSLIB
        EPSILON = 0

        assert type(df) in [pd.DataFrame, np.ndarray]
        if type (df) == pd.DataFrame:
            assert vcol != None, 'name of normal scored variable is missing'
            assert vcol in df.columns, 'vcol is not found in the columns list'
            ns = df[vcol].values

        else:
            original_dim = df.shape
            ns = df.flatten()

        # Inverse Normal scores transform
        nd = wt_ns.shape[0]
        ivr = np.zeros(ns.shape[0])
        for i in range(0, ns.shape[0]):
            ns_vrr = ns[i] + np.random.rand() * EPSILON

            # Now, get the normal scores value for "vrr"
            j = self.dlocate(wt_ns, 1, nd, ns_vrr)
            j = min(max(1, j), (nd - 1))
            ivr[i] = self.dpowint(wt_ns[j], wt_ns[j + 1], wt_vr[j], wt_vr[j + 1], ns_vrr, pwr)
        if type (df) == pd.DataFrame:
            return ivr
        if type (df) == np.ndarray:
            return ivr.reshape(original_dim)
        
    @staticmethod
    def dlocate(xx, iis, iie, x):
        """Return value `j` such that `x` is between `xx[j]` and `xx[j+1]`, where
        `xx` is an array of length `n`, and `x` is a given value. `xx` must be
        monotonic, either increasing or decreasing (updated with Python bisect)
        :param xx: array
        :param iis: start point
        :param iie: end point
        :param x: given value
        :return: TODO
        """
        n = len(xx)
        if iie <= iis:
            iis = 0
            iie = n - 1
        array = xx[iis: iie - 1]  # this is accounting for swith to 0,...,n-1 index
        j = bisect(array, x)
        return j
    
    @staticmethod
    def dsortem(ib, ie, a, iperm, b=0, c=0, d=0, e=0, f=0, g=0, h=0):
        """Sort array in ascending order.
        :param ib: start index
        :param ie: end index
        :param a: array
        :param iperm: 0 no other array is permuted.
                    1 array b is permuted according to array a.
                    2 arrays b, c are permuted.
                    3 arrays b, c, d are permuted.
                    4 arrays b, c, d, e are permuted.
                    5 arrays b, c, d, e, f are permuted.
                    6 arrays b, c, d, e, f, g are permuted.
                    7 arrays b, c, d, e, f, g, h are permuted.
                    >7 no other array is permuted.
        :param b: array to be permuted according to array a.
        :param c: array to be permuted according to array a.
        :param d: array to be permuted according to array a.
        :param e: array to be permuted according to array a.
        :param f: array to be permuted according to array a.
        :param g: array to be permuted according to array a.
        :param h: array to be permuted according to array a.
        :return: a: the array, a portion of which has been sorted.
                b, c, d, e, f, g, h: arrays permuted according to array a (see
                iperm)
        """
        a = a[ib:ie]
        inds = a.argsort()
        a = np.copy(a[inds])  # deepcopy forces pass to outside scope
        if iperm == 1:
            return a
        b_slice = b[ib:ie]
        b = b_slice[inds]
        if iperm == 2:
            return a, b
        c_slice = c[ib:ie]
        c = c_slice[inds]
        if iperm == 3:
            return a, b, c
        d_slice = d[ib:ie]
        d = d_slice[inds]
        if iperm == 4:
            return a, b, c, d
        e_slice = e[ib:ie]
        e = e_slice[inds]
        if iperm == 5:
            return a, b, c, d, e
        f_slice = f[ib:ie]
        f = f_slice[inds]
        if iperm == 6:
            return a, b, c, d, e, f
        g_slice = g[ib:ie]
        g = g_slice[inds]
        if iperm == 7:
            return a, b, c, d, e, f, g  # TODO: changed from 'a, b, c, d, e, f, h'
        h_slice = h[ib:ie]
        h = h_slice[inds]
        return a, b, c, d, e, f, g, h  # TODO: changed from 'a, b, c, d, e, f, h'

    
    @staticmethod    
    def dpowint(xlow, xhigh, ylow, yhigh, xval, pwr):
        """Power interpolate the value of `y` between (`xlow`, `ylow`) and
        (`xhigh`, `yhigh`) for a value of `x` and a power `pwr`.
        :param xlow: TODO
        :param xhigh: TODO
        :param ylow: TODO
        :param yhigh: TODO
        :param xval: TODO
        :param pwr: power
        :return: TODO
        """
        EPSLON = 1.0e-20
        if (xhigh - xlow) < EPSLON:
            dpowint_ = (yhigh + ylow) / 2.0
        else:
            dpowint_ = ylow + (yhigh - ylow) * (
                ((xval - xlow) / (xhigh - xlow)) ** pwr
            )
        return dpowint_

    @staticmethod
    def gauinv(p):
        """Compute the inverse of the standard normal cumulative distribution
        function.
        :param p: cumulative probability value
        :return: TODO
        """
        lim = 1.0e-10
        p0 = -0.322_232_431_088
        p1 = -1.0
        p2 = -0.342_242_088_547
        p3 = -0.020_423_121_024_5
        p4 = -0.000_045_364_221_014_8
        q0 = 0.099_348_462_606_0
        q1 = 0.588_581_570_495
        q2 = 0.531_103_462_366
        q3 = 0.103_537_752_850
        q4 = 0.003_856_070_063_4

        # Check for an error situation
        if p < lim:
            xp = -1.0e10
            return xp
        if p > (1.0 - lim):
            xp = 1.0e10
            return xp

        # Get k for an error situation
        pp = p
        if p > 0.5:
            pp = 1 - pp
        xp = 0.0
        if p == 0.5:
            return xp

        # Approximate the function
        y = np.sqrt(np.log(1.0 / (pp * pp)))
        xp = float(
            y
            + ((((y * p4 + p3) * y + p2) * y + p1) * y + p0)
            / ((((y * q4 + q3) * y + q2) * y + q1) * y + q0)
        )
        if float(p) == float(pp):
            xp = -xp
        return xp


if __name__ == '__main__':
    gen_real = Gen_TwoGeoStats()
    gen_real.load_well_data(pd.read_csv('sample_data_v1.csv'))
    gen_real.rasterize_grid()
    realization = gen_real.sgsim_realizations('Porosity',[0,0.3])
    np.save('realizations.npy',realization)