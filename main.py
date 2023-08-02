## Implement SGS
## Author: Honggeun Jo
## Date: 2023.08.01

import numpy as np
import pandas as pd
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from geostatspy import geostats


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
        # variogram = geostats.make_variogram(**self.Vario)
        return {
                "nreal": self.num_real,
                "df_": self.data,
                "Val_range": val_range,
                "xcol": "X_i",
                "ycol": "Y_i",
                "zcol": "Z_i",
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
                "var": self.Vario,
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
        TI = np.load(TI_file)
        with open(f"angle_{angle}/train.dat", "w") as f:
            f.write("train image \n")
            f.write("1 \n")
            f.write("value \n")
            for i in TI.flatten():
                f.write(f"{int(i)} \n")
        os.chdir(f'angle_{angle}')
        os.system(f"snesim.exe < snesim.par")       
        np.save('mps_reals.npy', np.loadtxt(f"snesim.out",skiprows = 4)[:,0].reshape(self.num_real, self.grid_dim[-1], self.grid_dim[-2], self.grid_dim[-3]))

        os.remove(f"snesim.par")
        os.remove(f"snesim.out")
        os.remove(f"snesim.exe")
        os.remove(f"data_temp.dat")
        os.remove(f"train.dat")
        
        

if __name__ == '__main__':
    gen_real = Gen_TwoGeoStats()
    gen_real.load_well_data(pd.read_csv('sample_data_v1.csv'))
    gen_real.rasterize_grid()
    realization = gen_real.sgsim_realizations('Porosity',[0,0.3])
    np.save('realizations.npy',realization)