import pickle
import gzip
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize

dirname = '/n/holystore01/LABS/iaifi_lab/Lab/creissel/neutrino_mass/'
variable=['energy_eV', 'pitch_angle_deg', 'carrier_frequency_Hz', 'avg_axial_frequency_Hz', 'radius_m']
length = 0.4e4
nrepeat = 1
norm = True

from os import listdir
from os.path import isfile, join
files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
files = [dirname + f for f in files]
files = [f for f in files if '.pkl.gz' in f]

for i,fname in enumerate(tqdm(files)):
    with gzip.open(fname, "rb") as f:
        data = pickle.load(f, encoding='bytes')

    for j in range(nrepeat):
        # preparation of two time series
        x1 = data['output_ts_I']
        x1 = np.stack(np.array(x1))
        x2 = data['output_ts_Q']
        x2 = np.stack(np.array(x2))

        X = np.stack([x1,x2], axis=-1)
        X = X[:,int(length)*j:int(length)*(j+1),:]

        # preparation of variables to be regressed
        y = data[variable]
        y = np.array(y)

        if (i ==0) and (j==0):
            Xfull = X
            yfull = y
        else:
            Xfull = np.append(Xfull, X, axis=0)
            yfull = np.append(yfull, y, axis=0)

if norm:
    mu_y = np.mean(yfull, axis=0)
    stds_y = np.std(yfull, axis=0)
    yfull = (yfull-mu_y)/stds_y

np.savez('data.npz', X=Xfull, y=yfull, mu_y=mu_y, stds_y=stds_y)
