import pickle
import gzip
import numpy as np
from tqdm import tqdm

dirname = '/n/holystore01/LABS/iaifi_lab/Lab/creissel/neutrino_mass/'
variable=['axial_frequency_Hz', 'carrier_frequency_Hz']
length = 0.4e4
nrepeat = 3
normalize = True

from os import listdir
from os.path import isfile, join
files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
files = [dirname + f for f in files]
files = [f for f in files if '.pkl.gz' in f]

for i,fname in enumerate(tqdm(files)):
    with gzip.open(fname, "rb") as f:
        data = pickle.load(f, encoding='bytes')

    for j in range(nrepeat):
        # preparation of time series data
        X = data['time_series']
        X = np.stack(np.array(X))
        X = X[:,int(length)*j:int(length)*(j+1)]
        X = X[:, :, np.newaxis] # to fit the convention of the SSM

        # preparation of variables to be regressed
        y = data[variable]
        y = np.array(y)

        if (i ==0) and (j==0):
            Xfull = X
            yfull = y
        else:
            Xfull = np.append(Xfull, X, axis=0)
            yfull = np.append(yfull, y, axis=0)

if normalize:
    mu_y = np.mean(yfull, axis=0)
    stds_y = np.std(yfull, axis=0)
    yfull = (yfull-mu_y)/stds_y

np.savez('data.npz', X=Xfull, y=yfull, mu_y=mu_y, stds_y=stds_y)
