import sys
import argparse
import uproot
import pandas as pd
import gzip
import tqdm as tqdm
module_dir = '/home/lfd34/project/project8/ssm/neutrino_project/data_preprocessing'
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

# Add this directory to sys.path if it's not already included
if module_dir not in sys.path:
    sys.path.append(module_dir)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import find_peaks
import scipy.stats

import json
import glob
import os
import re

from EggFileReader import EggFileReader # Reads in the simulation output

# filename_template = "/gpfs/gibbs/pi/heeger/lfd34/recon/ssm_sim_files/v.3.3.0/163k_points/Trap_CCA_long_trap_44_TE011_onRes_EVev_FieldMap_5_7apLength_QL400_coarse_CombinedSet/results/*/*.egg"
# files = glob.glob(filename_template)

print("did something")

signal_len = int(163840*1.0)     # Number of time samples 

def process(files):
    info_list = []
    counter = 0
    pattern = "long_tracks/SpreadsheetRow44_SSM_20bins_*.pkl"

    existing = glob.glob(pattern)

    if existing:
        indices = []
        for f in existing:
            m = re.search(r"_([0-9]+)\.pkl$", f)
            if m:
                indices.append(int(m.group(1)))
        counter50s = max(indices) + 1
    else:
        counter50s = 0

    print(f"Starting counter50s at {counter50s}")

    for fileStr in files:
        try: 
            # get the metadata
            truthInfo = uproot.open(os.path.dirname(fileStr)+'/LocustEventProperties.root')
            truthJsonFile = open(os.path.dirname(fileStr)+'/LocustEventProperties.json')
            truthInfoJson = json.load(truthJsonFile)
            file = EggFileReader(fileStr)
            I_data, Q_data = file.quick_load_IQ_stream(stream=0)
        except:
            continue
        i_ch0 = I_data[0, 0, :signal_len]  # In-phase
        q_ch0 = Q_data[0, 0, :signal_len]  # Quadrature
        # get various truth information
        loFrequency = truthInfo[truthInfo.keys()[1]+'/LOFrequency'].array()[0]
        samplingFrequency = truthInfo[truthInfo.keys()[1]+'/SamplingFrequencyMHz'].array()[0]*1E6
        downsampledStartFrequency = float(truthInfoJson['0']['0']['output-track-start-frequency'])
        startFrequencyHz = downsampledStartFrequency + float(loFrequency) - float(samplingFrequency)/2

        end = len(fileStr)
        radius_t = float(fileStr[(end-32):(end-30+7)])#fileStr[0][105:114])
        pitch = truthInfo[truthInfo.keys()[0]+'/PitchAngles'].array()[0][0]
        radius = truthInfo[truthInfo.keys()[0]+'/Radii'].array()[0][0]
        energy = truthInfo[truthInfo.keys()[0]+'/StartingEnergieseV'].array()[0][0]
        outputAvgCarrierFrequency = truthInfo[truthInfo.keys()[0]+'/AvgFrequencies'].array()[0][0]
        outputAvgAxialFrequency = truthInfo[truthInfo.keys()[0]+'/AvgAxialFrequencies'].array()[0][0]
        outputStartTime = truthInfo[truthInfo.keys()[0]+'/StartTimes'].array()[0][0]
        outputEndTime = truthInfo[truthInfo.keys()[0]+'/EndTimes'].array()[0][0]
        outputSlope = truthInfo[truthInfo.keys()[0]+'/Slopes'].array()[0][0]
        outputRadiusPhase = truthInfo[truthInfo.keys()[0]+'/RadialPhases'].array()[0][0]
        if(pitch < 0): #untrapped
            continue
        info_list.append(
            {
                'energy_eV': energy,
                'pitch_angle_deg': 90-(pitch-90),
                'start_carrier_frequency_Hz': startFrequencyHz,
                'avg_carrier_frequency_Hz': outputAvgCarrierFrequency,
                'avg_axial_frequency_Hz': outputAvgAxialFrequency,
                'slope_Hz': outputSlope,
                'radius_m': radius,
                'radius_input_m': radius_t,
                'radius_phase': outputRadiusPhase,
                'output_ts_I': np.array(i_ch0),
                'output_ts_Q': np.array(q_ch0)
            }
        )
        counter += 1
        if(counter%50==0):
            print(counter50s)
            df = pd.DataFrame(info_list)
            outfile = f"./long_tracks/SpreadsheetRow44_SSM_20bins_{counter50s}.pkl"
            if os.path.exists(outfile):
                    raise RuntimeError(f"Refusing to overwrite {outfile}")
            df.to_pickle(outfile)
            # save to pickle every 50 files
            df.to_pickle("./long_tracks/SpreadsheetRow44_SSM_20bins_%d.pkl"%(counter50s))
            counter50s+=1
            info_list = []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename_template", type = str)
    args = parser.parse_args()
    files = glob.glob(args.filename_template)
    if len(files) == 0:
        raise RuntimeError("No files matched the given template.")
    print(f"Found {len(files)} files")
    process(files)

if __name__ == "__main__":
        main()
