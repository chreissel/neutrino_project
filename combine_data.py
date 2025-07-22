import h5py
import numpy as np

file_1_path = "/n/holystore01/LABS/iaifi_lab/Lab/creissel/neutrino_mass/combined_data_v2.hdf5"
file_2_path = "/n/holystore01/LABS/iaifi_lab/Lab/creissel/neutrino_mass/combined_data_newsim.hdf5" 
output_file_path = "/n/holystore01/LABS/iaifi_lab/Lab/creissel/neutrino_mass/combined_data_fullsim.hdf5" 


def merge_hdf5_files(file_1_path, file_2_path, output_file_path):

    with h5py.File(file_1_path, 'r') as f1:
        datasets = [d for d in f1.keys()]

    for d in datasets:

        with h5py.File(file_1_path, 'r') as f1, h5py.File(file_2_path, 'r') as f2:
            data1 = f1[d][:]  # Read as NumPy array
            data2 = f2[d][:]

        combined_data = np.concatenate((data1, data2), axis=0)

        with h5py.File(output_file_path, 'a') as f_combined:
            f_combined.create_dataset(d, data=combined_data)

    print(f"Successfully merged all files into '{output_file_path}'")

# Example usage:
# Assuming your HDF5 files have a dataset named 'my_data' and a common column 'ID'
merge_hdf5_files(file_1_path, file_2_path, output_file_path)
