# Project 8 reconstruction using time series data

This repo contains all the code to run the electron energy reconstruction for Project 8. It is based on ' PyTorch Lightning ' to facilitate training, testing, and evaluation. An introduction to `PyTorch Lightning` can be found [here](https://lightning.ai/docs/pytorch/stable/starter/introduction.html). This README will guide you through the steps to prepare the inputs, configure and run a training, and access the results.

Physics background: **Project 8** uses Cyclotron Radiation Emission Spectroscopy to measure the tritium beta decay spectrum with unprecedented precision. When electrons from beta decay spiral in a magnetic field, they emit cyclotron radiation at frequencies from which we can derive their energy. The neutrino mass affects the endpoint of the beta decay spectrum, providing a path to measuring this fundamental parameter.

### Repository structure

```text
neutrino_project/
├── configs/                                
├── notebooks/            
├── scripts/
    ├── combine_data.py
    ├── data_prep.py
├── src/
    ├── __init__.py                
    ├── data/             
      ├── __init__.py
      ├── data.py
      ├── dataset.py
    ├── models/            
      ├── __init__.py
      ├── curriculum_scheduler.py
      ├── losses.py
      ├── model.py
      ├── networks.py
      ├── s4d.py
    ├── utils/            
      ├── __init__.py
      ├── noise.py
      ├── plotting.py
      ├── transforms.py
├── cli.py                
├── env.yml               
├── submit.sh             
└── README.md           
```

### Setup and Installation
1. Clone the repository
```
git clone https://github.com/chreissel/neutrino_project.git
cd neutrino_project
```
2. Create and activate the Conda environment
```
conda env create -f env.yml
conda activate ssm
```

### Data preparation
Simulation files are expected to be locally stored in `.pkl.gz` format. By running
```
python data_prep.py
```
we prepare a `.hdf5` file containing all information. This `.hdf5` is the input to the training procedure. The input and output paths can be adopted in [data_prep.py](data_prep.py).

### Training
`PyTorch Lightning` trainings are configured via a config file. An example config files can be found in [config/](config/) and can be easily adopted for tests. It particularly needs to specify the input and output dimensionality (aka the number of time series to fit and the parameters of interest). Also, the path to the `.hdf5` file can be configured. The training will automatically start by running
```
python cli.py fit -c configs/config.yaml
```
Submission of training jobs is implemented in the script [submit.sh](submit.sh). All training characteristics are tracked via [weights and biases](https://wandb.ai/), so we recommend signing up for optimal user experience.

### Evaluation
Basic model loading and evaluation alongside diagnostics plots is implemented in the `jupyter` notebook [notebooks/eval.ipynb](notebooks/eval.ipynb).

## Where do we stand?
The code should be ready and easy to run many (parallel) trainings. We plan to further add to the code, e.g., adding more diagnostics through the training, different model architectures for benchmarking and further functionality, so keep updated! 

Work in progress particulalry focuses on the following items:

- [X] Implementing per-electron uncertainty estimation
- [ ] Applying Q-transformations to the input time-series data
- [ ] Adding and benchmarking new model architectures (e.g. LSTM, Conformer)
- [ ] Using the full track length as model input
- [ ] Performing a full comparison with non-ML baseline methods

