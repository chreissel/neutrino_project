# Project 8 reconstruction using time series data

This repo contains all the code to run the energy reconstruction for Project 8. It is based on ' PyTorch Lightning ' to facilitate training, testing, and evaluation. An introduction to `PyTorch Lightning` can be found [here](https://lightning.ai/docs/pytorch/stable/starter/introduction.html). This README will guide you through the steps to prepare the inputs, configure and run a training, and access the results.

### Data preparation
The simulation is assumed to be available on the same cluster in `.pkl.gz` format. By running
```
python data_prep.py
```
we prepare a `.hdf5` file containing all information. This `.hdf5` is needed as input to the training procedure. The input and output paths can be adopted in [data_prep.py](data_prep.py).

### Training
`PyTorch Lightning` trainings are configured via a config file. An example config file is available in [config/config.yaml](config/config.yaml) and can be easily adopted for tests. It particularly needs to specify the input and output dimensionality (aka the number of time series to fit and the parameters of interest). Also, the path to the `.hdf5` file can be configured. The training will automatically start by running
```
python cli.py fit -c configs/config.yaml
```
Submission of training jobs is implemented in the script [submit.sh](submit.sh). All training characteristics are tracked via [weights and biases](https://wandb.ai/), so we recommend signing up for optimal user experience.

### Evaluation
Basic model loading and evaluation alongside diagnostics plots is implemented in the `jupyter` notebook [eval.ipynb](eval.ipynb).

## Where do we stand?
The code should be ready and easy to run many (parallel) trainings. We plan to further add to the code, e.g., adding more diagnostics through the training, different model architectures for benchmarking and further functionality, so keep updated! 
