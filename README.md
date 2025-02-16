# vMQP: Bayesian Circular Regression for Wind Direction Prediction

## Overview

This repository contains the implementation of the wind direction experiment from the article "Bayesian Circular Regression with von Mises Quasi-Processes" (vMQP). The code demonstrates the use of vMQP for predicting circular values, specifically wind directions, using a Bayesian approach with transductive learning and efficient Markov Chain Monte Carlo (MCMC) inference.

## Features

- **Bayesian Circular Regression**: Models circular target variables using a von Mises Quasi-Process (vMQP).
- **Efficient Gibbs Sampling**: Implements an augmentation scheme inspired by the Stratonovich transformation for fast MCMC inference.
- **Wind Direction Prediction**: Uses publicly available meteorological data to predict wind directions at unobserved locations.
- **Comprehensive Visualization**: Includes plotting scripts for analyzing results.

## Installation

Ensure you have Python 3.x installed along with the following dependencies:

```sh
pip install -r requirements.txt 
```


## Usage

### Running the Wind Direction Experiment

1. Clone this repository:
   ```sh
   git clone https://github.com/Yarden231/vMQP.git
   cd vMQP
   ```
2. Load the dataset (`data_calm.csv`, containing wind direction measurements from the German Weather Service, DWD).
3. Open and run the Jupyter Notebook:
   ```sh
   jupyter notebook Wind_experiment.ipynb
   ```

## File Structure

- `Wind_experiment.ipynb` - Jupyter Notebook for running the experiment.
- `data_calm.csv` - Wind direction dataset.
- `mcmc_params.py` - Configuration for MCMC sampling.
- `sampler.py` - Implementation of the sampling algorithm.
- `utils_vMQP.py` - Utility functions for preprocessing and calculations.
- `wind_plots.py` - Plotting functions for visualization.
- `requirements.txt` - List of required Python packages.

## Data

The dataset consists of wind direction measurements collected from 260 weather stations in Germany at 10-minute intervals. The experiment focuses on predicting wind directions at test locations based on training data from a subset of stations.

## Results

- The vMQP model outperforms Wrapped and Projected Gaussian Processes in terms of the Circular Continuous Ranked Probability Score (CRPS).
- The method successfully captures uncertainty and multimodality in wind direction predictions.


## Citation

If you use this code, please cite our article:

```
@article{Yarden Cohen, Alexandre Khae Wu Navarro, Jes Frellsen, Richard E. Turner, Raziel Riemer, Ari Pakman,
  title={Bayesian Circular Regression with von Mises Quasi-Processes},
  author={Anonymous},
  journal={AISTATS 2025},
  year={2025}
}
```




