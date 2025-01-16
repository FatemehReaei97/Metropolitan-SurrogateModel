# Metropolitan-SurrogateModel
A machine learning framework for efficient metropolitan-scale urban flood prediction combining network clustering with surrogate models.


## Overview

This repository contains the implementation of the methodology described in our paper:

> "A Surrogate Machine Learning Modeling Approach for Enhancing the Efficiency of Urban Flood Modeling at Metropolitan Scales"

The framework provides a novel approach to metropolitan-scale flood prediction by combining:
- Network clustering based on manhole connectivity
- Synthetic rainfall event generation
- Random Forest surrogate models
- Parallel processing for large-scale applications

## Key Features

- **Network Clustering**: Partitions complex urban drainage networks into manageable, hydrologically connected clusters
- **Synthetic Event Generation**: Creates diverse rainfall scenarios for model training
- **Machine Learning**: Implements Random Forest models to predict flood characteristics:
  - Flood duration
  - Flood peak
  - Flood volume
- **HPC Integration**: Supports parallel processing on High-Performance Computing systems

## Repository Structure
Metropolitan-SurrogateModel/
├── src/
│   ├── preprocessing/
│   │   └── synthetic_rainfall_generator.py
│   ├── clustering/
│   │   └── network_analyzer.py
│   ├── modeling/
│   │   └── rf_cluster_trainer.py
│   └── simulation/
│       └── swmm_parallel_runner.py
├── scripts/
│   └── hpc/
│       ├── train_clusters.job
│       └── run_swmm_simulations.job
