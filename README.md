Official repository for paper "Rethinking Resolution: Large-Scale Polygonal Building Detection Using Medium-Resolution (3-5m) Satellite Data".
<p align="center">
  <img src="resources/pipeline.png" alt="描述文本" width="1000">
</p>

## Installation
First, we need to create a conda environment using the following commands.
```
conda env create -f environment.yaml -n MRPolyBuild
```

Later on, simply install MRPolyBuilding by running `pip install -e .`

## Data Preparation
The satellite images used in this work 

## Train
Training MRPolyBuild includes the following two steps.

### Train Semantic Segmentation Networks

### Train Polygon Refinement Networks

## Inference
### Inference with PlanetScope Satellite Images

### Inference with Existing Probability Maps

## Evaluation
