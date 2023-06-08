# Extending Kernel PCA through Dualization: Sparsity, Robustness and Fast Algorithms
A duality framework to solve the KPCA problem efficiently with extension to robust and sparse losses.

## Abstract
The goal of this paper is to revisit Kernel Principal Component Analysis (KPCA) through dualization of a difference of convex functions. This allows to naturally extend KPCA to multiple objective functions and leads to efficient gradient-based algorithms avoiding the expensive SVD of the Gram matrix. Particularly, we consider objective functions that can be written as Moreau envelopes, demonstrating how to promote robustness and sparsity within the same framework. The proposed method is evaluated on synthetic and realworld benchmarks, showing significant speedup in KPCA training time as well as highlighting the benefits in terms of robustness and sparsity.


## Installation
Clone the repository, navigate to the directory and install required python packages with the provided requirements.txt file.

```
pip install -r requirements.txt
```

## Code structure

### General functions

- Available kernels are in `kernels.py`.

### LBFGS
The proposed LBFGS-based algorithm for KPCA and comparison with other SVD solvers are in `test_lbfgs.py`.

### DCA
The DCA for Huber and epsilon-insensitive losses are in `dc.py`.
A demo is in `test_dc.py`.

## Cite
If you use this code, please cite the corresponding work:

```
@InProceedings{tonin2023,
  title = {Extending Kernel {PCA} through Dualization: Sparsity, Robustness and Fast Algorithms},
  author = {Tonin, Francesco and Lambert, Alex and Patrinos, Panagiotis and Suykens, Johan},
  booktitle = {The 40th International Conference on Machine Learning},
  year = {2023},
  organization= {PMLR}
}
```