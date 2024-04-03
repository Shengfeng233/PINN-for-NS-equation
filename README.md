# PINN-for-NS-equation
A  pytorch implementation of PINN-for-NS-eqation

This repository is actually a quite remote version of my code, but people have been constantly asking about questions under this repository, so I decide to update this old repository.

Here I present an interesting example of reconstructing flow field of 2D unsteady flow past a circular cylinder at Re=3900. As you can see, PINNs can reconstruct the wake flow with only 36 measurement points available.
The full dataset of this case cylinder_Re3900_ke_all_100snaps.mat can be found and downloaded at <https://github.com/Shengfeng233/PINN-MPI/tree/main/CFD_data> (I am running out of free usage of git_lfs, sorry~)


# Remark
The annotates are both in Chinese and English

You can plot comparsion pics and gifs in plot.py

# Reference:
Xu S, Sun Z, Huang R, et al. A practical approach to flow field reconstruction with sparse or incomplete data through physics informed neural network[J]. Acta Mechanica Sinica, 2023, 39(3): 322302.

Xu S, Yan C, Zhang G, et al. Spatiotemporal parallel physics-informed neural networks: A framework to solve inverse problems in fluid mechanics[J]. Physics of Fluids, 2023, 35(6).

Raissi M, Perdikaris P, Karniadakis G E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations[J]. Journal of Computational Physics, 2019, 378: 686-707.


