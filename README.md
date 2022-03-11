# PINN-for-NS-equation
A further implementation of PINN-for-NS-eqation

# Major changes:
1.cosine annealing lr strategy
2.LHS sampling method for equtions points
3.forward problem for sparse data(0.5% of origin data)

The annotates are in Chinese

This implementation uses two dimensional cylinder pass flow data from Raissi(see reference)

You can plot comparsion pics and gifs in plot.py

# Reference:
Raissi M, Perdikaris P, Karniadakis G E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations[J]. Journal of Computational Physics, 2019, 378: 686-707.

Raissi M, Perdikaris P, Karniadakis G E. Physics informed deep learning (part i): Data-driven solutions of nonlinear partial differential equations[J]. arXiv preprint arXiv:1711.10561, 2017.

Raissi M, Perdikaris P, Karniadakis G E. Physics informed deep learning (part ii): Data-driven discovery of nonlinear partial differential equations. arXiv 2017[J]. arXiv preprint arXiv:1711.10566.
