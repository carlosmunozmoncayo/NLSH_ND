# NLSH_ND
Reproducibility repository for the manuscript "A hyperbolic nonlinear Schrödinger system in multiple spatial dimensions: structure, convergence, and numerical discretization" by Carlos Muñoz-Moncayo, Abhijit Biswas, David I. Ketcheson, and Manvendra Pratap Rajvanshi

### Requirements
The code in this repository requires Python 3.10.14 or higher and the following packages:
- numpy
- matplotlib

Moreover, to run the finite volume solver, Clawpack is required. Please refer to the [Clawpack installation guide](https://www.clawpack.org/installing_pip.html#install-quick-all).
Once Clawpack is installed, please point the `CLAW` environment variable to the Clawpack installation directory and then run
```bash
source setenv.sh
```
You might want to run the simulations in parallel. You'll need OpenMP for that. Please refer to [This guide](https://www.clawpack.org/openmp.html) for more information.



### Reproducing the results in the paper
To reproduce the numerical results in the paper, please refer to the README files in the subdirectories of this repository. 
The code is organized into the following subdirectories:
- `modulational_instability/`: Section 6.1.1
- `well_prepared/`: Section 6.1.2
- `standing_waves/`: Section 6.1.3
- `pyclaw/`: Section 6.2.1
- `perfect_reflection_defocusing/`: Section 6.2.2
- `diagonal_reflection_defocusing/`: Section 6.2.2

