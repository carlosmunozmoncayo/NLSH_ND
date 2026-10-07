### FV scheme validation for the NLSH in 2D
The Python interface provided by Clawpack is used for simplicity (note that up to small differnces in handling the parameters, the source is identical to the one used by AMRClaw).

To run the convergence tests for the FV solver for the NLSH in 2D, run the following command from this directory:

```bash
python convergence_FV_solver_test.py #Figure 6
python convergence_sharpclaw_solver_test.py #Figure 12 (Appendix B)
```

To preview an example simulation of the FV solver for the NLSH in 2D, run the following command from this directory:

```bash
python plane_wave_rectangle_nlsh.py iplot=1
```
