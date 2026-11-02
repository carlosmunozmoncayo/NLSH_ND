#!/bin/bash

##########################################################
#Run Dirichlet simulation
sed -i.bak "/#flag_inclusion/s/,[^,]*,/,'dirichlet',/" setrun.py && rm setrun.py.bak
make data
make .output OUTDIR=_output_dirichlet

##########################################################
#Run Neumann simulation
sed -i.bak "/#flag_inclusion/s/,[^,]*,/,'neumann',/" setrun.py && rm setrun.py.bak
make data
make .output OUTDIR=_output_neumann

##########################################################
#Run potential simulation
sed -i.bak "/#flag_inclusion/s/,[^,]*,/,'potential',/" setrun.py && rm setrun.py.bak
make data
make .output OUTDIR=_output_potential

###########################################################
#Make mapping for plots
make mapping