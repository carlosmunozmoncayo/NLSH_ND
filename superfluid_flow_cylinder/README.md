### Superfluid flow past a cylinder for the defocusing NLSH in 2D
To reproduce Figures 10 and 11 in the paper, run the commands below from this directory.
##### Beware that `bash run_all.sh` might take a couple of hours to run, even in parallel. By default it saves the solution every 0.25 time units, which will take around 200GB of disk space. If you only care about the plots, you only need three frames, so you can change the output frequency or prescribe the output times in `setrun.py` by modifying the `clawdata.output_style` block.

```bash
bash run_all.sh 
make mapping
python plots_paper.py #Figure 11
#If you also want to reproduce the sample physical grid and initial condition (Figure 10)
make mapping
tar -xzvf single_grid_data.tar.gz
python plot_geometry_ICs.py #Figure 10
```