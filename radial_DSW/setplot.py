
"""
Set up the plot figures, axes, and items to be done for each frame.

This module is imported by the plotting routines and then the
function setplot is called to set the plot parameters.
"""

from __future__ import absolute_import
from mapc2p import mapc2p
import numpy as np
from clawpack.visclaw import gridtools  # would work with v5.7.0


import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 200        # 100-200 is usually plenty for a 6x6 in figure
import matplotlib as mpl
mpl.rcParams.update({
    'savefig.dpi': 200,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
})

def rasterize_mesh(current_data):
    import matplotlib.pyplot as plt
    ax = plt.gca()
    for coll in ax.collections:      # pcolormesh / QuadMesh objects live here
        coll.set_rasterized(True)
    for im in ax.images:
        im.set_rasterized(True)

def rasterize_patches(current_data):
    import matplotlib.pyplot as plt
    ax = plt.gca()
    for coll in ax.collections:
        coll.set_rasterized(True)
    for p in ax.patches:
        p.set_rasterized(True)


#--------------------------
def setplot(plotdata=None):
#--------------------------
    
    """ 
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of visclaw.data.ClawPlotData.
    Output: a modified version of plotdata.
    
    """ 


    from clawpack.visclaw import colormaps

    if plotdata is None:
        from clawpack.visclaw.data import ClawPlotData
        plotdata = ClawPlotData()


    plotdata.clearfigures() # clear any old figures,axes,items data

    
    # Figure for pcolor plot
    plotfigure = plotdata.new_plotfigure(name='pcolor', figno=0)
    plotfigure.kwargs = {'figsize':(6,6)}

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    # plotaxes.xlimits = [-8,8]
    # plotaxes.ylimits = [-8,8]
    plotaxes.title = r'$|q_0|=\sqrt{z_0^2+w_0^2}$'#'Solution'
    plotaxes.title_t_format  = '%.2f'
    plotaxes.scaled = True
    plotaxes.afteraxes = rasterize_mesh      # <-- add this to the pcolor figure

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = plot_mass
    plotitem.pcolor_cmap = 'Greys' #'turbo'#colormaps.yellow_red_blue
    plotitem.pcolor_cmin = 0.95
    plotitem.pcolor_cmax = 1.05
    plotitem.add_colorbar = False #[False]+9*[True]

    plotitem.amr_celledges_show = [0,0,0]  
    plotitem.amr_patchedges_show = [0,0,0]
    plotitem.MappedGrid = True
    plotitem.mapc2p = mapc2p
    # plotitem.amr_patchedges_color = ['green','k']


    # # Figure for contour plot
    # plotfigure = plotdata.new_plotfigure(name='contour', figno=1)

    # # Set up for axes in this figure:
    # plotaxes = plotfigure.new_plotaxes()
    # plotaxes.xlimits = [0,1]
    # plotaxes.ylimits = [0,1]
    # plotaxes.title = 'Solution'
    # plotaxes.scaled = True
    # plotaxes.afteraxes = addgauges

    # # Set up for item on these axes:
    # plotitem = plotaxes.new_plotitem(plot_type='2d_contour')
    # plotitem.plot_var = 0
    # plotitem.contour_nlevels = 20
    # plotitem.contour_min = 0.01
    # plotitem.contour_max = 0.99
    # plotitem.amr_contour_colors = ['r','g','b']  # color on each level
    # plotitem.amr_patch_bgcolor = ['#ffeeee', '#eeeeff', '#eeffee']
    # plotitem.celledges_show = 0
    # plotitem.patchedges_show = 0


    # # Figure for grid cells
    plotfigure = plotdata.new_plotfigure(name='cells', figno=1)
    plotfigure.kwargs = {'figsize':(6,6)}

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    # plotaxes.xlimits = [-8,8]
    # plotaxes.ylimits = [-8,8]
    plotaxes.title = 'Grid patches'
    plotaxes.title_t_format  = '%.2f'
    plotaxes.scaled = True
    plotaxes.afteraxes = rasterize_patches      # <-- add this to the pcolor figure

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_patch')
    plotitem.amr_patch_bgcolor = ['#6C7A96', "#7D8C7A", "#A89A72"]
    plotitem.amr_celledges_show = [0,0,0]
    plotitem.amr_patchedges_show = [1,1,1]
    plotitem.amr_patchedges_color = ['#8B95A8', '#6D7B6B', '#7C7158']
    plotitem.MappedGrid = True
    plotitem.mapc2p = mapc2p

    # #-----------------------------------------
    # # Figures for gauges
    # #-----------------------------------------
    # plotfigure = plotdata.new_plotfigure(name='q', figno=300, \
    #                 type='each_gauge')
    # plotfigure.clf_each_gauge = True

    # # Set up for axes in this figure:
    # plotaxes = plotfigure.new_plotaxes()
    # plotaxes.xlimits = 'auto'
    # plotaxes.ylimits = 'auto'
    # plotaxes.title = 'q'

    # # Plot q as blue curve:
    # plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    # plotitem.plot_var = 0
    # plotitem.plotstyle = 'b-'

    
    # Parameters used only when creating html and/or latex hardcopy
    # e.g., via visclaw.frametools.printframes:

    plotdata.printfigs = True                # print figures
    plotdata.print_format = 'pdf'            # file format
    plotdata.print_framenos = [i for i in range(11)]#'all'          # list of frames to print
    plotdata.print_fignos = 'all'            # list of figures to print
    plotdata.html = True                     # create html files of plots?
    plotdata.html_homelink = '../README.html'   # pointer for top of index
    plotdata.html_movie = 'JSAnimation'      # new style, or "4.x" for old style
    plotdata.latex = True                    # create latex file of plots?
    plotdata.latex_figsperline = 2           # layout of plots
    plotdata.latex_framesperline = 1         # layout of plots
    plotdata.latex_makepdf = False           # also run pdflatex?
    plotdata.parallel = True                 # make multiple frame png's at once

    return plotdata

def plot_mass(current_data):
    return np.sqrt(current_data.q[0,:,:]**2 + current_data.q[3,:,:]**2)
    
    
# To plot gauge locations on pcolor or contour plot, use this as
# an afteraxis function:

def addgauges(current_data):
    from clawpack.visclaw import gaugetools
    gaugetools.plot_gauge_locations(current_data.plotdata, \
         gaugenos='all', format_string='ko', add_labels=True)
