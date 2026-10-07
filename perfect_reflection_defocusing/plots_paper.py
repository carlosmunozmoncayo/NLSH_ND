import numpy as np
from clawpack.pyclaw import Solution
from clawpack.visclaw import gridtools  # would work with v5.7.0
import matplotlib.pyplot as plt
from map_module_f2py import map_module
from matplotlib.ticker import MaxNLocator



mapc2p = map_module.vectorized_mapc2p


fs1 = 22
fs2 = 25

nx = 400#1600
ny = 80#320

nx_lines = 50
ny_lines = 10

def plot_amplitude_frames(frames, vmin=0.7, vmax=1.0, cmap='jet',
                          nx=nx, ny=ny, nx_lines=nx_lines, ny_lines=ny_lines,
                          equal_aspect=False, panel_height=3, hspace=0):

    # frame-independent geometry — build once
    XOUT, YOUT = np.meshgrid(np.linspace(0, 1, nx),
                             np.linspace(0, 1, ny), indexing='ij')
    XPLOT, YPLOT = mapc2p(XOUT, YOUT)

    XOUT_l, YOUT_l = np.meshgrid(np.linspace(0, 1, nx_lines),
                                 np.linspace(0, 1, ny_lines), indexing='ij')
    XPLOT_l, YPLOT_l = mapc2p(XOUT_l, YOUT_l)

    n = len(frames)
    fig, axs = plt.subplots(n, 1, figsize=(12, panel_height*n),
                            sharex=True, sharey=True,
                            gridspec_kw={'hspace': hspace})
    axs = np.atleast_1d(axs)

    for ax, frame in zip(axs, frames):
        framesoln = Solution()
        framesoln.read(path="_output",frame=frame, read_aux=True)

        z0 = gridtools.grid_output_2d(framesoln, 0, XOUT, YOUT)
        w0 = gridtools.grid_output_2d(framesoln, 3, XOUT, YOUT)
        amplitude = np.sqrt(z0**2 + w0**2)

        ax.pcolormesh(XPLOT, YPLOT, amplitude, shading='auto',
                      cmap=cmap, vmin=vmin, vmax=vmax)

        for i in range(XPLOT_l.shape[0]):
            ax.plot(XPLOT_l[i, :], YPLOT_l[i, :], color='w', linewidth=0.5)
        for j in range(XPLOT_l.shape[1]):
            ax.plot(XPLOT_l[:, j], YPLOT_l[:, j], color='w', linewidth=0.5)

        # time stamped inside the panel so hspace can stay at 0
        ax.text(0.85, 0.95, f't = {framesoln.t:.1f}', transform=ax.transAxes,
                ha='left', va='top', fontsize=fs1, color='k',
                bbox=dict(facecolor='w', alpha=1, edgecolor='k', pad=2))

        ax.set_ylabel('y', fontsize=fs2)
        ax.tick_params(axis='both', which='major', labelsize=fs1)
        if equal_aspect:
            ax.set_aspect('equal')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))

    axs[-1].set_xlabel('x', fontsize=fs2)

    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.89, 0.11, 0.015, 0.77])
    cbar = fig.colorbar(plt.cm.ScalarMappable(
                            norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
                        cax=cbar_ax)
    cbar.set_label(r'$|q_0|$', fontsize=fs2)
    cbar.ax.tick_params(labelsize=fs1)
    

    return fig, axs

hs = 0.1
nr = 2.5

fig, axs = plot_amplitude_frames([0, 1, 5, 11,14][::-1],equal_aspect=False, panel_height=nr,hspace=hs)#2.5)
fig.savefig('amplitude_perfect_defocusing.png', dpi=300, bbox_inches='tight')