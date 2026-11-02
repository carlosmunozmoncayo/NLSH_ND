import numpy as np
import matplotlib.pyplot as plt
from clawpack.pyclaw import Solution
from clawpack.visclaw import gridtools  # would work with v5.7.0
from mapc2p import mapc2p
# reconstruct complex fields
from matplotlib.pyplot import quiver


npoints = 501

xout = np.linspace(20-4.08,40-4.08,npoints)
yout = np.linspace(-10.,10.,npoints)
XOUT, YOUT = np.meshgrid(xout,yout, indexing='ij')
XOUT, YOUT = np.meshgrid(xout,yout, indexing='ij')

XPLOT, YPLOT = mapc2p(XOUT, YOUT)
RADII = np.sqrt((XPLOT-20.0)**2 + (YPLOT)**2)
frame_numbers = [45, 125, 165]  # Example frame numbers

#Read and interpolate data first for Neumann BCs
sols = {}
for frame_no in frame_numbers:
    framesoln = Solution(frame_no, path="_output_neumann")   
    psi_real = gridtools.grid_output_2d(framesoln, 0, XOUT, YOUT)
    psi_imag = gridtools.grid_output_2d(framesoln, 3, XOUT, YOUT)
    u_real = gridtools.grid_output_2d(framesoln, 1, XOUT, YOUT) #x-velocity
    v_real = gridtools.grid_output_2d(framesoln, 2, XOUT, YOUT) #y-velocity
    u_imag = gridtools.grid_output_2d(framesoln, 4, XOUT, YOUT) #x-velocity
    v_imag = gridtools.grid_output_2d(framesoln, 5, XOUT, YOUT) #y-velocity
    t = framesoln.t
    sols[frame_no] = (psi_real, psi_imag, u_real, v_real, u_imag, v_imag, t)

sols_dirichlet = {}
for frame_no in frame_numbers:
    framesoln = Solution(frame_no, path="_output_dirichlet")  
    psi_real = gridtools.grid_output_2d(framesoln, 0, XOUT, YOUT)
    psi_imag = gridtools.grid_output_2d(framesoln, 3, XOUT, YOUT)
    u_real = gridtools.grid_output_2d(framesoln, 1, XOUT, YOUT) #x-velocity
    v_real = gridtools.grid_output_2d(framesoln, 2, XOUT, YOUT) #y-velocity
    u_imag = gridtools.grid_output_2d(framesoln, 4, XOUT, YOUT) #x-velocity
    v_imag = gridtools.grid_output_2d(framesoln, 5, XOUT, YOUT) #y-velocity
    t = framesoln.t
    sols_dirichlet[frame_no] = (psi_real, psi_imag, u_real, v_real, u_imag, v_imag, t)


sols_potential_10000 = {}
for frame_no in frame_numbers:
    framesoln = Solution(frame_no, path="_output_potential")   
    psi_real = gridtools.grid_output_2d(framesoln, 0, XOUT, YOUT)
    psi_imag = gridtools.grid_output_2d(framesoln, 3, XOUT, YOUT)
    u_real = gridtools.grid_output_2d(framesoln, 1, XOUT, YOUT) #x-velocity
    v_real = gridtools.grid_output_2d(framesoln, 2, XOUT, YOUT) #y-velocity
    u_imag = gridtools.grid_output_2d(framesoln, 4, XOUT, YOUT) #x-velocity
    v_imag = gridtools.grid_output_2d(framesoln, 5, XOUT, YOUT) #y-velocity
    t = framesoln.t
    sols_potential_10000[frame_no] = (psi_real, psi_imag, u_real, v_real, u_imag, v_imag, t)


fs1 = 20
fs2 = 18
#Update all font sizes in matplotlib
plt.rcParams.update({'font.size': fs2, 'axes.titlesize': fs1, 'axes.labelsize': fs2, 'xtick.labelsize': fs2, 'ytick.labelsize': fs2, 'legend.fontsize': fs2})
fig, axs = plt.subplots(3, 3, figsize=(15, 9), constrained_layout=True)  # 3 rows, 3 columns
# Flatten for easy iteration
# axs = axs.flatten()

# We'll store one of the pcolormesh handles to use for the colorbar
pcm = None
cmap = 'Purples_r'  # Red-Blue reversed colormap

for ax, frame_no in zip(axs[0,:], frame_numbers):
    psi_real, psi_imag, u_real, v_real, u_imag, v_imag, t = sols[frame_no]

    rho = np.sqrt(psi_real**2 + psi_imag**2)
    rho = np.where(RADII <= 1.5, np.nan, rho)  # avoid division by zero

    psi = psi_real + 1j * psi_imag
    u = u_real + 1j * u_imag
    v = v_real + 1j * v_imag

    # compute current components Jx, Jy
    Jx = np.imag(np.conj(psi) * u)
    Jy = np.imag(np.conj(psi) * v)

    eps = 1e-12
    vx = np.where(rho > eps, Jx / rho, 0.0)
    vy = np.where(rho > eps, Jy / rho, 0.0)

    ######################################
    # Normalized quiver only around vortices
    rho0 = 0.6
    rho1 = 0.7
    mask = np.logical_and(rho > rho0, rho < rho1)
    skip = 50       
    mag = np.ones_like(vx)
    vx_dir = np.zeros_like(vx)
    vy_dir = np.zeros_like(vy)
    vx_dir[mask] = vx[mask] / mag[mask]
    vy_dir[mask] = vy[mask] / mag[mask]

    # Pcolormesh (store the handle for colorbar)
    pcm = ax.pcolormesh(XPLOT, YPLOT, rho**2, shading='auto', cmap=cmap, rasterized=True)

    # Add quiver
    ax.quiver(
        XPLOT.T[mask.T][::skip],
        YPLOT.T[mask.T][::skip],
        vx_dir.T[mask.T][::skip],
        vy_dir.T[mask.T][::skip],
        color='k', width=0.003, scale=15, zorder=1000
    )

    ###################################
    #Another quiver at the inflow side
    
    mask_inflow = XPLOT<30.0
    skip = 200
    
    # mask_inflow = np.logical_and(mask_inflow, np.abs(YPLOT)>1.0)
    mask_inflow = np.logical_and(mask_inflow, RADII>1.5)
    mask_inflow = np.logical_and(mask_inflow, rho>rho1)
    
    ax.quiver(
        XPLOT.T[mask_inflow.T][::skip],
        YPLOT.T[mask_inflow.T][::skip],
        vx.T[mask_inflow.T][::skip],
        vy.T[mask_inflow.T][::skip],
        color='k', width=0.003, scale=15, zorder=1000
    )
    

    ax.set_xlim(xout[0], xout[-1])
    ax.set_ylim(-6, 6)

    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = 20 + 1.5 * np.cos(theta)
    y_circle = 0 + 1.5 * np.sin(theta)
    ax.plot(x_circle, y_circle, 'k-', linewidth=0.5, zorder=1)
    ax.fill_between(x_circle, y_circle, -10, where=(y_circle > -10), color='darkgray', zorder=0)
    ax.fill_between(x_circle, y_circle, 10, where=(y_circle < 10), color='darkgray', zorder=0)

    # if frame_no in [48, 125]:
    #     ax.set_ylabel('y')
    # if frame_no in [125, 185]:
    #     ax.set_xlabel('x')
    ax.set_ylabel('t = %.2f' % t)
    # ax.set_title('t = %.2f' % t)
for ax, frame_no in zip(axs[1,:], frame_numbers):
    psi_real, psi_imag, u_real, v_real, u_imag, v_imag, t = sols_dirichlet[frame_no]

    rho = np.sqrt(psi_real**2 + psi_imag**2)
    rho = np.where(RADII <= 1.5, np.nan, rho)  # avoid division by zero

    psi = psi_real + 1j * psi_imag
    u = u_real + 1j * u_imag
    v = v_real + 1j * v_imag

    # compute current components Jx, Jy
    Jx = np.imag(np.conj(psi) * u)
    Jy = np.imag(np.conj(psi) * v)

    eps = 1e-10
    vx = np.where(rho > eps, Jx / rho, 0.0)
    vy = np.where(rho > eps, Jy / rho, 0.0)

    ######################################
    # Normalized quiver only around vortices
    rho0 = 0.35
    rho1 = 0.5
    mask = np.logical_and(rho > rho0, rho < rho1)
    skip = 50       
    mag = np.ones_like(vx)
    vx_dir = np.zeros_like(vx)
    vy_dir = np.zeros_like(vy)
    vx_dir[mask] = vx[mask] / mag[mask]
    vy_dir[mask] = vy[mask] / mag[mask]

    # Pcolormesh (store the handle for colorbar)
    pcm = ax.pcolormesh(XPLOT, YPLOT, rho**2, shading='auto', cmap=cmap, rasterized=True)

    # Add quiver
    ax.quiver(
        XPLOT.T[mask.T][::skip],
        YPLOT.T[mask.T][::skip],
        vx_dir.T[mask.T][::skip],
        vy_dir.T[mask.T][::skip],
        color='k', width=0.003, scale=15, zorder=1000
    )

    ###################################
    #Another quiver at the inflow side
    
    mask_inflow = XPLOT<30.0
    skip = 100
    
    # mask_inflow = np.logical_and(mask_inflow, np.abs(YPLOT)>1.0)
    mask_inflow = np.logical_and(mask_inflow, RADII>1.5)
    mask_inflow = np.logical_and(mask_inflow, rho>rho1)
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = 20 + 1.5 * np.cos(theta)
    y_circle = 0 + 1.5 * np.sin(theta)
    ax.plot(x_circle, y_circle, 'k-', linewidth=0.5, zorder=10)
    ax.fill_between(x_circle, y_circle, -10, where=(y_circle > -10), color='darkgray', zorder=0)
    ax.fill_between(x_circle, y_circle, 10, where=(y_circle < 10), color='darkgray', zorder=0)
    

    ax.set_xlim(xout[0], xout[-1])
    ax.set_ylim(-6, 6)

list_sols = [sols_potential_10000]#, sols_potential, sols_potential_100]
for idx, sol in enumerate(list_sols):
    idx_offset = idx + 2  # since the first column is Neumann BCs
    for ax, frame_no in zip(axs[idx_offset,:], frame_numbers):
        psi_real, psi_imag, u_real, v_real, u_imag, v_imag, t = sol[frame_no]

        rho = np.sqrt(psi_real**2 + psi_imag**2)
        # rho = np.where(RADII <= 1.5, np.nan, rho)  # avoid division by zero

        psi = psi_real + 1j * psi_imag
        u = u_real + 1j * u_imag
        v = v_real + 1j * v_imag

        # compute current components Jx, Jy
        Jx = np.imag(np.conj(psi) * u)
        Jy = np.imag(np.conj(psi) * v)

        eps = 1e-2
        vx = np.where(rho > eps, Jx / rho, 0.0)
        vy = np.where(rho > eps, Jy / rho, 0.0)
        #Zero out velocities exceeding a threshold to avoid large arrows
        threshold = 1
        vx[np.abs(vx) > threshold] = 0.0
        vy[np.abs(vy) > threshold] = 0.0

        ######################################
        # Normalized quiver only around vortices
        rho0 = 0.6
        rho1 = 0.7
        mask = np.logical_and(rho > rho0, rho < rho1)
        skip = 50       
        mag = np.ones_like(vx)
        #Normalize the velocity vectors for quiver
        # mag = np.sqrt(vx**2 + vy**2)
        vx_dir = np.zeros_like(vx)
        vy_dir = np.zeros_like(vy)
        vx_dir[mask] = vx[mask] / mag[mask]
        vy_dir[mask] = vy[mask] / mag[mask]

        # Pcolormesh (store the handle for colorbar)
        pcm = ax.pcolormesh(XPLOT, YPLOT, rho**2, shading='auto', cmap=cmap, rasterized=True)

        # Add quiver
        ax.quiver(
            XPLOT.T[mask.T][::skip],
            YPLOT.T[mask.T][::skip],
            vx_dir.T[mask.T][::skip],
            vy_dir.T[mask.T][::skip],
            color='k', width=0.003, scale=15, zorder=1000
        )
        ###################################
        #Another quiver at the inflow side
        
        mask_inflow = XPLOT<30.0
        skip = 100
        
        # mask_inflow = np.logical_and(mask_inflow, np.abs(YPLOT)>1.0)
        mask_inflow = np.logical_and(mask_inflow, RADII>1.5)
        mask_inflow = np.logical_and(mask_inflow, rho>rho1)
        

        ax.set_xlim(xout[0], xout[-1])
        ax.set_ylim(-6, 6)

# --- Add one shared colorbar on the left ---
fig.colorbar(pcm, ax=axs, label=r'$\rho$', orientation='vertical', fraction=0.02, pad=0.02)



row_titles = ['Neumann BCs', 'Dirichlet BCs', r'Potential']#, r'$V_0(x)=-10^3$',r'$V_0(x)=-10^2$']
for row, title in enumerate(row_titles):
    left_ax = axs[row,0]
    left_ax.set_ylabel(title, labelpad=10)

col_titles = [r'$t=11.25$', r'$t=31.25$', r'$t=41.25$']
for col, title in enumerate(col_titles):
    top_ax = axs[0,col]
    top_ax.set_title(title, pad=2)


#Set color limits
vmin = 0.0
vmax = 1.1
for ax in axs[0,:]:
    pcm = ax.collections[0]
    pcm.set_clim(vmin, vmax)
for ax in axs[1,:]:
    pcm = ax.collections[0]
    pcm.set_clim(vmin, vmax)
for ax in axs[2,:]:
    pcm = ax.collections[0]
    pcm.set_clim(vmin, vmax)
# for ax in axs[3,:]:
#     pcm = ax.collections[0]
#     pcm.set_clim(vmin, vmax)
#Share x and y axes
for ax in axs[:,0]:
    ax.label_outer()
for ax in axs[:,1]:
    ax.label_outer()
for ax in axs[:,2]:
    ax.label_outer()
for ax in axs[0,:]:
    ax.label_outer()

#Fix the aspect ratio of the axes to be equal
for ax in axs.flatten():
    ax.set_aspect('equal', adjustable='box')


#Save figure as pdf
fig.savefig("vortex_plots.pdf",dpi=300)
plt.show()