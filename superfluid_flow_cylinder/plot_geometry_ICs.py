import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from clawpack.visclaw import gridtools  # would work with v5.7.0

from clawpack.pyclaw import Solution


from map_module_f2py import map_module
mapc2p = map_module.mapc2p

sol = Solution()
sol.read(frame=0, path="_output_single_grid",read_aux=True)
XC,YC = sol.grid.c_nodes
XP, YP = np.zeros_like(XC), np.zeros_like(YC)
for i in range(XC.shape[0]):
    for j in range(XC.shape[1]):
        XP[i, j], YP[i, j] = mapc2p(XC[i, j], YC[i, j])

XC_centers, YC_centers = sol.grid.c_centers
XP_centers, YP_centers = np.zeros_like(XC_centers), np.zeros_like(YC_centers)
for i in range(XC_centers.shape[0]):
    for j in range(XC_centers.shape[1]):
        XP_centers[i, j], YP_centers[i, j] = mapc2p(XC_centers[i, j], YC_centers[i, j])
# Z_centers = np.where((XP_centers - 20)**2 + YP_centers**2 <= 1.5**2, 1, np.nan)
Z_numerical = sol.aux[7,:,:]
Z_numerical = np.where(Z_numerical<0.5, np.nan, Z_numerical)
Z_numerical_boundary = Z_numerical.copy()
#Where Z_numerical is gt 0.5 and has a neighbor that is nan, set it to 2
for i in range(Z_numerical.shape[0]):
    for j in range(Z_numerical.shape[1]):
        if not np.isnan(Z_numerical[i,j]):
            #Check neighbors
            neighbors = []
            if i > 0:
                neighbors.append(Z_numerical[i-1,j])
            if i < Z_numerical.shape[0]-1:
                neighbors.append(Z_numerical[i+1,j])
            if j > 0:
                neighbors.append(Z_numerical[i,j-1])
            if j < Z_numerical.shape[1]-1:
                neighbors.append(Z_numerical[i,j+1])
            if any(np.isnan(n) for n in neighbors):
                Z_numerical_boundary[i,j] = 2
            else:
                Z_numerical_boundary[i,j] = np.nan


fig, axs = plt.subplots(1, 2, figsize=(20, 6))

colors = {
    'sponge':      '#8DC3A7',   # muted green-teal
    'refinement':  '#F4E3A3',   # pale yellow-gold
    'obstacle':    '#444444',   # dark neutral grey
    'inclusion':   '#D400D4',   # muted red-brown
}

#Mapped grid plot
#Undersample the grid for better visibility
for i in range(0, XP.shape[0], 10):
    axs[0].plot(XP[i, :], YP[i, :], 'k-', lw=1)  # Horizontal lines
for j in range(0, XP.shape[1], 10):
    axs[0].plot(XP[:, j], YP[:, j], 'k-', lw=1)  # Vertical lines
axs[0].set_title('Mapped Grid in Physical Domain')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_aspect('equal', adjustable='box')

#Plot circle centered at (20,0) with radius 1.5
circle = plt.Circle((20, 0), 1.5, color=colors['inclusion'], fill=False, lw=2)
axs[0].add_artist(circle)
axs[0].fill_betweenx(y=[-50, 50], x1=-30, x2=-10, color=colors['sponge'], alpha=0.5, zorder=0)
axs[0].fill_betweenx(y=[-50, 50], x1=50, x2=70, color=colors['sponge'], alpha=0.5, zorder=0)
axs[0].fill_betweenx(y=[30, 50], x1=-10, x2=50, color=colors['sponge'], alpha=0.5, zorder=0)
axs[0].fill_betweenx(y=[-50, -30], x1=-10, x2=50, color=colors['sponge'], alpha=0.5, zorder=0)
#shade light yellow the square [16,24]x[-10,10]
axs[0].fill_betweenx(y=[-10, 10], x1=16, x2=36, color=colors['refinement'], alpha=0.7, zorder=0)

axs[0].set_xlim([-30, 70])
axs[0].set_ylim([-15, 50])

# Create an inset (a smaller Axes inside the main one)
axins = inset_axes(axs[0], width="65%", height="65%", loc="upper left")

# Plot the same data in the inset
# axins.plot(XP, YP)
for i in range(XP.shape[0]):
    axins.plot(XP[i, :], YP[i, :], 'k-', lw=0.5)  # Horizontal lines
for j in range(XP.shape[1]):
    axins.plot(XP[:, j], YP[:, j], 'k-', lw=0.5)  # Vertical lines

#Scatter Z
axins.scatter(XP_centers, YP_centers, c=Z_numerical,  s=2, edgecolors='k')
axins.scatter(XP_centers, YP_centers, c=Z_numerical_boundary,  s=10 ,marker='x')

# axins.scatter(XP_centers, YP_centers, c=Z_nummerical, cmap='plasma', s=10, edgecolors='r')

# Set zoomed region limits
x1, x2, y1, y2 = 18, 22, -2, 2  # region to zoom into
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
#shade light yellow the square [16,24]x[-10,10]
axins.fill_betweenx(y=[-10, 10], x1=16, x2=36, color=colors['refinement'], alpha=0.7, zorder=0)
circle_inset = plt.Circle((20, 0), 1.5, color=colors['inclusion'], fill=False, lw=2)
axins.add_artist(circle_inset)
#Remove ticks
axins.set_xticks([])
axins.set_yticks([])



# Optionally, mark the zoomed region on the main plot
mark_inset(axs[0], axins, loc1=3, loc2=4, fc="none", ec="black", lw=2.5,zorder=11)
for spine in axins.spines.values():
    spine.set_zorder(11)
    spine.set_linewidth(2.5)      # increase thickness
    spine.set_edgecolor('black')  # optional: change color
axins.set_zorder(11)

#Set aspect ratio to equal in the inset
axins.set_aspect('equal', adjustable='box')

##############################
#Slice plot
##############################
xout = np.linspace(-30, 70, 800)
yout = np.zeros_like(xout)
z0 = gridtools.grid_output_2d(sol, 0, xout, yout)
w0 = gridtools.grid_output_2d(sol, 3, xout, yout)
zetaout = np.sqrt(z0**2 + w0**2)
for i in range(len(xout)):
    xout[i], yout[i] = mapc2p(xout[i], yout[i])

for out in [z0, w0, zetaout]:
    out[np.where(np.logical_and(xout >= 18.5, xout <= 21.5))] = np.nan


#Generator
xg = np.linspace(-30,-10,10)
yg1 = 1.3*np.ones(xg.shape)
yg2 = -1.3*np.ones(xg.shape)
axs[1].fill_between(xg, yg1, yg2, color=colors['sponge'], alpha=0.5)
axs[1].text(-20,-1.28,'Sponge\nlayer', verticalalignment='bottom', horizontalalignment='center')

#Absorber
xg = np.linspace(50,70,10)
yg1 = 1.3*np.ones(xg.shape)
yg2 = -1.3*np.ones(xg.shape)
axs[1].fill_between(xg, yg1, yg2, color=colors['sponge'], alpha=0.5)
axs[1].text(60,-1.28,'Sponge\nlayer', verticalalignment='bottom',horizontalalignment='center')

#Fine region
xg = np.linspace(16,36,10)
yg1 = 1.3*np.ones(xg.shape)
yg2 = -1.3*np.ones(xg.shape)
axs[1].fill_between(xg, yg1, yg2, color=colors['refinement'], alpha=0.7)
axs[1].text(31,-1.28,'High-refinement\nzone', verticalalignment='bottom', horizontalalignment='center',zorder=6)

#Two red thick vertical lines at x = 18.5 and x=21.5, going from y = -1.3 to y = 1.3
x = [18.5,18.5]
y = [-1.3, 1.3]
axs[1].plot(x, y, color=colors['inclusion'], lw=2, label='Inclusion ',zorder=5)
x = [21.5,21.5]
y = [-1.3, 1.3]
axs[1].plot(x, y, color=colors['inclusion'], lw=2,zorder=5)






axs[1].plot(xout, zetaout, '#FF5A1F', linestyle='-.', label=r'$\rho=|q_0|$')
axs[1].plot(xout, z0, 'b--', label=r'$z_0=$Re$(q_0)$')
axs[1].plot(xout, w0, 'k:', label=r'$w_0=$Im$(q_0)$')

axs[1].set_xlabel('x')
axs[1].set_ylabel('Amplitude (arb. units)')
axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),fancybox=True, shadow=True, ncol=4)
axs[1].set_title(r'Initial Condition along $y=0$')

#Save as pdf
plt.savefig('mapped_grid_and_initial_condition.pdf', bbox_inches='tight')
plt.show()