import numpy as np
import matplotlib.pyplot as plt
from clawpack.pyclaw import Solution
from clawpack.visclaw import gridtools  # would work with v5.7.0
from clawpack import pyclaw


xout = np.linspace(-20,20,500)
yout = np.linspace(-20,20.,500)
x_slice = yout*0
XOUT, YOUT = np.meshgrid(xout,yout, indexing='ij')
XOUT, YOUT = np.meshgrid(xout,yout, indexing='ij')


sols = []
rhos = []
rho_slices = []

for frame in [0,5,7]:
    sol = Solution(frame, read_aux=False, path='./_output')
    sols.append(sol)
    psi_real = gridtools.grid_output_2d(sol, 0, XOUT, YOUT)
    psi_imag = gridtools.grid_output_2d(sol, 3, XOUT, YOUT)

    rho = np.sqrt(psi_real**2 + psi_imag**2)    
    rho_slice = gridtools.grid_output_2d(sol, 0, x_slice, yout)**2 + gridtools.grid_output_2d(sol, 3, x_slice, yout)**2
    rhos.append(rho)


fig, ax = plt.subplots(2,3,figsize=(12,8))
fs = 12
plt.rcParams.update({'font.size': fs})
for i in range(3):
    ax[0,i].pcolormesh(XOUT, YOUT, rhos[i], cmap='Purples', vmin=0., vmax=3,rasterized=True)
    ax[0,i].set_title(f't = {sols[i].t:.0f}')
    ax[0,i].set_aspect('equal')
    ax[0,i].set_xlabel('x')
    # ax[0,i].set_ylabel('y')
    # fig.colorbar(ax[0,i].collections[0], ax=ax[0,i])
    
    ax[1,i].plot(yout, rho_slices[i],color='#4B0082')
    # ax[1,i].set_title(f'Time = {sols[i].t:.2f}')
    ax[1,i].set_xlabel('y')

ax[0,0].set_ylabel('y')
ax[1,0].set_ylabel(r'$\rho=|q_0|^2\approx |u|^2$')

ax[1,0].set_ylim(1.8,2.2)
#Save as png
plt.tight_layout()
plt.savefig('transverse_instability_focusing_soliton.pdf', dpi=300)
plt.show()
