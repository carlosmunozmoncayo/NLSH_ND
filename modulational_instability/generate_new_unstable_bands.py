import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from cmcrameri import cm

fs1 = 15
fs2 = 15

def new_unstable_band(tau, p, kappasigmap):
    a = 1 - tau*kappasigmap
    b = (p-1)*kappasigmap/2
    denom = 4*tau*a**2 + 8*tau**2*a*b
    return -a**3/denom if denom <0 else np.nan

def band_boundaries(tau, p):
    upper = 1/tau if tau > 0 else np.nan
    lower = -1/(tau*(p-2)) if tau > 0 else np.nan
    return lower, upper

max_tau = 1
max_kappasigmap = 10
tau_list = np.linspace(0., max_tau, 400)
kappasigmap_list = np.linspace(-max_kappasigmap, max_kappasigmap, 400)
T, K = np.meshgrid(tau_list, kappasigmap_list)

p_list = [3, 5, 7]
vectorized_new_unstable_band = np.vectorize(new_unstable_band)
vectorized_band_boundaries = np.vectorize(band_boundaries)
Zs = [vectorized_new_unstable_band(T, p, K) for p in p_list]
lower_bounds = [vectorized_band_boundaries(tau_list, p)[0] for p in p_list]
upper_bounds = [vectorized_band_boundaries(tau_list, p)[1] for p in p_list]

# one norm for all three panels
cmap = "RdBu"#cm.vanimo
norm = colors.SymLogNorm(linthresh=0.01, vmin=-100, vmax=100, base=10)

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True,
                        gridspec_kw={'wspace': 0})

for ax, p, Z in zip(axs, p_list, Zs):
    ax.pcolormesh(T, K, Z, cmap=cmap, norm=norm)
    ax.set_xlabel(r'$\tau$', fontsize=fs2)
    ax.set_title(rf'$p={p}$', fontsize=fs2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.tick_params(axis='both', which='major', labelsize=fs1)
    ax.plot(tau_list, upper_bounds[p_list.index(p)], label=r'$\frac{1}{\tau}$', color='magenta', lw=2)
    ax.plot(tau_list, lower_bounds[p_list.index(p)], label=r'$\frac{1}{\tau(2-p)}$', color='cyan', lw=2)
    ax.set_ylim(-max_kappasigmap, max_kappasigmap)
    #Color gray for the stable region
    ax.fill_between(tau_list, lower_bounds[p_list.index(p)], upper_bounds[p_list.index(p)], color='gray', alpha=0.5, label='Stable region')


axs[1].legend(loc='lower right', fontsize=fs2)


axs[0].set_ylabel(r'$\kappa \sigma^{p-1}$', fontsize=fs2)

fig.subplots_adjust(right=0.87)
cbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.77])
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=cbar_ax, extend='both')
cbar.set_label(r'$\theta(\kappa\sigma^{p-1}, \tau)$', fontsize=fs2)
cbar.ax.tick_params(labelsize=fs1)
plt.savefig('new_MI_unstable_band.png', bbox_inches='tight', dpi=300)