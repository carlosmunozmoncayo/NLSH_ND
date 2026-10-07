#!/usr/bin/env python
# encoding: utf-8
r"""
Two-dimensional NLSH on a rectangular domain
a diagonal NLSH plane wave with periodic boundary conditions
"""
 
import numpy as np
from clawpack.visclaw import gridtools  # would work with v5.7.0
from modules import rs_nlsh_2d
import warnings

tau = 0.01
sigma = 1.0
pnonlinear = 3.0
kappa = -1.0
k_wavenumber = 1 #In the diagonal direction, so kx = ky = k/sqrt(2)
L_diagonal = 4*np.pi/k_wavenumber
L_x = L_diagonal/np.sqrt(2)
L_y = L_diagonal/np.sqrt(2)

def setup(kernel_language='Fortran', use_petsc=False, outdir='./_output', 
          solver_type='classic',
          tau = tau,
          disable_output=False, num_cells=(50, 50), tfinal = 5, branch="physical",
            pnonlinear = pnonlinear, kappa = kappa, sigma = sigma):
    from clawpack import riemann

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if solver_type=='classic':
        solver=pyclaw.ClawSolver2D(rs_nlsh_2d)
        solver.dimensional_split=False
        solver.limiters = 0 #No limiting to the best of my knowledge
        #The loop in limiters.tvd.limit should be bypassed since 0 is not a valid key
        solver.step_source = exact_source_step_NLSH
        solver.source_split = 2
        solver.order = 2
        solver.cfl_max = 1
        solver.cfl_desired = 0.99
    elif solver_type=='sharpclaw':
        solver=pyclaw.SharpClawSolver2D(rs_nlsh_2d)
        solver.dq_src = sharpclaw_source_step_NLSH
        solver.weno_order = 5
        solver.time_integrator = 'SSP33'
        solver.lim_type   = 2
        solver.cfl_max = 1.0*np.sqrt(tau)
        solver.cfl_desired = 0.9*np.sqrt(tau)
    #     raise ValueError("Only classic solver for the moment.")

    solver.fwave = False

    x = pyclaw.Dimension(-L_x/2,L_x/2,num_cells[0],name='x')
    y = pyclaw.Dimension(-L_y/2,L_y/2,num_cells[1],name='y')
    domain = pyclaw.Domain([x,y])

    num_eqn = 6
    num_waves = 4
    num_aux = 0 # Not using mapped grid in this example
    state = pyclaw.State(domain,num_eqn,num_aux)
    solver.num_waves = num_waves
    solver.num_eqn = num_eqn

    grid = state.grid
    X, Y = grid.p_centers


    state.problem_data['tau'] = tau
    state.problem_data['pnonlinear'] = pnonlinear
    state.problem_data['kappa'] = kappa

    solver.bc_lower[0]=pyclaw.BC.periodic
    solver.bc_upper[0]=pyclaw.BC.periodic
    solver.bc_lower[1]=pyclaw.BC.periodic
    solver.bc_upper[1]=pyclaw.BC.periodic

    solver.dt_initial = 1.e-6

    # Set initial condition
    u_real, u_imag, ux_real, ux_imag, uy_real, uy_imag = plane_wave_NLSH_2D(X,Y, 0.0,
                                                 kappa,sigma, k_wavenumber/np.sqrt(2), k_wavenumber/np.sqrt(2),tau=tau,p=3,branch=branch)

    state.q[:,:,:] = 0.
    state.q[0,:,:] = u_real
    state.q[1,:,:] = ux_real
    state.q[2,:,:] = uy_real
    state.q[3,:,:] = u_imag
    state.q[4,:,:] = ux_imag
    state.q[5,:,:] = uy_imag

    claw = pyclaw.Controller()
    claw.keep_copy = True
    if disable_output:
        claw.output_format = None
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.outdir = outdir
    claw.tfinal = tfinal
    claw.num_output_times = int(claw.tfinal*10)
    claw.write_aux_init = True
    claw.setplot = setplot
    if use_petsc:
        claw.output_options = {'format':'binary'}

    return claw


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

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = r'$\mathrm{Re}(q_0)$'
    plotaxes.scaled = False
    plotaxes.axescmd = 'axes([.3,.45,.5,.45])'

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = 0
    plotitem.pcolor_cmap = colormaps.yellow_red_blue
    plotitem.pcolor_cmin = -1
    plotitem.pcolor_cmax = 1
    plotitem.add_colorbar = True


    # Figure for cross section (x=y)
    plotaxes = plotfigure.new_plotaxes('x-slice')
    plotaxes.axescmd = 'axes([.1,.1,.8,.25])'
    plotaxes.title = 'Transect at x=y'
    def plot_xsec(current_data):
        from pylab import plot,linspace,zeros,ones,legend,xlabel,sqrt,grid,xlim
        from pylab import nan,where,ylim,loadtxt,arange
        from clawpack.pyclaw import Solution
        pd = current_data.plotdata
        frameno = current_data.frameno
        framesoln = Solution(frameno, path=pd.outdir, file_format=pd.format)
        
        xout = linspace(-np.sqrt(2)*np.pi,np.sqrt(2)*np.pi,1000)
        yout = np.copy(xout)
        
        z0 = gridtools.grid_output_2d(framesoln, 0, xout, yout)
        w0 = gridtools.grid_output_2d(framesoln, 3, xout, yout)
        zetaout = np.sqrt(z0**2 + w0**2)
        plot(xout, zetaout, 'm', label=r'$|q_0|$')
        plot(xout, z0, 'b--', label=r'Re$(q_0)$')
        plot(xout, w0, 'r:', label=r'Im$(q_0)$')
        legend()
        xlabel('x=y')#
        ylim(-2.,2.)
        grid(True)
    
    plotaxes.afteraxes = plot_xsec


    # # Figure for grid cells
    plotfigure = plotdata.new_plotfigure(name='cells', figno=1)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    # plotaxes.xlimits = [-10,10]
    # plotaxes.ylimits = [-1,1]
    plotaxes.title = 'Grid patches'
    plotaxes.scaled = False
    plotitem.add_colorbar = True

    plotdata.printfigs = True                # print figures
    plotdata.print_format = 'png'            # file format
    plotdata.print_framenos = 'all'          # list of frames to print
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


def exact_source_step_NLSH(solver,state,dt):
    """
    Exact source step for NLSH equations.
    """
    q = state.q
    tau = state.problem_data['tau']
    pnonlinear = state.problem_data['pnonlinear']
    kappa = state.problem_data['kappa']

    tauinv = 1.0 / tau
    if pnonlinear > 3.0:
        local_q_norm = np.sqrt(q[0,:,:]**2 + q[3,:,:]**2)
        a = kappa * local_q_norm**(pnonlinear - 1.0)
    else:
        a = kappa * (q[0,:,:]**2 + q[3,:,:]**2)
    c = np.cos(a * dt)
    s = np.sin(a * dt)
    c_tau = np.cos(tauinv * dt)
    s_tau = np.sin(tauinv * dt)
    expdtS = np.zeros((6,6,q.shape[1],q.shape[2]))
    expdtS[0,0,:,:] = c;        expdtS[0,3,:,:] = -s
    expdtS[1,1,:,:] = c_tau;    expdtS[1,4,:,:] = -s_tau
    expdtS[2,2,:,:] = c_tau;    expdtS[2,5,:,:] = -s_tau
    expdtS[3,0,:,:] = s;        expdtS[3,3,:,:] = c
    expdtS[4,1,:,:] = s_tau;    expdtS[4,4,:,:] = c_tau
    expdtS[5,2,:,:] = s_tau;    expdtS[5,5,:,:] = c_tau
    state.q = np.einsum('ijxy,jxy->ixy',expdtS,q)


def sharpclaw_source_step_NLSH(solver,state,dt):
    """
    Evaluate source term for NLSH equations and multiply by dt.
    """
    q = state.q
    tau = state.problem_data['tau']
    pnonlinear = state.problem_data['pnonlinear']
    kappa = state.problem_data['kappa']
    tauinv = 1.0 / tau
    if pnonlinear > 3.0:
        local_q_norm = np.sqrt(q[0,:,:]**2 + q[3,:,:]**2)
        a = kappa * local_q_norm**(pnonlinear - 1.0)
    else:
        a = kappa * (q[0,:,:]**2 + q[3,:,:]**2)
    dq = np.zeros(q.shape)
    dq[0,:,:] = -a * q[3,:,:]
    dq[1,:,:] = -tauinv * q[4,:,:]
    dq[2,:,:] = -tauinv * q[5,:,:]
    dq[3,:,:] = a * q[0,:,:]
    dq[4,:,:] = tauinv * q[1,:,:]
    dq[5,:,:] = tauinv * q[2,:,:]
    return dq * dt

def _plane_wave_params(k, sigma, kappa, p, tau):
    """Return a, b = 1 + a tau, and sqrt(D) for the NLSH dispersion relation."""
    a = kappa * sigma ** (p - 1.0)
    b = 1.0 + a * tau
    sD = np.sqrt((1.0 - a * tau) ** 2 + 4.0 * tau * k * k)
    return a, b, sD


def nlsh_dispersion(k, sigma, kappa, p, tau, branch="physical"):
    """Frequency omega and g = 1 + tau*omega of an NLSH plane wave, stably.

    branch : 'physical' (-> NLS as tau -> 0) or 'spurious' (~ -1/tau).
    Both omega and g are computed without cancellation, since
    q1 = i k / g * q0 needs g accurately too.
    """
    if tau <= 0.0:
        raise ValueError("tau must be positive (use nls_plane_wave for tau = 0)")
    a, b, sD = _plane_wave_params(k, sigma, kappa, p, tau)
    if a * tau >= 1.0 and k != 0.0:
        warnings.warn(f"a*tau = {a*tau:.3g} >= 1: the two branches are not "
                      "scale-separated; 'physical' is defined by continuity "
                      "from tau = 0 only.", RuntimeWarning, stacklevel=2)

    if k == 0.0:
        # Roots are exactly -a (q1 = 0) and -1/tau (q0 = 0); they cross at a*tau = 1.
        if branch == "physical":
            return -a, 1.0 - a * tau
        raise ValueError("The spurious mode at k = 0 has q0 = 0 and cannot be "
                         "parametrised by the q0 amplitude sigma.")

    # g+ = 1 + tau*omega+ = (1 - a tau + sqrt(D))/2, rewritten when a tau > 1
    # to avoid cancellation; g+ * g- = -tau k^2 fixes g-.
    if a * tau <= 1.0:
        g_phys = 0.5 * (1.0 - a * tau + sD)
    else:
        g_phys = 2.0 * tau * k * k / (a * tau - 1.0 + sD)

    if branch == "physical":
        return 2.0 * (k * k - a) / (b + sD), g_phys
    if branch == "spurious":
        return -(b + sD) / (2.0 * tau), -tau * k * k / g_phys
    raise ValueError("branch must be 'physical' or 'spurious'")


def nlsh_plane_wave(x, t, k, sigma, kappa, p, tau, branch="physical"):
    """Exact NLSH plane wave. Returns (q0, q1, omega)."""
    om, g = nlsh_dispersion(k, sigma, kappa, p, tau, branch)
    q0 = sigma * np.exp(1j * (k * x - om * t))
    q1 = (1j * k / g) * q0
    return q0, q1, om

def plane_wave_NLSH_2D(x,y,t,kappa,sigma,kx,ky,tau,p=3,branch="physical"):
    """
    Returns real and imaginary parts of q0(approximates u) and q1 (approximates u_x),
    where q0 = sigma*exp(i*(k*x - omega*t)) and
    q1 = q0_x*i*k/(tau*omega+1) are plane wave solutions to NLSH
    sgn =-1 for the stable branch
    """
    k = np.sqrt(kx**2 + ky**2)
    om, g = nlsh_dispersion(k, sigma, kappa, p, tau, branch)
    q0 = sigma * np.exp(1j * (kx * x + ky * y - om * t))
    q1 = (1j * kx / g) * q0
    q2 = (1j * ky / g) * q0
    q0_real = np.real(q0)
    q0_imag = np.imag(q0)
    q1_real = np.real(q1)
    q1_imag = np.imag(q1)
    q2_real = np.real(q2)
    q2_imag = np.imag(q2)
    return q0_real, q0_imag, q1_real, q1_imag, q2_real, q2_imag


if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup,setplot)
