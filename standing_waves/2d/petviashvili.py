r"""
This module implements the Petviashvili method to compute ground state solutions
of the NLSH (and NLS) equation in 2D:
\frac{\Delta}{1-\mu\tau} q_0 -\mu  = -\kappa q_0^3,
where \mu>0, \kappa>0, and \tau>0 are parameters which should satisfy 1-\mu\tau>0.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os

thisdir = os.path.dirname(os.path.abspath(__file__))



def run_and_save(tau=0.1,mu=1.0,kappa=1.0,
                 Lx=20.0,Ly=20.0,mx=512,my=512,
                 niters=100,filename=None,save=True):
    dx = Lx/mx
    dy = Ly/my
    x = np.linspace(-Lx/2,Lx/2,mx)
    y = np.linspace(-Ly/2,Ly/2,my)
    X,Y = np.meshgrid(x,y,indexing='ij')
    kx = 2*np.pi*np.fft.fftfreq(mx,d=dx)
    ky = 2*np.pi*np.fft.fftfreq(my,d=dy)
    KX,KY = np.meshgrid(kx,ky,indexing='ij')
    K2 = KX**2 + KY**2
    ddx = 1j*KX
    ddy = 1j*KY

    #Initial condition
    q0 = np.exp(-(X**2+Y**2))
    # Fourier symbol of linear operator
    Lk = (mu + K2/(1 - mu*tau))
    q0hat = np.fft.fftn(q0)

    for it in range(niters):
        #Nonlinear term (computed in physical space)
        Nq0 = kappa*q0**3
        Nq0hat = np.fft.fftn(Nq0)
        #Update in Fourier space
        q0hat_predictor = Nq0hat /Lk
        #Compute stabilizing factor (in physical space)
        Lq0 = np.fft.ifftn(Lk*q0hat).real
        s = (np.sum(q0*Lq0) / np.sum(q0*Nq0))
        #Corrector step
        q0hat = s**(3/2)*q0hat_predictor
        #Back to physical space
        q0 = np.fft.ifftn(q0hat).real
        if it%10==0:
            print(f"Iteration {it}, S={s:.6f}, max(q0)={np.max(q0):.6f}")
    print(f"Final iteration {niters}, S={s:.6f}, max(q0)={np.max(q0):.6f}")
    q1 = np.zeros((2,mx,my))
    q1[0,:,:] = np.fft.ifftn(ddx*q0hat).real/(1 - mu*tau)
    q1[1,:,:] = np.fft.ifftn(ddy*q0hat).real/(1 - mu*tau)

    dump_array = np.zeros((mx*my,5))
    dump_array[:,0] = np.ravel(X)
    dump_array[:,1] = np.ravel(Y)
    dump_array[:,2] = np.ravel(q0)
    dump_array[:,3] = np.ravel(q1[0,:,:])
    dump_array[:,4] = np.ravel(q1[1,:,:])

    if save:
        if filename is None:
            #Check if data directory exists
            if not os.path.exists(f"{thisdir}/data"):
                os.makedirs(f"{thisdir}/data")
            filename = f"{thisdir}/data/ground_state2D_tau{tau}_mu{mu}_kappa{kappa}_Lx{Lx}_Ly{Ly}_mx{mx}_my{my}.dat"

        np.savetxt(filename,dump_array,
                    header=f"Parameters: tau, mu, kappa, Lx, Ly, mx, my\n"+
                    f"{tau} {mu} {kappa} {Lx} {Ly} {mx} {my}\n"+
                    "Columns: x, y, q0, q1_x, q1_y")
        save_to_fortran(filename)
        return None
    else: 
        params = {'tau': tau, 'mu': mu, 'kappa': kappa, 'Lx': Lx, 'Ly': Ly, 'mx': mx, 'my': my}
        return params, X, Y, q0, q1
    
def read(filename):
    #Check if file exists, if not, try any .dat file in the folder
    try:
        print(f"Reading data file {filename}")
        with open(filename,'r') as f:
            pass
    except FileNotFoundError:
        import glob
        files = glob.glob("data/ground_state2D*.dat")
        if len(files)==0:
            print(f"File {filename} not found, and no .dat files in the folder.")
            sys.exit(1)
        else:
            filename = files[0]
            print(f"File not found,\n using {filename} instead.")
    data = np.loadtxt(filename,skiprows=2)
    #Read headers to get parameters
    with open(filename,'r') as f:
        _ = f.readline().strip()
        params = f.readline().strip().split()
    #Remove first hash entry
    params = [p for p in params if p[0] != '#']
    tau = float(params[0])
    mu = float(params[1])
    kappa = float(params[2])
    Lx = float(params[3])
    Ly = float(params[4])
    mx = int(params[5])
    my = int(params[6])
    print(f"Parameters read from file:\n tau={tau}, mu={mu}, kappa={kappa}, Lx={Lx}, Ly={Ly}, mx={mx}, my={my}")
    x = np.unique(data[:,0])
    y = np.unique(data[:,1])
    X,Y = np.meshgrid(x,y,indexing='ij')
    q0 = np.reshape(data[:,2],(mx,my))
    q1 = np.zeros((2,mx,my))
    q1[0,:,:] = np.reshape(data[:,3],(mx,my))
    q1[1,:,:] = np.reshape(data[:,4],(mx,my))
    return (tau,mu,kappa,Lx,Ly,mx,my,X,Y,q0,q1)

def plot(filename):
    tau,mu,kappa,Lx,Ly,mx,my,X,Y,q0,q1 = read(filename)
    q1x = q1[0,:,:]
    q1y = q1[1,:,:]

    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.pcolormesh(X,Y,q0,cmap='jet')
    plt.colorbar()
    plt.title('q0 (ground state)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.subplot(1,3,2)
    plt.pcolormesh(X,Y,q1x,cmap='jet')
    plt.colorbar()
    plt.title('q1_x')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1,3,3)
    plt.pcolormesh(X,Y,q1y,cmap='jet')
    plt.colorbar()
    plt.title('q1_y')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.suptitle(f"Ground state solution of NLSH in 2D\nParameters: tau={tau}, mu={mu}, kappa={kappa}, Lx={Lx}, Ly={Ly}, mx={mx}, my={my}")

    plt.tight_layout()
    plt.show()


def save_to_fortran(filename_in,filename_out=None):
    if filename_out is None:
        filename_out = filename_in.replace('.dat','_fortran.bin')
    _,_,_,_,_,mx,my,X,Y,q0,q1 = read(filename_in)
    q1x = q1[0,:,:]
    q1y = q1[1,:,:]
    x = X[:,0]
    y = Y[0,:]
    q0fortran = np.asfortranarray(q0)
    q1xfortran = np.asfortranarray(q1x)
    q1yfortran = np.asfortranarray(q1y)
    with open(filename_out, "wb") as f:
        np.array([mx, my], dtype=np.int32).tofile(f) 
        x.astype('float64').tofile(f)
        y.astype('float64').tofile(f)
        q0fortran.astype('float64').tofile(f)
        q1xfortran.astype('float64').tofile(f)
        q1yfortran.astype('float64').tofile(f)
    print(f"Data saved to Fortran binary file {filename_out}")

if __name__ == "__main__":
    run_and_save()
    run_and_save(tau=0.01)
    run_and_save(tau=0.0)
    plot("test.dat")
