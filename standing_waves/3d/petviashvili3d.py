r"""
This module implements the Petviashvili method to compute ground state solutions
of the NLSH (and NLS) equation in 3D:
\frac{\Delta}{1-\mu\tau} q_0 -\mu  = -\kappa q_0^3,
where \mu>0, \kappa>0, and \tau>0 are parameters which should satisfy 1-\mu\tau>0.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys



def run_and_save(tau=0.9,mu=1.0,kappa=1.0,
                 Lx=20.0,Ly=20.0,Lz=20.,mx=128,my=128,mz=128,
                 niters=30,filename=None,save=True):
    dx = Lx/mx
    dy = Ly/my
    dz = Lz/mz
    x = np.linspace(-Lx/2,Lx/2,mx)
    y = np.linspace(-Ly/2,Ly/2,my)
    z = np.linspace(-Lz/2,Lz/2,mz)
    X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
    kx = 2*np.pi*np.fft.fftfreq(mx,d=dx)
    ky = 2*np.pi*np.fft.fftfreq(my,d=dy)
    kz = 2*np.pi*np.fft.fftfreq(mz,d=dz)
    KX,KY,KZ = np.meshgrid(kx,ky,kz,indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    ddx = 1j*KX
    ddy = 1j*KY
    ddz = 1j*KZ

    #Initial condition
    q0 = np.exp(-(X**2+Y**2+Z**2))
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
        # if it%10==0:
        print(f"Iteration {it}, S={s:.6f}, max(q0)={np.max(q0):.6f}")
    q1 = np.zeros((3,mx,my,mz))
    q1[0,:,:,:] = np.fft.ifftn(ddx*q0hat).real/(1 - mu*tau)
    q1[1,:,:,:] = np.fft.ifftn(ddy*q0hat).real/(1 - mu*tau)
    q1[2,:,:,:] = np.fft.ifftn(ddz*q0hat).real/(1 - mu*tau)

    dump_array = np.zeros((mx*my*mz,7))
    dump_array[:,0] = np.ravel(X)
    dump_array[:,1] = np.ravel(Y)
    dump_array[:,2] = np.ravel(Z)
    dump_array[:,3] = np.ravel(q0)
    dump_array[:,4] = np.ravel(q1[0,:,:,:])
    dump_array[:,5] = np.ravel(q1[1,:,:,:])
    dump_array[:,6] = np.ravel(q1[2,:,:,:])

    if save:
        #Save to file
        if filename is None:
            filename = f"ground_state3D_tau{tau}_mu{mu}_kappa{kappa}_Lx{Lx}_Ly{Ly}_Lz{Lz}_mx{mx}_my{my}_mz{mz}.dat"

        np.savetxt(filename,dump_array,
                    header=f"Parameters: tau, mu, kappa, Lx, Ly, Lz, mx, my, mz\n"+
                    f"{tau} {mu} {kappa} {Lx} {Ly} {Lz} {mx} {my} {mz}\n"+
                    "Columns: x, y, z, q0, q1_x, q1_y, q1_z")
        return None
    else:
        params = {'tau': tau, 'mu': mu, 'kappa': kappa, 'Lx': Lx, 'Ly': Ly, 'Lz': Lz, 'mx': mx, 'my': my, 'mz': mz}
        return params, X, Y, Z, q0, q1
    
def read(filename):
    #Check if file exists, if not, try any .dat file in the folder
    try:
        print(f"Reading data file {filename}")
        with open(filename,'r') as f:
            pass
    except FileNotFoundError:
        import glob
        files = glob.glob("ground_state3D*.dat")
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
    Lz = float(params[5])
    mx = int(params[6])
    my = int(params[7])
    mz = int(params[8])
    print(f"Parameters read from file:\n tau={tau}, mu={mu}, kappa={kappa}, Lx={Lx}, Ly={Ly}, Lz={Lz}, mx={mx}, my={my}, mz={mz}")
    x = np.unique(data[:,0])
    y = np.unique(data[:,1])
    z = np.unique(data[:,2])
    q0 = np.reshape(data[:,3],(mx,my,mz))
    q1 = np.zeros((3,mx,my,mz))
    q1[0,:,:,:] = np.reshape(data[:,4],(mx,my,mz))
    q1[1,:,:,:] = np.reshape(data[:,5],(mx,my,mz))
    q1[2,:,:,:] = np.reshape(data[:,6],(mx,my,mz))
    return (tau,mu,kappa,Lx,Ly,Lz,mx,my,mz,x,y,z,q0,q1)

def plot(filename):
    tau,mu,kappa,Lx,Ly,Lz,mx,my,mz,x,y,z,q0,q1 = read(filename)
    q0 = q0[:,:,mz//2]
    q1x = q1[0,:,:,mz//2]
    q1y = q1[1,:,:,mz//2]
    q1z = q1[2,mx//2,:,:]

    X,Y = np.meshgrid(x,y,indexing='ij')

    plt.figure(figsize=(15,5))
    plt.subplot(2,2,1)
    plt.pcolormesh(X,Y,q0,cmap='jet')
    plt.colorbar()
    plt.title('q0 (ground state) across z=0')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(2,2,2)
    plt.pcolormesh(X,Y,q1x,cmap='jet')
    plt.colorbar()
    plt.title('q1_x across z=0')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(2,2,3)
    plt.pcolormesh(X,Y,q1y,cmap='jet')
    plt.colorbar()
    plt.title('q1_y across z=0')
    plt.xlabel('x')
    plt.ylabel('y')

    Y,Z = np.meshgrid(y,z,indexing='ij')

    plt.subplot(2,2,4)
    plt.pcolormesh(Y,Z,q1z,cmap='jet')
    plt.colorbar()
    plt.title('q1_z across x=0')
    plt.xlabel('y')
    plt.ylabel('z')



    plt.suptitle(f"Ground state solution of NLSH in 3D\nParameters:" +
                  f"tau={tau}, mu={mu}, kappa={kappa}, " +
                  f"Lx={Lx}, Ly={Ly}, Lz={Lz}, mx={mx}, my={my}, mz={mz}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_and_save()
    plot("test.dat")
