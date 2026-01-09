# dynamic diffraction

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize
import scipy.signal as signal

# constants
hbar_eV = 6.582119569e-16
h_planck = 6.6260755e-34  # Planck's constant. J.s of m^2.kg/s
c_mps = 2.99792458e8
re_nm = 2.81794e-6
e=1.60217663e-19

# scatter factor nine fit parameters, defined in:
# International Tables for X-ray Crystallography, vol.C , page 565
# a1 b1 a2 b2 a3 b3 a4 b4 c
Scatter_Factor_C=np.array([2.31000,20.8439,1.02000,10.2075,1.58860,0.568700,0.865000,51.6512,0.2156])
Scatter_Factor_Cbound=np.array([2.26069,22.6907,1.56165,0.656665,1.05075,9.75618,0.839259,55.5949,0.286977])
# a helper function for scatter factor
def get_f0(x, fit_para=Scatter_Factor_C):
    return fit_para[0]*np.exp(-fit_para[1]*x**2)+fit_para[2]*np.exp(-fit_para[3]*x**2)+fit_para[4]*np.exp(-fit_para[5]*x**2)+fit_para[6]*np.exp(-fit_para[7]*x**2)+fit_para[8]

# for f1 and f2, use (1 of 2):
# Chantler's calculation: https://physics.nist.gov/PhysRefData/FFast/html/form.html
# Henke's calculation: henke.lbl.gov/optical_constants/asf.html

# x,y,z unit vectors
xhat = np.array((1,0,0))
yhat = np.array((0,1,0))
zhat = np.array((0,0,1))

# a helper function to get the <u2> using the debye formalism
def adps(m, thetaD, T):
    """Calculates atomic displacement factors within the Debye model.

    <u^2> = (3h^2/4 pi^2 m kB thetaD)(phi(thetaD/T)/(ThetaD/T) + 1/4)

    arguments:
    m -- float -- mass of the ion in atomic mass units (e.g., C = 12)
    thetaD -- float -- Debye Temperature
    T -- float -- temperature.

    return:
    Uiso -- float -- the thermal factor from the Debye recipe at temp T
    """
    h = 6.6260755e-34  # Planck's constant. J.s of m^2.kg/s
    kB = 1.3806503e-23  # Boltzmann's constant. J/K
    amu = 1.66053886e-27  # Atomic mass unit. kg

    def __phi(x):
        """Evaluates the phi integral needed in Debye calculation.

        phi(x) = (1/x) int_0^x xi/(exp(xi)-1) dxi

        arguments:
        x -- float -- value of thetaD (Debye temperature)/T

        returns:
        phi -- float -- value of the phi function
        """

        def __debyeKernel(xi):
            """Function needed by debye calculators."""
            y = xi / (np.exp(xi) - 1)
            return y

        int = scipy.integrate.quad(__debyeKernel, 0, x)
        phi = (1 / x) * int[0]

        return phi

    m = m * amu
    u2 = (3 * h**2 / (4 * np.pi**2 * m * kB * thetaD)) * (
        __phi(thetaD / T) / (thetaD / T) + 1.0 / 4.0
    )

    return u2 # in m^2

# a helper function to rotate vectors w.r.t. an axis
def rotate(vector, axis, angle): 
    axis = axis/np.sqrt(axis @ axis) #normalize
    r = Rot.from_rotvec(axis*angle)
    return r.apply(vector)

# a helper function to get the vector in real space based on [H,K,L]
def get_real_vec(a_vec,b_vec,c_vec,miller_indices,normalize=True):
    indice_inv=[]
    for i in range(3):
        if np.abs(miller_indices[i])<1:
            indice_inv.append(0)
        else:
            indice_inv.append(1/miller_indices[i])
    real_vec=a_vec*indice_inv[0]+b_vec*indice_inv[1]+c_vec*indice_inv[2]
    if normalize:
        real_vec=real_vec/np.sqrt(np.sum(real_vec**2))
    return real_vec

# a helper function to get the angle beteween two vectors
def get_vec_angle(a_vec,b_vec):
    a_norm=a_vec/np.linalg.norm(a_vec)
    b_norm=b_vec/np.linalg.norm(b_vec)
    return np.arccos(np.dot(a_norm,b_norm))

# a helper function to do the FFT:
def FFT_full(xdata,ydata,padding=0,Hanns_window=False,cali=True):
    lenx=len(xdata)
    delx=xdata[1]-xdata[0]
    # windowing
    if Hanns_window:
        hann_func=signal.windows.hann(len(xdata))
        ydata=ydata*hann_func
        
    # padding
    if padding>1:
        xdata_tot=np.linspace(xdata[0]-int(lenx*padding)*delx,xdata[-1]+int(lenx*padding)*delx,int(lenx*(2*padding+1)))
        ydata_zeros=np.zeros(int(padding*lenx),dtype='complex')
        ydata_tot=np.concatenate((ydata_zeros,np.concatenate((ydata,ydata_zeros))))
    else:
        xdata_tot=xdata
        ydata_tot=ydata
    
    # FFT    
    len_tot=len(xdata_tot)
    if len_tot%2==0:
        fdata=1/delx/len_tot*np.linspace(-len_tot/2,len_tot/2-1,len_tot)
    else:
        fdata=1/delx/len_tot*np.linspace(-len_tot/2,len_tot/2,len_tot)
    Adata=np.fft.fftshift(np.fft.fft(np.fft.ifftshift(ydata_tot)))
    
    # calibration (based on power!)
    if cali:
        delf=fdata[1]-fdata[0]
        pulseE=np.sum(np.abs(ydata)**2)*delx
        pulseE_output=np.sum(np.abs(Adata)**2)*delf
        cali_factor=np.sqrt(pulseE/pulseE_output)
        Adata=Adata*cali_factor
    return [fdata,Adata]


# single crystal lattice
class single_crystal():
    
    # initialize the crystal using a,b,c (in Angstrom) and alpha,beta,gamma (in deg), and Debye temperature (in K)
    def __init__(self,a=3.5668,b=3.5668,c=3.5668,alpha=90,beta=90,gamma=90,angletype='deg',crystaltype='diamond'):
        # save the crystal geometry
        # change unit of length from A to nm, change unit of angle to rad
        self.a=a/10
        self.b=b/10
        self.c=c/10
        if angletype=='deg':
            self.alpha=alpha/180*np.pi
            self.beta=beta/180*np.pi
            self.gamma=gamma/180*np.pi
        else:
            self.alpha=alpha
            self.beta=beta
            self.gamma=gamma
        # save crystal type
        self.crystaltype=crystaltype
        if crystaltype=='diamond':
            self.atomtype='C'
            self.atommass=12
            self.atomnumber=6
            self.TD=1510
        else:
            print('Warning: the queried crystal is not supported!')
            print('Assuming it is a diamond...')
            self.atomtype='C'
            self.atommass=12
            self.atomnumber=6
            self.TD=1515
        self.u2=adps(self.atommass,self.TD,293.15)*1e18 # assuming Room temperature (293.15 K), in nm^2
        # save unit cell volume
        self.cell_vol=self.a*self.b*self.c*np.sqrt(1-np.cos(self.alpha)**2-np.cos(self.beta)**2-np.cos(self.gamma)**2+2*np.cos(self.alpha)*np.cos(self.beta)*np.cos(self.gamma))
        # set vetors
        self.set_realspace_vet()
        self.set_recspace_vet()
        # save the fit parameters for f0
        if self.atomtype=='C':
            self.f0_fitpara=Scatter_Factor_C
        else:
            print('Warning: the fit parameters for f0 are not saved for atom '+atomtype+', will assume it is carbon.')
            self.f0_fitpara=Scatter_Factor_C
        
    # set the a,b,c vectors in real space
    def set_realspace_vet(self):
        a_vec=self.a*xhat
        b_vec=self.b*np.array([np.cos(self.gamma),np.sin(self.gamma),0])
        c1=np.cos(self.beta)
        c2=(np.cos(self.alpha)-np.cos(self.beta)*np.cos(self.gamma))/np.sin(self.gamma)
        c3=np.sqrt(1-c1**2-c2**2)
        c_vec=self.c*np.array([c1,c2,c3])
        self.a_vec=np.array(a_vec)
        self.b_vec=np.array(b_vec)
        self.c_vec=np.array(c_vec)
        return
    
    # set the ka,kb,kc vectors in reciprocal space
    def set_recspace_vet(self):
        denom = self.a_vec @ np.cross(self.b_vec, self.c_vec)
        self.a_rec = 2*np.pi*np.cross(self.b_vec, self.c_vec)/denom
        self.b_rec = 2*np.pi*np.cross(self.c_vec, self.a_vec)/denom
        self.c_rec = 2*np.pi*np.cross(self.a_vec, self.b_vec)/denom    
        return


class scatter_geometry():
    
    # initialize the geometry using surface norm (H,K,L) and the scatter plane (H,K,L), the xray energy (in eV), the azimuthal angle phi (in deg), the crystal thickness (in m)
    def __init__(self,crystal,surface_norm=[4,0,0],surface_x=[0,1,0],scatter_norm=[4,0,0],Exray=9.83e3,d=0.1e-3,geotype='auto',phi=0,modeltype='Henke',tag_output=True):
        # 1. save the inputs
        self.crystal=crystal
        self.surface_norm=np.array(surface_norm)
        self.scatter_norm=np.array(scatter_norm)
        self.Exray=Exray
        self.K0=Exray/hbar_eV/c_mps # in 1/m
        self.d=d
        # 2. save the scatter amplitude
        atomtype=self.crystal.atomtype
        try:
            [energy, f1, f2]=np.loadtxt('formfactor_'+modeltype+'_'+atomtype+'.txt',skiprows=1,unpack=True)
            if modeltype=='Chantler':
                energy=energy*1e3 # original file is in keV
            self.f1=np.interp(Exray,energy,f1)
            self.f2=np.interp(Exray,energy,f2)
            if tag_output:
                print('The atom is '+atomtype+', interpolated structure factor (f1, f2): %.3f, %.3f'%(self.f1,self.f2))
        except:
            if tag_output:
                print('Warning: cannot find the file for atom'+atomtype+', will assume it is carbon.')
            [energy, f1, f2]=np.loadtxt('formfactor_Henke_C.txt',skiprows=1,unpack=True)
            self.f1=np.interp(Exray,energy,f1)
            self.f2=np.interp(Exray,energy,f2)
            if tag_output:
                print('Interpolated structure factor (f1, f2): %.3f, %.3f'%(self.f1,self.f2))
        # 3. convert to vectors
        self.surfacenorm_vec=-get_real_vec(self.crystal.a_vec,self.crystal.b_vec,self.crystal.c_vec,surface_norm)
        self.surfacex_vec=get_real_vec(self.crystal.a_vec,self.crystal.b_vec,self.crystal.c_vec,surface_x)
        surfacey_vec=np.cross(self.surfacenorm_vec,self.surfacex_vec)
        self.surfacey_vec=surfacey_vec/np.linalg.norm(surfacey_vec)
        self.H_vec=get_real_vec(self.crystal.a_vec,self.crystal.b_vec,self.crystal.c_vec,scatter_norm)
        self.H_rec=self.crystal.a_rec*self.scatter_norm[0]+self.crystal.b_rec*self.scatter_norm[1]+self.crystal.c_rec*self.scatter_norm[2]
        self.d_H=2*np.pi/np.sqrt(np.sum(self.H_rec**2)) # scatter plane spacing
        # 4. calculate the angles
        # the angle between scatter plane and surface normal (eta)
        self.eta=get_vec_angle(-self.surfacenorm_vec,self.H_vec)
        # Glancing angle (Bragg angle) (theta)
        self.theta=np.arcsin(np.sqrt(np.sum(self.H_rec**2))*1e9/2/self.K0) # the 1e9 factor is used to change th units from 1/nm to 1/m
        # the angle between (K0, H) plane and the x direction defined by the lattice
        self.phi=phi/180*np.pi
        # 5. get K0 and KH based on the geometric setup
        # in the auto mode, we assume we rotate the crystal using +surface_x and +surface_y anticlockwise
        # NOTE 1: here we need to rotate clockwise instead
        # NOTE 2: we need to (1) rotate y (psi) to make sure polarization; (2) rotate x to get the correct angle
        # determine the rotation angle (along y)
        if geotype=='auto':
            if tag_output:
                print('Specified auto geometry, will calculate the geometry based on crystal geometry, assuming light polarization is along surface_x.')
            angle_xH=np.pi/2-get_vec_angle(self.surfacex_vec,self.H_vec)
            K0_temp=rotate(self.surfacenorm_vec,self.surfacey_vec,-angle_xH)
            # determine the rotation angle (along x)
            rot_theta_now=get_vec_angle(-K0_temp,self.H_vec)
            rot_theta_now=2*np.arcsin(np.sin(rot_theta_now/2)/np.cos(angle_xH))
            rot_theta_target=2*np.arcsin(np.sin((np.pi/2-self.theta)/2)/np.cos(angle_xH))
            self.rot_psi=angle_xH
            if np.dot(self.H_vec,self.surfacey_vec)<0:
                self.rot_theta=-(rot_theta_target-rot_theta_now)
            else:
                self.rot_theta=+(rot_theta_target-rot_theta_now)
            if tag_output:
                print('Rotation angles: psi = %.2f deg, theta = %.2f deg'%(self.rot_psi/np.pi*180,self.rot_theta/np.pi*180))
            # we now get the result
            K0_vec=rotate(self.surfacenorm_vec,self.surfacey_vec,-self.rot_psi)
            K0_vec=rotate(K0_vec,self.surfacex_vec,-self.rot_theta)
            KH_vec=-rotate(K0_vec,self.H_vec,np.pi)
            # add additional rotation phi around H
            self.K0_vec=rotate(K0_vec,self.H_vec,self.phi)
            self.KH_vec=rotate(KH_vec,self.H_vec,self.phi)
        elif geotype=='on-plane':
            if tag_output:
                print('Specified on-plane geometry, will calculate the geometry such that surface norm, H, and K0 are on the same plane.')
            rot_axis=np.cross(self.H_vec,self.surface_norm)
            rot_axis=rot_axis/np.linalg.norm(rot_axis)
            K0_vec=rotate(-self.H_vec,rot_axis,np.pi/2-self.theta)
            KH_vec=-rotate(K0_vec,self.H_vec,np.pi)
            self.K0_vec=K0_vec
            self.KH_vec=KH_vec
            # invert the directions to make sure |b|>1
            if np.abs(np.sum(self.K0_vec*self.surfacenorm_vec))<np.abs(np.sum(self.KH_vec*self.surfacenorm_vec)):
                self.K0_vec=-self.K0_vec
                self.K0_vec=-self.KH_vec
        else:
            print('Warning: cannot find the requested geometry mode! (auto or on-plane)')
            
        # check:
        angle1=get_vec_angle(KH_vec,self.H_vec)
        angle2=np.pi/2-self.theta
        if tag_output:
            print('angle: result %.2f tar %.2f'%(angle1/np.pi*180,angle2/np.pi*180))
        
        # get also the direction vector for polarization
        self.sigma_polarization=np.cross(self.K0_vec,self.KH_vec)
        self.pi_polarization=np.cross(self.H_vec,self.sigma_polarization)
        self.sigma_polarization=self.sigma_polarization/np.linalg.norm(self.sigma_polarization)
        self.pi_polarization=self.pi_polarization/np.linalg.norm(self.pi_polarization)
        # 6. calculate the chi factors
        self.calc_chi_factor()
        
    # calculate the chi factors
    def calc_chi_factor(self):
        # chi0
        f0=get_f0(0,self.crystal.f0_fitpara)
        F0=8*(f0+self.f1-self.crystal.atomnumber+1j*self.f2)
        chi0=-(re_nm*1e-9)*F0/np.pi/(self.crystal.cell_vol*1e-27)*(h_planck*c_mps)**2/(self.Exray*e)**2
        # the imag part of chi0 should be always positive, due to the definition exp(i(k*r-omega*t))
        if np.imag(chi0)<0:
            chi0=np.conjugate(chi0)
        
        # chiH
        # get f0
        lambda_angstrom=h_planck*c_mps/(self.Exray*e)*1e10
        x=np.sin(self.theta)/lambda_angstrom
        f0=get_f0(x,self.crystal.f0_fitpara)
        # get geo_SF
        self.calc_geo_SF()
        # get W
        W=2*np.pi**2*self.crystal.u2/self.d_H**2 # which is equivalent to 8*np.pi**2*self.crystal.u2*np.sin(self.theta)**2/(lambda_angstrom*0.1)**2
        self.W=W
        FH=(f0+self.f1-self.crystal.atomnumber+1j*self.f2)*np.exp(-W)*np.abs(self.geo_SF)
        chiH=-(re_nm*1e-9)*FH/np.pi/(self.crystal.cell_vol*1e-27)*(h_planck*c_mps)**2/(self.Exray*e)**2
        chiHbar=chiH
        
        if np.imag(chiH)<0:
            chiH=np.conjugate(chiH)
        if np.imag(chiHbar)<0:
            chiHbar=np.conjugate(chiHbar)

        self.chi0=chi0
        self.chiH=chiH
        self.chiHbar=chiHbar
        self.F0=F0
        self.FH=FH
        return
    
    # calculate the geometrical structure factors
    def calc_geo_SF(self):
        if self.crystal.crystaltype=='diamond':
            atom1_vec=0
            atom2_vec=[self.crystal.a_vec/2+self.crystal.b_vec/2,self.crystal.b_vec/2+self.crystal.c_vec/2,self.crystal.a_vec/2+self.crystal.c_vec/2]
            atom3_vec=[self.crystal.a_vec/4+self.crystal.b_vec/4+self.crystal.c_vec/4,
                       self.crystal.a_vec*3/4+self.crystal.b_vec*3/4+self.crystal.c_vec/4,
                       self.crystal.a_vec*3/4+self.crystal.b_vec/4+self.crystal.c_vec*3/4,
                       self.crystal.a_vec/4+self.crystal.b_vec*3/4+self.crystal.c_vec*3/4]
            geo_SF=np.exp(1j*np.sum(self.H_rec*atom1_vec))*1
            for i in range(len(atom2_vec)):
                geo_SF+=np.exp(1j*np.sum(self.H_rec*atom2_vec[i]))
            for i in range(len(atom3_vec)):
                geo_SF+=np.exp(1j*np.sum(self.H_rec*atom3_vec[i]))
            self.geo_SF=geo_SF
        else:
            print('Warning: the queried crystal is not supported!')
            print('Assuming it is a diamond...')
            atom1_vec=0
            atom2_vec=[self.crystal.a_vec/2+self.crystal.b_vec/2,self.crystal.b_vec/2+self.crystal.c_vec/2,self.crystal.a_vec/2+self.crystal.c_vec/2]
            atom3_vec=[self.crystal.a_vec/4+self.crystal.b_vec/4+self.crystal.c_vec/4,
                       self.crystal.a_vec*3/4+self.crystal.b_vec*3/4+self.crystal.c_vec/4,
                       self.crystal.a_vec*3/4+self.crystal.b_vec/4+self.crystal.c_vec*3/4,
                       self.crystal.a_vec/4+self.crystal.b_vec*3/4+self.crystal.c_vec*3/4]
            geo_SF=np.exp(1j*np.sum(self.H_rec*atom1_vec))*1
            for i in range(len(atom2_vec)):
                geo_SF+=np.exp(1j*np.sum(self.H_rec*atom2_vec[i]))
            for i in range(len(atom3_vec)):
                geo_SF+=np.exp(1j*np.sum(self.H_rec*atom3_vec[i]))
            self.geo_SF=geo_SF
        if np.abs(geo_SF)<1e-6:
            print('Warning: the peak seems to be a forbidden one.')
        return
            
    
    # set the chi factors manually
    def set_chi_factor(self,chi0,chiH,chiHbar):
        self.chi0=chi0
        self.chiH=chiH
        self.chiHbar=chiHbar
        return
    
    # get the info
    def get_info(self):
        print('Here is a brief summary of the geometry:')
        print('surface norm: [%d,%d,%d], scatter plane: [%d,%d,%d], eta: %.2f deg'%(self.surface_norm[0],self.surface_norm[1],self.surface_norm[2],self.scatter_norm[0],self.scatter_norm[1],self.scatter_norm[2],self.eta/np.pi*180))
        print('scatter plane spacing: %.5f A'%(self.d_H*10))
        print('X-ray energy: %.1f eV, bragg angle (theta): %.2f deg'%(self.Exray,self.theta/np.pi*180))
        print('Here is a brief summary of chi:')
        print('         real        imag')
        print('chi0:    %+.4e %+.4e'%(np.real(self.chi0),np.imag(self.chi0)))
        print('chiH:    %+.4e %+.4e'%(np.real(self.chiH),np.imag(self.chiH)))
        print('chiHbar: %+.4e %+.4e'%(np.real(self.chiHbar),np.imag(self.chiHbar)))

        return
    
    # plot the scatter geometry in real space
    def plot_geo(self,figsize=(6,6)):
        fig=plt.figure(figsize=figsize)
        ax=fig.add_subplot(projection='3d')
        ax.quiver(0, 0, 0, *self.crystal.a_vec, color = [0,0.5,0.5], length=0.3, linewidth=3, normalize = True, label = '$a$')
        ax.quiver(0, 0, 0, *self.crystal.b_vec, color = [0.5,0,0.5], length=0.3, linewidth=3, normalize = True, label = '$b$')
        ax.quiver(0, 0, 0, *self.crystal.c_vec, color = [0.5,0.5,0], length=0.3, linewidth=3, normalize = True, label = '$c$')
        
        ax.quiver(0, 0, 0, *self.surfacenorm_vec, linestyle='--', color = "red", length=0.7, label = r'$\hat{z}$')
        ax.quiver(0, 0, 0, *self.surfacex_vec, linestyle='--', color = "brown", length=0.7, label = r'$\hat{x}$ (theta rot axis)')
        ax.quiver(0, 0, 0, *self.surfacey_vec, linestyle='--', color = "purple", length=0.7, label = r'$\hat{y}$ (psi rot axis)')
        ax.quiver(0, 0, 0, *-self.surfacenorm_vec, color = "red", length=0.5, label = "surface: [%d %d %d]"%(self.surface_norm[0],self.surface_norm[1],self.surface_norm[2]))
        ax.quiver(0, 0, 0, *self.H_vec, color = "blue",length=1, label = "scatter: [%d %d %d]"%(self.scatter_norm[0],self.scatter_norm[1],self.scatter_norm[2]))
        ax.quiver(-self.K0_vec[0], -self.K0_vec[1], -self.K0_vec[2], *self.K0_vec, color = "green",length=1, label = "K0")
        ax.quiver(0, 0, 0, *self.KH_vec, linestyle='--', color = "green", length=1, label = "KH")
        ax.legend()
        ax.set_xlim([-1.5,1.5])
        ax.set_ylim([-1.5,1.5])
        ax.set_zlim([-1.5,1.5])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        return
        
        
    # calculate the dynamic diffraction transmission and refraction in Bragg geometry
    def calc_Bragg_diff(self,delta,polarization='sigma',debug=False):
        # 1. get all the parameters
        # geometry factors: gamma0 gammaH and b
        gamma0=np.sum(self.K0_vec*self.surfacenorm_vec)
        gammaH=np.sum(self.KH_vec*self.surfacenorm_vec)
        b=gamma0/gammaH
        self.gamma0=gamma0
        self.gammaH=gammaH
        self.b=b
        # polarizarion
        if polarization=='pi':
            Polar=np.cos(2*self.theta)
        else:
            Polar=1
        # calibrate delta (in eV) to alpha
        alpha=-4*np.sin(self.theta)**2*delta/self.Exray
        # further get alpha_tilde
        alpha_tilde=0.5*(alpha*b+self.chi0*(1-b))
        # calculate x1, x2, R1, R2
        x1=(self.chi0-alpha_tilde+np.sqrt(alpha_tilde**2+b*self.chiH*self.chiHbar*Polar*Polar))*self.K0/2/gamma0
        x2=(self.chi0-alpha_tilde-np.sqrt(alpha_tilde**2+b*self.chiH*self.chiHbar*Polar*Polar))*self.K0/2/gamma0
        R1=(x1*2*gamma0/self.K0-self.chi0)/Polar/self.chiHbar
        R2=(x2*2*gamma0/self.K0-self.chi0)/Polar/self.chiHbar
        # get t00 and R0H
        t00=np.exp(1j*x1*self.d)*(R2-R1)/(R2-R1*np.exp(1j*(x1-x2)*self.d))
        r0H=R1*R2*(1-np.exp(1j*(x1-x2)*self.d))/(R2-R1*np.exp(1j*(x1-x2)*self.d))
        # get extinction length (symmetric)
        self.extinct_len=np.sin(self.theta)/self.K0/np.abs(Polar)/np.sqrt(np.abs(self.chiH*self.chiHbar))
        # get T0 defined in PRAB 16, 120701 (2013)
        self.T0=2*(self.extinct_len)**2/2.998e8/(self.d/self.gamma0) 
        # outputs
        if debug:
            print('Here is a summarization of parameters:')
            print('delta=%.2f eV, alpha=%.2e+%.2e i, alpha_tilde=%.2e+%.2e i'%(delta,np.real(alpha),np.imag(alpha),np.real(alpha_tilde),np.imag(alpha_tilde)))
            print('gamma0=%.5f, gammaH=%.5f, b=%.2f'%(gamma0,gammaH,b))
            print('polarization='+polarization+', P=%.2f'%(Polar))
            print('Here is a summarization of numbers:')
            print('x1=%.2e+%.2e i'%(np.real(x1),np.imag(x1)))
            print('x2=%.2e+%.2e i'%(np.real(x2),np.imag(x2)))
            print('x1-x2=%.2e + %.2e i'%(np.real(x1-x2),np.imag(x1-x2)))
            print('R1=%.2e+%.2e i'%(np.real(R1),np.imag(R1)))
            print('R2=%.2e+%.2e i'%(np.real(R2),np.imag(R2)))
            print('Here is a summarization of results:')
            print('|t00|^2=%.2f, imag t00=%.2f'%(np.abs(t00)**2,np.imag(t00)))
            print('|r0H|^2=%.2f, imag r0H=%.2f'%(np.abs(r0H)**2,np.imag(r0H)))
            print('extinction length (symmetric)=%.3f um and %.3f um (*2)'%(self.extinct_len*1e6,self.extinct_len*1e6*2))
            print('T0 (defined in Yang-Shvydko paper): %.3f fs'%(self.T0*1e15))
        return t00,r0H
    


