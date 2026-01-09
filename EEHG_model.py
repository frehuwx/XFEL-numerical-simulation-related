# EEHG related codes
import numpy as np
import scipy
import matplotlib.pyplot as plt

# constants
hbar=1.05457e-34
e=1.60218e-19
c=2.9979e8
me=9.10938e-31
re=2.81794e-15 # classical electron radius
epsilon0=8.85419e-12
Ia=4*np.pi*epsilon0*me*c**3/e # Alfven current

# laser class
class laser():
    # intialization using wavelength, pulse energy, pulse FWHM, and beam waist size
    def __init__(self,wavelen,pulseE,pulseFWHM,w0):
        # intial parameters
        self.wavelen=wavelen # wavelength in m
        self.energy=pulseE # pulse energy in J
        self.FWHM=pulseFWHM # FWHM of power profile in s
        self.w0=w0 # w0 in m
        # deduced parameters
        self.zr=np.pi*self.w0*self.w0/self.wavelen # rayleigh range
        sig_t=pulseFWHM/2/np.sqrt(2*np.log(2))
        self.peakP=pulseE/(np.sqrt(2*np.pi)*sig_t) # peak power in W
        peakI=self.peakP/(np.pi*w0*w0)
        self.peakE=np.sqrt(2*peakI/c/epsilon0) # peak electric field
    # set new pulse energy
    def set_pulseE(self,new_pulseE):
        self.energy=new_pulseE
        # reset other parameters
        sig_t=self.FWHM/2/np.sqrt(2*np.log(2)) # sigma in time for power profile (sqrt(2) smaller than E field sigma)
        self.peakP=self.energy/(np.sqrt(2*np.pi)*sig_t) # peak power in W
        peakI=self.peakP/(np.pi*self.w0*self.w0)
        self.peakE=np.sqrt(2*peakI/c/epsilon0) # peak electric field
        return
    # get the complex laser field: coordinate system (z,r,t), beam waist at z0
    def field(self,z0,z,r,t,phi0):
        k=2*np.pi/self.wavelen
        sig_t=self.FWHM/2/np.sqrt(2*np.log(2))
        w=self.w0*np.sqrt(1+((z-z0)/self.zr)**2)
        gouy=np.arctan((z-z0)/self.zr)
        t_relative=(z-t*c)/c
        if abs(z)>0.01*self.zr:
            R_cur=z*(1+(self.zr/z)**2)
            E=self.peakE*(self.w0/w)*np.exp(-r**2/w**2)*np.sin(-(k*z+k*r*r/2/R_cur-gouy)+k*c*t+phi0)*np.exp(-t_relative**2/4/sig_t**2)
        else:
            E=self.peakE*(self.w0/w)*np.exp(-r**2/w**2)*np.sin(-(k*z-gouy)+k*c*t+phi0)*np.exp(-t_relative**2/4/sig_t**2)
        return E

# chicane class
class chicane():
    # initialization
    def __init__(self,lb=1,ld=1,angle=0):
        self.lb=lb
        self.ld=ld
        self.angle=angle
        self.get_R56()
    
    # R56 (equal double delay)
    def get_R56(self):
        R56=(4*self.lb*self.angle/np.sin(self.angle)+2*self.ld/np.cos(self.angle)-4*self.lb-2*self.ld)*2
        self.R56=R56
        return
    
    # B factor for EEHG
    def get_B(self,E_sig,E_electron,lambda_laser):
        return 2*np.pi/(lambda_laser)*self.R56*E_sig/E_electron
    
    
# modulator class
class Modulator():
    # inialization
    def __init__(self,wavelen,Nmod,Kmod):
        self.wavelen=wavelen
        self.Nmod=Nmod
        self.Kmod=Kmod
        self.k=2*np.pi/wavelen
    # get resonant wavelength, E_electron in MeV
    def resonant_wavelen(self,E_electron):
        gamma=E_electron/0.511
        return self.wavelen/2/gamma**2*(1+self.Kmod**2/2)
    # set K by desinating targeting wavelength, E_electron in MeV, tar_wavelen in m
    def set_Kmod(self,E_electron,tar_wavelen):
        gamma=E_electron/0.511
        self.Kmod=np.sqrt(2*(tar_wavelen/self.wavelen*2*gamma**2-1))
    
# function to get the energy modulation for single electron, E_electron in MeV
def delE_mod(laser,mod,E_electron,Phase_electron,tag_output=True):
    # electron beam
    gamma=E_electron/0.511 # energy/rest energy (in MeV)
    # print laser beam peak power
    if tag_output:
        print('peak power: %.4f MW'%(laser.peakP*1e-6))

    # get electron traj
    t=np.linspace(-mod.Nmod*mod.wavelen/c/2,mod.Nmod*mod.wavelen/c/2,4001) # modulator centers at 0
    beta0=np.sqrt(1-(1+mod.Kmod*mod.Kmod/2)/gamma/gamma)
    Sp=mod.Kmod**2/(8*mod.k*beta0*gamma**2)
    z_electron=beta0*c*t-Sp*np.sin(2*mod.k*beta0*c*t)
    x_electron=mod.Kmod/gamma/mod.k/beta0*np.cos(mod.k*z_electron)
    betax_electron=-mod.Kmod/gamma*np.sin(mod.k*z_electron)
    
    # integrate along the traj to get energy modulation
    E_field_list=[]
    sum_mod=0
    for i in range(len(t)-1):
        z_temp=z_electron[i]
        x_temp=x_electron[i]
        E_field_temp=laser.field(0,z_temp,x_temp,t[i],Phase_electron)
        E_field_list.append(E_field_temp)
        delz=z_electron[i+1]-z_electron[i]
        sum_mod+=betax_electron[i]*E_field_temp*delz
    # plot electron traj & laser beam waist
    if tag_output:
        w=laser.w0*np.sqrt(1+((z_electron-0)/laser.zr)**2)
        plt.figure()
        plt.plot(z_electron,x_electron*1e3,label='electron traj')
        plt.plot(z_electron,w*1e3,'r',label='laser w')
        plt.plot(z_electron,-w*1e3,'r')
        plt.legend(loc=1)
        plt.xlabel('z (m)')
        plt.ylabel('x (mm)')
    return sum_mod

# given a desired energy modulation, infer what laser pulse energy is needed
def laserE4delE(laser,mod,E_electron,tar_delE,laserE_max=3e-3,tag_output=True):
    laserE_list=np.linspace(0,laserE_max,101) # 0 - 3 mJ
    delE_list=[]
    laserE_old=laser.energy
    for i in range(len(laserE_list)):
        laser.set_pulseE(laserE_list[i])
        delE_list.append(delE_mod(laser,mod,E_electron,np.pi,False))
    delE_list=np.array(delE_list)
    laserE_result=np.interp(tar_delE,delE_list,laserE_list)
    if tag_output:
        plt.figure()
        plt.plot(laserE_list*1e3,delE_list/1e6)
        plt.plot(laserE_result*1e3,tar_delE/1e6,'o')
        plt.xlabel('laser pulse energy (mJ)')
        plt.ylabel('energy modulation (MeV)')
    # recover the old laserE
    laser.set_pulseE(laserE_old)
    return laserE_result

# EEHG theo model: frequency component for h harmonics
def EEHG_FreqComp(A1,B1,A2,B2,h):
    return np.abs(scipy.special.jv(h+1,h*A2*B2)*scipy.special.jv(1,A1*(B1-h*B2))*np.exp(-0.5*(B1-h*B2)**2))
# EEHG theo model: frequency component for h harmonics, now adding a chirp factor k = 1/(k1*sigE) * dE/dz 
def EEHG_withchirp_FreqComp(A1,B1,A2,B2,m,k):
    return np.abs(scipy.special.jv(m,(-1+m*(1+k*B1))/(1+k*(B1+B2))*A2*B2)*scipy.special.jv(1,A1*(m*B2-B1-B2)/(1+k*(B1+B2)))*np.exp(-0.5*(m*B2-B2-B1)**2/(1+k*(B1+B2))**2))
# EEHG optimization
def EEHG_opti(A1,B1,h,rangeA=[0.85,1.4],rangeB=[0.95,1.05],tag_output=True):
    B2_guess=B1/h
    A2_guess=1/B2_guess
    A2=np.linspace(A2_guess*rangeA[0],A2_guess*rangeA[1],501)
    B2=np.linspace(B2_guess*rangeB[0],B2_guess*rangeB[1],251)
    Amp=np.zeros((len(A2),len(B2)))
    for idx_A2 in range(len(A2)):
        for idx_B2 in range(len(B2)):
            Amp[idx_A2][idx_B2]=EEHG_FreqComp(A1,B1,A2[idx_A2],B2[idx_B2],h)
    # plot
    if tag_output:
        plt.figure()
        extent=[B2[0],B2[-1],A2[-1],A2[0]]
        plt.imshow(Amp,extent=extent,aspect='auto')
        plt.xlabel('B2')
        plt.ylabel('A2')
        
    # output maximized results
    if tag_output:
        A2_maxidx=np.argmax(Amp[:,0:125],axis=0)
        max_list=[]
        for i in range(125):
            max_list.append(Amp[A2_maxidx[i]][i])
        max_list=np.array(max_list)
        B2_max=B2[np.argmax(max_list)]
        A2_max=A2[A2_maxidx[np.argmax(max_list)]]
        amp_max=np.max(max_list)
        print('Maximized bunching with small B2: A2 = %.5f, B2 = %.5f, bunching = %.3f%%'%(A2_max,B2_max,amp_max*100))
        A2_maxidx=np.argmax(Amp[:,125:-1],axis=0)
        max_list=[]
        for i in range(125):
            max_list.append(Amp[A2_maxidx[i]][i+125])
        max_list=np.array(max_list)
        B2_max=B2[np.argmax(max_list)+125]
        A2_max=A2[A2_maxidx[np.argmax(max_list)]]
        amp_max=np.max(max_list)
        print('Maximized bunching with large B2: A2 = %.5f, B2 = %.5f, bunching = %.3f%%'%(A2_max,B2_max,amp_max*100))
        
    return [A2,B2,Amp]
# EEHG optimization, with chirp
def EEHG_withchirp_opti(A1,B1,m,k,rangeA=[0.85,1.4],rangeB=[0.95,1.05],tag_output=True):
    B2_guess=B1/m
    A2_guess=abs(1/B2_guess)
    A2=np.linspace(A2_guess*rangeA[0],A2_guess*rangeA[1],501)
    B2=np.linspace(B2_guess*rangeB[0],B2_guess*rangeB[1],251)
    Amp=np.zeros((len(A2),len(B2)))
    for idx_A2 in range(len(A2)):
        for idx_B2 in range(len(B2)):
            Amp[idx_A2][idx_B2]=EEHG_withchirp_FreqComp(A1,B1,A2[idx_A2],B2[idx_B2],m,k)
    # plot
    if tag_output:
        plt.figure()
        extent=[B2[0],B2[-1],A2[-1],A2[0]]
        plt.imshow(Amp,extent=extent,aspect='auto')
        plt.xlabel('B2')
        plt.ylabel('A2')
        
    # output maximized results
    if tag_output:
        A2_maxidx=np.argmax(Amp[:,0:125],axis=0)
        max_list=[]
        for i in range(125):
            max_list.append(Amp[A2_maxidx[i]][i])
        max_list=np.array(max_list)
        B2_max=B2[np.argmax(max_list)]
        A2_max=A2[A2_maxidx[np.argmax(max_list)]]
        amp_max=np.max(max_list)
        print('Maximized bunching with small B2: A2 = %.5f, B2 = %.5f, bunching = %.3f%%'%(A2_max,B2_max,amp_max*100))
        h=(m*(1+k*B1)-1)/(1+k*(B1+B2_max))
        print('Corresponding harmonics number is: %.3f'%h)
        A2_maxidx=np.argmax(Amp[:,125:-1],axis=0)
        max_list=[]
        for i in range(125):
            max_list.append(Amp[A2_maxidx[i]][i+125])
        max_list=np.array(max_list)
        B2_max=B2[np.argmax(max_list)+125]
        A2_max=A2[A2_maxidx[np.argmax(max_list)]]
        amp_max=np.max(max_list)
        print('Maximized bunching with large B2: A2 = %.5f, B2 = %.5f, bunching = %.3f%%'%(A2_max,B2_max,amp_max*100))
        h=(m*(1+k*B1)-1)/(1+k*(B1+B2_max))
        print('Corresponding harmonics number is: %.3f'%h)
    return [A2,B2,Amp]

# A function for all parameters
Lambda=8 # scaling factor for IBS
def scatter_effects(chicane,E_electron,E_sig,current,emit,beta,total_length,B1,output=True):
    rho=chicane.lb/chicane.angle
    spacing=np.pi/B1*E_sig*1e3
    gamma=E_electron/0.511
    D_ISR=1/(4*np.pi*epsilon0)*55/(24*np.sqrt(3))*(hbar*e**2*c)/(rho**3)*gamma**7 # in SI unit
    D_IBS=(np.sqrt(np.pi)*Lambda*(me*c**2)**2*re*np.sqrt(gamma)*current)/(2*Ia*emit**1.5*np.sqrt(beta))
    delE_ISR=np.sqrt(0.5*D_ISR*(chicane.lb))/e/1e3 # in keV
    delE_IBS=np.sqrt(0.5*D_IBS*total_length)/e/1e3 # in keV
    if output:
        print('Summary: ')
        print('EEHG band spacing: %.3f keV'%(spacing))
        print('D_ISR=%.3e J^2/m, delE_ISR=%.3f keV'%(D_ISR,delE_ISR))
        print('D_IBS=%.3e J^2/m, delE_IBS=%.3f keV'%(D_IBS,delE_IBS))
    return [spacing,delE_ISR,delE_IBS]