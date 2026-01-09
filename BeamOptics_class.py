# matrix based beam optics processing

###########################################################################################################################
# optics class using 6D coordinate x,x',y,y',z,deltaz' + 1D of constant for correctors
# __init__ : creates an empty instance with identical matrix
# __add__ : add two optics, a+b+c means a beamline a->b->c
# append: append an optics to the beamline
# info: print out the data
# evolve: give a list of 6D coordinates, evolve them with the beamline
# MatTran(dim='x'/'y'/'z'): give the transport matrix assuming decoupled dimensions
# MatTwiss(dim='x'/'y'/'z'): give the corresponding Twiss evolve matrix
# periodTwiss(dim='x'/'y'/'z'): give the solution to MatTwiss*Twiss_in=Twiss_out=Twiss_in
# getTraj(Twiss_ini,emit,dim='x'): have the initial Twiss para and geometric emittance, get the [z, beamsize] list
###########################################################################################################################
# Supported optics components:
# drift(name='drift',L=0) a drift with L
# quad(name='quad',L=1,K=1) a quadrapole with length L and strength K/k1, K>0 focusing, K<0 defocusing
# kicker(name='kicker',L=0,cx=0,cy=0) a kicker, length L, bending angle is cx / cy
###########################################################################################################################
# functions:
# getTwiss(x,xp) : statistically get the Twiss parameter using a list of particle coordinates
# getMismatch(x,xp,Twiss0) : statistically get the mismatch factor M=(gamma*beta0-2*alpha*alpha0+beta*gamma0)/2
# slice_emittance(x,xp,t,n) : statistically get the slice emittance for particles with (x,xp,t), assuming n slices
# slice_Twiss(x,xp,t,n) : statistically get the sliced Twiss parmeters for particles with (x,xp,t), assuming n slices
# slice_mismatch(x,xp,t,n,Twiss0) : statistically get the sliced mismatch factor M=(gamma*beta0-2*alpha*alpha0+beta*gamma0)/2
# MatchTwiss(Twiss_in,Twiss_out,del_phi): use the input and output Twiss para, and phase advance to build a transfer mat
###########################################################################################################################


# imports
import sys
import h5py as hdf5
import matplotlib.pyplot as plt
import numpy as np

###########################################################################################################################
# optics class using 6D coordinate x,x',y,y',t,deltaz' + 1D of constant
# __init__ : creates an empty instance with identical matrix
# clear : clear everything
# __add__ : add two optics, a+b+c means a beamline a->b->c
# append : append an optics to the beamline
# info : print out the data
# evolve : give a list of 6D coordinates, evolve them with the beamline
# MatTran(dim='x'/'y'/'z') : give the transport matrix assuming decoupled dimensions
# MatTwiss(dim='x'/'y'/'z') : give the corresponding Twiss evolve matrix
# periodTwiss(dim='x'/'y'/'z') : give the solution to MatTwiss*Twiss_in=Twiss_out=Twiss_in
# getTraj(Twiss_ini,emit,dim='x'/'y') : have the initial Twiss para and geometric emittance, get the [z, beamsize] list
# getBeamTraj(coor_ini_list,dim='x'/'y') : have a few initial particles, get the [z, x] list
###########################################################################################################################
class optics():
    def __init__(self,TotalMatrix=np.identity(6)):
        self.namelist=[] # namelist of the optics
        self.parameterlist=[] # parameters
        self.matrixlist=[] # list of matrixs
        self.transferMatrix=TotalMatrix # total transfer matrix
    def clear(self):
        self.namelist=[]
        self.parameterlist=[]
        self.matrixlist=[]
        self.transferMatrix=np.identity(7)
    def __add__(self,optics2add):
        result=optics()
        result.namelist=np.concatenate((self.namelist,optics2add.namelist))
        result.parameterlist=np.concatenate((self.parameterlist,optics2add.parameterlist))
        result.matrixlist=np.concatenate((self.matrixlist,optics2add.matrixlist))
        result.transferMatrix=np.matmul(optics2add.transferMatrix,self.transferMatrix)
        return result
    def append(self,optics2append): # append the optics
        self.namelist.extend(optics2append.namelist)
        self.parameterlist.extend(optics2append.parameterlist)
        self.matrixlist.extend(optics2append.matrixlist)
        self.transferMatrix=np.matmul(optics2append.transferMatrix,self.transferMatrix)
    def info(self): # get the parameter info
        print('the optics:')
        for i in range(len(self.namelist)):
            print(i+1, end=': ')
            print('name: '+self.namelist[i], end=' ')
            print('parameters:', end=' ')
            paratemp=self.parameterlist[i]
            for keyName in paratemp.keys():
                print(keyName+':', end=' ')
                print(paratemp[keyName], end=', ')
            print('')
    def evolve(self,list_particle): # evolve the particles (6D coordinate)
        list_particle=np.array(list_particle)
        particle_num=len(list_particle)
        const_dim=np.ones((particle_num,1))
        list_particle=np.concatenate((list_particle,const_dim),axis=1)
        list_particle_evolve=np.zeros((particle_num,6))
        for i in range(len(list_particle)):
            particle_temp=list_particle[i]
            particle_temp_evolve=np.matmul(self.transferMatrix,particle_temp)
            list_particle_evolve[i]=particle_temp_evolve[0:6]
        return list_particle_evolve
    def MatTrans(self,dim='x'): # get the transport matrix
        # extract transfer Matrix
        if dim=='x':
            M_transport=[[self.transferMatrix[0][0],self.transferMatrix[0][1]],[self.transferMatrix[1][0],self.transferMatrix[1][1]]]
        elif dim=='y':
            M_transport=[[self.transferMatrix[2][2],self.transferMatrix[2][3]],[self.transferMatrix[3][2],self.transferMatrix[3][3]]]
        else:
            M_transport=[[self.transferMatrix[4][4],self.transferMatrix[4][5]],[self.transferMatrix[5][4],self.transferMatrix[5][5]]]
        M_transport=np.array(M_transport)
        return M_transport
    def MatTwiss(self,dim='x'): # get the twiss matrix
        # extract transfer Matrix
        M_transport=self.MatTrans(dim=dim)
        # get the corresponding Twiss matrix
        M_Twiss=np.zeros((3,3))
        M_Twiss[0][0]=M_transport[0][0]*M_transport[0][0]
        M_Twiss[0][1]=-2*M_transport[0][0]*M_transport[0][1]
        M_Twiss[0][2]=M_transport[0][1]*M_transport[0][1]
        M_Twiss[1][0]=-M_transport[0][0]*M_transport[1][0]
        M_Twiss[1][1]=M_transport[0][1]*M_transport[1][0]+M_transport[0][0]*M_transport[1][1]
        M_Twiss[1][2]=-M_transport[0][1]*M_transport[1][1]
        M_Twiss[2][0]=M_transport[1][0]*M_transport[1][0]
        M_Twiss[2][1]=-2*M_transport[1][0]*M_transport[1][1]
        M_Twiss[2][2]=M_transport[1][1]*M_transport[1][1]
        return M_Twiss
    def PeriodTwiss(self,dim='x'):
        M_Twiss=self.MatTwiss(dim=dim)
        Twiss=np.linalg.solve(M_Twiss-np.identity(3),[1,0,0])
        # normalize: beta*gamma-alpha*alpha=det(Omega)=1
        Twiss_norm=Twiss/Twiss[0]
        k=Twiss_norm[0]*Twiss_norm[2]-Twiss_norm[1]*Twiss_norm[1]
        return Twiss_norm/np.sqrt(k)
    # get the beamsize along the beamline
    # emittance needs to be geometric not normalized as it uses sqrt(emit*beta) for the beamsize
    def getTraj(self,Twiss_ini,emit,dim='x'):
        z_list=[]
        beamsize_list=[]
        # initial point
        z_list.append(0)
        beamsize_list.append(np.sqrt(emit*Twiss_ini[0]))
        # loop
        z_temp=0
        mat_temp=np.identity(7)
        for i in range(len(self.matrixlist)):
            z_temp+=self.parameterlist[i]['len']
            z_list.append(z_temp)
            mat_temp=np.matmul(self.matrixlist[i],mat_temp) # progressing 7d matrix
            optics_temp=optics(mat_temp) # create a virtual beamline
            M_Twiss_temp=optics_temp.MatTwiss(dim=dim)
            Twiss_temp=np.matmul(M_Twiss_temp,Twiss_ini)
            beamsize_list.append(np.sqrt(emit*Twiss_temp[0]))
        z_list=np.array(z_list)
        beamsize_list=np.array(beamsize_list)
        return [z_list,beamsize_list]
    # get the beamposition along the beamline
    def getBeamTraj(self,coor_ini_list,dim='x'):
        z_list=[]
        x_list=[]
        if dim=='x':
            idx2look=0
        else:
            idx2look=2
        # initial point
        const_dim=np.ones((len(coor_ini_list),1))
        coor_ini_list=np.concatenate((coor_ini_list,const_dim),axis=1)
        for idx_particle in range(len(coor_ini_list)):
            # wanted particle
            coor_ini_temp=coor_ini_list[idx_particle]
            # initial coor
            x_list_temp=[]
            x_list_temp.append(coor_ini_temp[idx2look])
            # along the beamline: z axis
            if idx_particle==0:
                z_list.append(0)
            # loop
            z_temp=0
            mat_temp=np.identity(7)
            for i in range(len(self.matrixlist)):
                z_temp+=self.parameterlist[i]['len']
                if idx_particle==0:
                    z_list.append(z_temp)
                mat_temp=np.matmul(self.matrixlist[i],mat_temp) # progressing 7d matrix
                coor_temp=np.matmul(mat_temp,coor_ini_temp)
                x_list_temp.append(coor_temp[idx2look])
            x_list.append(x_list_temp)
        z_list=np.array(z_list)
        x_list=np.array(x_list)
        return [z_list,x_list]
    
###########################################################################################################################
# Supported optics components:
# using 6D coordinate x,x',y,y',t,deltaz'
# drift(name='drift',L=0) a drift with L
# quad(name='quad',L=1,K=1) a quadrapole with length L and strength K/k1, K>0 focusing, K<0 defocusing
# kicker(name='kicker',L=0,cx=0,cy=0) a kicker, length L, bending angle is cx / cy
###########################################################################################################################
# drift: with parameter len
class drift(optics):
    def __init__(self,name='drift',L=0):
        self.namelist=[name]
        parameter_drift=dict()
        parameter_drift['len']=L
        self.parameterlist=[parameter_drift]
        self.transferMatrix=np.array([[1,L,0,0,0,0,0],
                                      [0,1,0,0,0,0,0],
                                      [0,0,1,L,0,0,0],
                                      [0,0,0,1,0,0,0],
                                      [0,0,0,0,1,0,0],
                                      [0,0,0,0,0,1,0],
                                      [0,0,0,0,0,0,1]])
        self.matrixlist=[self.transferMatrix]
# quad: with parameter k, len
class quad(optics):
    def __init__(self,name='quad',L=1,K=1):
        self.namelist=[name]
        parameter_quad=dict()
        parameter_quad['len']=L
        parameter_quad['K']=K
        self.parameterlist=[parameter_quad]
        # focal case:
        if K>0:
            K=K
            self.transferMatrix=np.array([[np.cos(np.sqrt(K)*L),np.sin(np.sqrt(K)*L)/np.sqrt(K),0,0,0,0,0],
                                          [-np.sqrt(K)*np.sin(np.sqrt(K)*L),np.cos(np.sqrt(K)*L),0,0,0,0,0],
                                          [0,0,np.cosh(np.sqrt(K)*L),np.sinh(np.sqrt(K)*L)/np.sqrt(K),0,0,0],
                                          [0,0,np.sqrt(K)*np.sinh(np.sqrt(K)*L),np.cosh(np.sqrt(K)*L),0,0,0],
                                          [0,0,0,0,1,0,0],
                                          [0,0,0,0,0,1,0],
                                          [0,0,0,0,0,0,1]])
        # defocal case:
        else:
            K=-K
            self.transferMatrix=np.array([[np.cosh(np.sqrt(K)*L),np.sinh(np.sqrt(K)*L)/np.sqrt(K),0,0,0,0,0],
                              [np.sqrt(K)*np.sinh(np.sqrt(K)*L),np.cosh(np.sqrt(K)*L),0,0,0,0,0],
                              [0,0,np.cos(np.sqrt(K)*L),np.sin(np.sqrt(K)*L)/np.sqrt(K),0,0,0],
                              [0,0,-np.sqrt(K)*np.sin(np.sqrt(K)*L),np.cos(np.sqrt(K)*L),0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.matrixlist=[self.transferMatrix]

# kicker: with parameter len, cx, cy, assuming small len
class kicker(optics):
    def __init__(self, name='kicker', L=0, cx=0, cy=0):
        self.namelist=[name]
        parameter_kicker=dict()
        parameter_kicker['len']=L
        parameter_kicker['cx']=cx
        parameter_kicker['cy']=cy
        self.parameterlist=[parameter_kicker]
        self.transferMatrix=np.array([[1,0,0,0,0,0,0],
                                      [0,1,0,0,0,0,cx],
                                      [0,0,1,0,0,0,0],
                                      [0,0,0,1,0,0,cy],
                                      [0,0,0,0,1,0,0],
                                      [0,0,0,0,0,1,0],
                                      [0,0,0,0,0,0,1]])
        self.matrixlist=[self.transferMatrix]
        
###########################################################################################################################
# functions:
# getTwiss(x,xp) : statistically get the Twiss parameter using a list of particle coordinates
# Mismatch(Twiss1,Twiss0): # get the mismatch by definition
# getMismatch(x,xp,Twiss0) : statistically get the mismatch factor M=(gamma*beta0-2*alpha*alpha0+beta*gamma0)/2
# slice_emittance(x,xp,t,n) : statistically get the slice emittance for particles with (x,xp,t), assuming n slices
# slice_Twiss(x,xp,t,n) : statistically get the sliced Twiss parmeters for particles with (x,xp,t), assuming n slices
# slice_mismatch(x,xp,t,n,Twiss0) : statistically get the sliced mismatch factor M
# slice_statistic(xdata,ydata,slicenum) : sliced average & rms
# MatchTwiss(Twiss_in,Twiss_out,del_phi) : use the input and output Twiss para, and phase advance to build a transfer mat
###########################################################################################################################

def getTwiss(x,xp): # get the geo emittance and Twiss para
    x=np.array(x)
    xp=np.array(xp)
    beta=np.mean(x*x)-np.mean(x)**2
    gamma=np.mean(xp*xp)-np.mean(xp)**2
    alpha=-np.mean(x*xp)+np.mean(x)*np.mean(xp)
    detOmega=beta*gamma-alpha*alpha
    emit=np.sqrt(detOmega)
    return [emit,beta/emit,alpha/emit,gamma/emit]

def Mismatch(Twiss1,Twiss0): # get the mismatch by definition
    return (Twiss1[2]*Twiss0[0]-2*Twiss1[1]*Twiss0[1]+Twiss1[0]*Twiss0[2])/2

def getMismatch(x,xp,Twiss0): # statistically get the mismatch factor M=(gamma*beta0-2*alpha*alpha0+beta*gamma0)/2
    [emit,beta,alpha,gamma]=getTwiss(x,xp)
    return (gamma*Twiss0[0]-2*alpha*Twiss0[1]+beta*Twiss0[2])/2

def slice_emittance(x,xp,t,n): # get the slice emittance
    x=np.array(x)
    xp=np.array(xp)
    t=np.array(t)
    t_min=np.min(t)
    t_max=np.max(t)
    delt=(t_max-t_min)/n
    time_list=[]
    emit_list=[]
    for i in range(n-1):
        t_min_temp=t_min+delt*i
        t_max_temp=t_min+delt*(i+1)
        mask_temp=np.logical_and(t>t_min_temp, t<t_max_temp)
        x_temp=x[mask_temp]
        xp_temp=xp[mask_temp]
        [emit_temp,beta_temp,alpha_temp,gamma_temp]=getTwiss(x_temp,xp_temp)
        time_list.append((t_min_temp+t_max_temp)/2)
        emit_list.append(emit_temp)
    time_list=np.array(time_list)
    emit_list=np.array(emit_list)
    return [time_list,emit_list]

def slice_Twiss(x,xp,t,n): # get the slice Twiss
    x=np.array(x)
    xp=np.array(xp)
    t=np.array(t)
    t_min=np.min(t)
    t_max=np.max(t)
    delt=(t_max-t_min)/n
    time_list=[]
    beta_list=[]
    alpha_list=[]
    for i in range(n-1):
        t_min_temp=t_min+delt*i
        t_max_temp=t_min+delt*(i+1)
        mask_temp=np.logical_and(t>t_min_temp, t<t_max_temp)
        x_temp=x[mask_temp]
        xp_temp=xp[mask_temp]
        [emit_temp,beta_temp,alpha_temp,gamma_temp]=getTwiss(x_temp,xp_temp)
        time_list.append((t_min_temp+t_max_temp)/2)
        beta_list.append(beta_temp)
        alpha_list.append(alpha_temp)
    time_list=np.array(time_list)
    beta_list=np.array(beta_list)
    alpha_list=np.array(alpha_list)
    return [time_list,beta_list,alpha_list]

def slice_mismatch(x,xp,t,n,Twiss0): # statistically get the sliced mismatch factor M
    [time_list,beta_list,alpha_list]=slice_Twiss(x,xp,t,n)
    gamma_list=(1+alpha_list*alpha_list)/beta_list
    return [time_list, (gamma_list*Twiss0[0]-2*alpha_list*Twiss0[1]+beta_list*Twiss0[2])/2]

def slice_statistic(xdata,ydata,slicenum): # sliced average & rms
    binned_x=np.linspace(min(xdata),max(xdata),slicenum+1)
    binned_avg=[]
    binned_std=[]
    for i in range(slicenum):
        mask=np.logical_and(xdata>=binned_x[i],xdata<=binned_x[i+1])
        if np.sum(mask)>0:
            ydata_temp=ydata[mask]
            binned_avg.append(np.mean(ydata_temp))
            binned_std.append(np.std(ydata_temp))
        else:
            binned_avg.append(0)
            binned_std.append(0)
    binned_avg=np.array(binned_avg)
    binned_std=np.array(binned_std)
    print('done.')
    return [binned_x[0:-1],binned_avg,binned_std]

def MatchTwiss(Twiss_in,Twiss_out,del_phi): # use the input and output Twiss para, and phase advance to build a transfer mat
    beta_0=Twiss_in[0]
    beta_s=Twiss_out[0]
    alpha_0=Twiss_in[1]
    alpha_s=Twiss_out[1]
    M_trans=np.zeros((2,2))
    M_trans[0][0]=np.sqrt(beta_s/beta_0)*(np.cos(del_phi)+alpha_0*np.sin(del_phi))
    M_trans[0][1]=np.sqrt(beta_s*beta_0)*np.sin(del_phi)
    M_trans[1][0]=((alpha_0-alpha_s)*np.cos(del_phi)-(1+alpha_0*alpha_s)*np.sin(del_phi))/np.sqrt(beta_s*beta_0)
    M_trans[1][1]=np.sqrt(beta_0/beta_s)*(np.cos(del_phi)-alpha_s*np.sin(del_phi))
    return M_trans
