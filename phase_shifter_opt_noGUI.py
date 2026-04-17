from bstrd import BSCache
from epics import PV
import epics
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import datetime
import optuna

# Tags for test control
Tag_bebug=True
Tag_usefakeresults=True

################################################################
# PV names
################################################################
phase_shifter_PVs=['','','','','','','','','','','','','','','']
intensity_cali_PV='SATFE10-PEPG046:FCUP-INTENSITY-CAL'
intensity_uncali_PV='SATFE10-PEPG046-EVR0:CALCI'
PMOS_Maloja=['SATOP21-PMOS127-2D:SPECTRUM_X','SATOP21-PMOS127-2D:SPECTRUM_Y']
#PSRD_Maloja=['','']
PMOS_Furka=['SATOP31-PMOS132-2D:SPECTRUM_X','SATOP31-PMOS132-2D:SPECTRUM_Y']



################################################################
# basic classes
################################################################

# use BSCache to read a desired channel list:
# (referencing https://gitea.psi.ch/slic/sfbd/src/branch/master/app/spectralanalysis.py#L1 )
class DaqByBSCache():
    def __init__(self,channel_list):
        self.bs = BSCache(100000,receive_timeout=10000)  # 100 second timeout, size for 10 second data taken
        self.bs.stop()
        self.channel = None
        self.channels = channel_list
        self.hasBStream=False

    def connect_name(self, name):
        index = names.index(name)
        self.connect(index)

    def connect(self,ich):
        if ich < 0 or ich >= len(self.channels):
            return False
        self.channel = self.channels[ich]
        print('Connecting to BS-Channel:',self.channel)
        self.bs.channels.clear()
        self.hasBStream=True
        try:
            self.bs.get_vars(self.channel)  # this starts the stream into the cache
        except ValueError:
            print('Cannot find requested channels in BS stream')
            self.hasBStream=False
        self.pv = None

    def terminate(self):
        print('Stopping BSStream Thread...')
        self.bs.stop()
        self.bs.pt.running.clear() # for some reason I have to
        if self.pv:
            self.pv.disconnect()

    def flush(self):
        self.bs.flush()

    def read(self):
        return next(self.bs)
    
    def read_nshots(self,n_shots):
        data=[]
        for i in range(int(n_shots)):
            data.append(self.read())
        return data


# use optuna for Bayesian optimization
# TPE based for now, and will always assume it is to maximize something
class BayesianOpt():
    def __init__(self,Channels2Opt=[],Channels2Listen=[],n_shots=1,n_iter=1,bounds=[]):
        # parameters
        self.Channels2Opt=Channels2opt
        self.Channels2Listen=Channels2Listen
        self.n_shots=n_shots
        self.n_iter=n_iter
        self.bounds=bounds
        self.n_trial=0
        
        # setup the BS cache and Optimizer
        self.DaqCache=DaqByBSCache(self.Channels2Listen)
        # add the requested channels to the cache
        for i in range(len(Channels2Listen)):
            self.DaqCache.connect(i)
        self.Optimizer=optuna.create_study(studyname='Optimizer'+str(datetime.datetime.now()),direction='maximize')
        
        # get the values before optimization
        self.init_vals=[]
        for i in range(len(self.Channels2Opt)):
            self.init_vals.append(epics.caget(self.Channels2Opt[i]))
        # also initialize the best cases
        self.best_vals=self.init_vals
    
    def set_analysis_func(self,analysis_func): # setting the post analysis function for the optimization
        self.analysis_func=analysis_func
    
    def revert(self): # revert the optimizing channels to their initial value
        for i in range(len(self.Channels2Opt)):
            if not Tag_debug:
                epics.caput(self.Channels2Opt[i],self.init_vals[i])
            else:
                print('I am trying to put '+self.Channels2Opt[i]+' to '+'%.2f'%(self.init_vals[i])+'. I will not do it in debug')
    
    def set2best(self): # set the optimizing channels to their best value
        for i in range(len(self.Channels2Opt)):
            if not Tag_debug:
                epics.caput(self.Channels2Opt[i],self.best_vals[i])
            else:
                print('I am trying to put '+self.Channels2Opt[i]+' to '+'%.2f'%(self.best_vals[i])+'. I will not do it in debug')
    
    def object_func(self,trial): # the objective function to optimize
        print('starting %d trial:'%(self.n_trial))
        # get the new parameters
        new_paras=[]
        for i in range(len(self.Channels2Opt)):
            temp=trial.suggest_float(self.Channels2Opt[i], self.bounds[i][0], self.bounds[i][1])
            if not Tag_debug:
                epics.caput(self.Channels2Opt[i],temp)
            else:
                print('I am trying to put '+self.Channels2Opt[i]+' to '+'%.2f'%(temp)+'. I will not do it in debug')
            new_paras.append(temp)
        
        # wait some time
        time.sleep(10)
        
        # record the data
        if not Tag_usefakeresults:
            self.DaqCache.flush()
            data_temp=self.DaqCache.read_nshots(self.n_iter)
            self.rawdata=data_temp
        else:
            freq=np.linspace(-10,10,501)
            data_list=[]
            for i in range(self.n_iter):
                data_temp={}
                spec_temp=get_fake_data(freq,np.array(new_paras))
                data_temp[Channels2Listen[0]]=np.array(freq)
                data_temp[Channels2Listen[1]]=np.array(spec_temp)
                fake=np.linspace(0,len(values),len(values))
                data_temp[Channels2Listen[2]]=-np.sum((values-fake)**2)+1e2
                data_list.append(data_temp)
            self.rawdata=data_list

        # post analysis to get the evaluation value
        val=self.analysis_func(self.rawdata)
        if val>-1e9:
            print('Finished with success.')
        else:
            print('Finished with errors.')
        self.n_trial+=1
        
        return val
    
    def measure(self): # do one measurement
        # record the data
        if not Tag_usefakeresults:
            self.DaqCache.flush()
            data_temp=self.DaqCache.read_nshots(self.n_iter)
            self.rawdata=data_temp
        else:
            freq=np.linspace(-10,10,501)
            data_list=[]
            for i in range(self.n_iter):
                data_temp={}
                spec_temp=get_fake_data(freq,np.array(new_paras))
                data_temp[Channels2Listen[0]]=np.array(freq)
                data_temp[Channels2Listen[1]]=np.array(spec_temp)
                fake=np.linspace(0,len(values),len(values))
                data_temp[Channels2Listen[2]]=-np.sum((values-fake)**2)+1e2
                data_list.append(data_temp)
            self.rawdata=data_list
        
        # post analysis to get the evaluation value
        val=self.analysis_func(self.rawdata)
        if val>-1e9:
            print('Measurement finish with success.')
        else:
            print('Measurement finish with errors.')
        self.n_trial+=1
        
        return val
    
    def run_optimization(self,n_iter=-1):
        # update the iteration time
        if int(n_iter)>0:
            self.n_iter=n_iter
        
        # we perform an initial measurement and add that to our trial list
        val_temp=self.measure()
        if val_temp>-1e9:
            parameter_dict={}
            distribution_dict={}
            for i in range(len(self.Channels2Opt)):
                parameter_dict[self.Channels2Opt[i]]=self.init_vals[i]
                distribution_dict[self.Channels2Opt[i]]=optuna.distributions.FloatDistribution(self.bounds[i][0],self.bounds[i][1])
            init_trial=optuna.trial.create_trial(params=parameter_dict, distributions=distribution_dict, value=val_temp)
            self.Optimizer.add_trial(init_trial)
        
        # do the iteraction
        self.n_trial=0
        self.Optimizer.optimize(self.object_func,self.n_iter)
        
        # update the best values
        best_results=self.Optimizer.best_params
        self.best_vals=[]
        for i in range(len(self.Channels2Opt)):
            self.best_vals.append(best_results[self.Channels2Opt[i]])
            
        if len(self.best_vals)==len(self.init_vals):
            print('finished with the best values:')
            print(self.best_vals)
        else:
            print('finished with error.')
        

################################################################
# implementation to ML-SASE
################################################################
    
################################################################
# data analysis part
################################################################
def FFT_avgs(xdata_eV,ydatalist):
    delx=(np.max(xdata_eV)-np.min(xdata_eV))/(len(xdata_eV)-1)
    nx=len(xdata_eV)
    xdata_fs=1/delx/nx*np.linspace(-nx/2,nx/2-1,nx)
    e=1.6022e-19
    h=6.626070e-34
    xdata_fs=xdata_fs*h/e*1e15
    Aylist=[]
    for i in range(len(ydatalist)):
        Aylist.append(np.fft.fftshift(np.fft.fft(ydatalist[i])))
    Aylist=np.array(Aylist)
    avg1=np.abs(np.nanmean(Aylist,axis=0))
    avg2=np.nanmean(np.abs(Aylist),axis=0)
    return [xdata_fs,avg1,avg2]

def Gaussian_bg(x,A,x0,sig,bg):
    return A*np.exp(-(x-x0)**2/2/sig**2)+bg
def Gaussian_fit(x,y,xrange=[]):
    # trim data
    if len(xrange)==2:
        mask=np.logical_and(x>xrange[0],x<xrange[1])
        x_trim=np.array(x[mask])
        y_trim=np.array(y[mask])
    else:
        x_trim=np.array(x)
        y_trim=np.array(y)
    # guess and fit
    try:
        # guess
        delx=(np.max(x_trim)-np.min(x_trim))/(len(x_trim)-1)
        A_guess=np.max(y_trim)
        x0_guess=x_trim[np.argmax(y_trim)]
        sig_guess=np.sum(y_trim>A_guess/2)*delx/2.355
        # fit
        p,proc=curve_fit(Gaussian_bg,x_trim,y_trim,p0=[A_guess,x0_guess,sig_guess,0])
    except:
        p=np.zeros(4)
        p[0]=-1e10
    return p

def analysis_function(freq,spec,intensity,method='spec'):
    if method=='spec': # spectral analysis
        [time_fs,avg1,avg2]=FFT_avgs(freq,spec)
        p=Gaussian_fit(time_fs,avg1,xrange=[1,10])
        return p[0]
    else:
        return np.nanmean(intensity)
    

# get some fake data
def get_fake_data(freq,values):
    del_eV=2
    fake=np.linspace(0,len(values),len(values))
    sig=6
    A=-np.sum((values-fake)**2)+1e2
    return A*np.exp(-freq**2/2/sig**2)*(np.sin(np.pi/del_eV*freq))**2


################################################################
# combine the analysis with the classes
################################################################

# build the optimizer
idx_list=[0,1]
for idx in idx_list:
    Channels2opt.append(phase_shifter_PVs[idx])
    bounds.append([0,360])
Channels2listen=[PMOS_Maloja[0],PMOS_Maloja[1],intensity_cali_PV]

ML_opt=BayesianOpt(Channels2opt=Channels2opt,Channels2Listen=Channels2listen,n_shots=100,n_iter=10,bounds=bounds)


method='inten'
freq_name=Channels2listen[0]
spec_name=Channels2listen[1]
inten_name=Channels2listen[2]
def ML_analysis(data):
    if method=='inten':
        inten_list=[]
        for i in range(len(data)):
            inten_list.append(data[inten_name])
        val=np.nanmmean(np.array(inten_list))
    else:
        freq=data[0][freq_name] # we will assume that the x axis remains unchanged
        spec_list=[]
        for i in range(len(data)):
            spec_list.append(data[spec_name])
        [time_fs,avg1,avg2]=FFT_avgs(freq,np.array(spec_list))
        p=Gaussian_fit(time_fs,avg1,xrange=[1,10])
        val=p[0]
    return val
        
ML_opt.set_analysis_func(ML_analysis)
    
# test 1    
ML_opt.measure()

# test 2
ML_opt.run_optimization()



