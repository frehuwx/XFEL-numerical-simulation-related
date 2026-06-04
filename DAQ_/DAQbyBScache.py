from bstrd import BSCache
from epics import PV
import epics
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import datetime


################################################################
# PV names
################################################################

PMOS_Maloja=['SATOP21-PMOS127-2D:SPECTRUM_X','SATOP21-PMOS127-2D:SPECTRUM_Y']
PSRD_Furka=['SATOP31-PSRD132:SPECTRUM_X','SATOP31-PSRD132:SPECTRUM_Y']
PMOS_Furka=['SATOP31-PMOS132-2D:SPECTRUM_X','SATOP31-PMOS132-2D:SPECTRUM_Y']
PSSS=['SARFE10-PSSS059:SPECTRUM_X','SARFE10-PSSS059:SPECTRUM_Y']

################################################################
# BSCache class
################################################################

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
        # self.bs.channels.clear()
        self.hasBStream=True
        try:
            self.bs.get_vars([self.channel])  # this starts the stream into the cache
        except ValueError:
            print('Cannot find requested channels in BS stream!')
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

################################################################
# Take the measurement
################################################################

# configure the measurement: 
# measurement types: 'dummy' or 'scan'
tag_meas_type='dummy'
tag_return2init=False
wait_time=5
nshots=100

# configure the PVs
scan_PV=''
spec_PVs=PSRD_Furka
other_PVs=[]

# configure the scan range
val_begin=0
val_end=0
nsteps=1
scan_range=np.linspace(val_begin,val_end,nsteps)

# configure the scan name
scanname_comments=''

# get the channels
Channels2Listen=np.concatenate(spec_PVs,other_PVs)
print(Channels2Listen)

# do the measurement
if tag_meas_type='dummy':
    print('This is a dummy scan, with %d shots'%(nshots))
    tag_continue=input('type yes to continue')
    if tag_continue=='yes':
        # do the dummy measurment
        DaqCache=DaqByBSCache(Channels2Listen)
        # add the requested channels to the cache
        for i in range(len(Channels2Listen)):
            DaqCache.connect(i)
        # take the actual measurement
        raw_data=DaqCache.read_nshots(nshots)
        
    
elif tag_meas_type='scan':
    print('This is a parameter meter scan, with %d steps and %d shots for each step'%(nsteps,nshots))
    print('The scan values are:'+str(scan_range))
    if tag_continue=='yes':
        # do the scan
    
else:
    print('unable to configure the scan!')
