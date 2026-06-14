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

# bunch destinations
Bunch2Athos=2 # Remembr to change it to 1 if bunch destinations are swapped!

# spectrometers
PMOS_Maloja=['SATOP21-PMOS127-2D:SPECTRUM_X','SATOP21-PMOS127-2D:SPECTRUM_Y'] # PMOS Maloja
PSRD_Furka=['SATOP31-PSRD132:SPECTRUM_X','SATOP31-PSRD132:SPECTRUM_Y'] # PSRD
PMOS_Furka=['SATOP31-PMOS132-2D:SPECTRUM_X','SATOP31-PMOS132-2D:SPECTRUM_Y'] # PMOS Furka
PSSS=['SARFE10-PSSS059:SPECTRUM_X','SARFE10-PSSS059:SPECTRUM_Y'] # PSSS (Aramis)

# laser (seed 2) channels
Laser_chs=[ #'SSL2-LSPC-SPEC1:SPECTRUM', # seed 2 spec meas (old)
            #'SSL2-LSPC-SPEC1:WAVELENGTHS', # seed 2 spec meas (x axis) (old)
            'SSL2-CPCW-SPEC01:SPECTRUM', # seed 2 spec meas
            'SSL-LADC-WL004:ADC1_CAL'] # seed 2 intensity

# other channels
if Bunch2Athos==1:
    otherPV_chs=[ "SINBC02-DBCM410:LM-AG-CH2-B1",
                  "S10MA01-DCDR080:LM-AG-CH1-B1",
                  "SATUN13-DBPM070:Q1",
                  "SATBD02-DBPM010:Q1",
                  "SATBD02-DBPM010:Q1-VALID",
                  "SATMA02-RLLE-DSP:PHASE-VS",
                  "SATMA02-RLLE-DSP:AMPLT-VS",
                  "SATFE10-PEPG046:FCUP-INTENSITY-CAL",
                  "SATFE10-PEPG046-EVR0:CALCI"]
else:
    otherPV_chs=[ "SINBC02-DBCM410:LM-AG-CH2-B2",
                  "S10MA01-DCDR080:LM-AG-CH1-B2",
                  "SATMA02-RLLE-DSP:PHASE-VS",
                  "SATMA02-RLLE-DSP:AMPLT-VS",
                  "SATFE10-PEPG046:FCUP-INTENSITY-CAL",
                  "SATFE10-PEPG046-EVR0:CALCI"]

# BPMs
BPM_chs_raw=[ "SINBC02-DBPM140",
              "SINBC02-DBPM320",
              "S10BC02-DBPM140",
              "S10BC02-DBPM320",
              "SATMA01-DBPM620",
              "SATDI01-DBPM030",
              "SATDI01-DBPM060",
              "SATUN04-DBPM010",
              "SATUN05-DBPM410",
              "SATMA02-DBPM030",
              "SATSY02-DBPM020",
              "SATSY02-DBPM210",
              "SATSY03-DBPM030",
              "SATSY03-DBPM060",
              "SATSY03-DBPM090",
              "SATSY03-DBPM120",
              "SATCL01-DBPM140",
              "SATDI01-DBPM210",
              "SATDI01-DBPM240",
              "SATDI01-DBPM270",
              "SATDI01-DBPM310",
              "SATBD02-DBPM010"]
for n_und in range(6, 23):
    if n_und == 14:
        continue
    BPM_chs_raw.append('SATUN%02i-DBPM070' % n_und)
            
# configure the PV names
BPM_chs=[]
for bpm in BPM_chs_raw:
    for dim in ['X', 'Y']:
        if Bunch2Athos==1:
            BPM_chs.append('%s:%s1' % (bpm, dim))
        else:
            BPM_chs.append('%s:%s2' % (bpm, dim))


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
other_PVs=[otherPV_chs, Laser_chs, BPM_chs]

# configure the scan range
val_begin=0
val_end=0
nsteps=1
scan_range=np.linspace(val_begin,val_end,nsteps)

# configure the scan name
scanname_comments=''

# get the channels
Channels2Listen=spec_PVs
for channel_list in other_PVs:
    Channels2Listen=np.concatenate((Channels2Listen,channel_list))
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
