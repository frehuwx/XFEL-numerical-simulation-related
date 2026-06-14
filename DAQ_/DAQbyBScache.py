from bstrd import BSCache
from epics import PV
import epics
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import datetime
import h5py


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
# get the file name
################################################################
def setup_filename(comments=''): # setup the folder & file name
    # get the current date
    year=datetime.now().year
    month=datetime.now().month
    day=datetime.now().day
    hour=datetime.now().hour
    minute=datetime.now().minute
    second=datetime.now().second
    # get the directory
    foldername='/sf/data/measurements/%d/%02d/%02d/'%(year,month,day)+'spec_meas_by_bschannel/'
    filename='/sf/data/measurements/%d/%02d/%02d/'%(year,month,day)
    # get the filename
    try:
        ch_cur=os.listdir(foldername)
        foldername=foldername+'scan%04d/'%(len(ch_cur)+1)
        os.mkdir(foldername)
        if len(comments)>0:
            filename=foldername+'scan%04d_'%(len(ch_cur)+1)+comments+'_%04d_%02d_%02d_%02d_%02d_%02d'%(year,month,day,hour,minute,second)+'.h5'
        else:
            filename=foldername+'scan%04d_'%(len(ch_cur)+1)+'%04d_%02d_%02d_%02d_%02d_%02d'%(year,month,day,hour,minute,second)+'.h5'
    except:
        os.mkdir(foldername)
        foldername=foldername+'scan%04d/'%(1)
        os.mkdir(foldername)
        if len(comments)>0:
            filename=foldername+'scan%04d_'%(1)+comments+'_%04d_%02d_%02d_%02d_%02d_%02d'%(year,month,day,hour,minute,second)+'.h5'
        else:
            filename=foldername+'scan%04d_'%(1)+'%04d_%02d_%02d_%02d_%02d_%02d'%(year,month,day,hour,minute,second)+'.h5'
    return [foldername,filename]

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
scan_PVs=[]
spec_PVs=[]
spec_PVs=PSRD_Furka
other_PVs=[otherPV_chs, Laser_chs, BPM_chs]

# configure the scan range
vals_begin=[]
vals_end=[]
nsteps=1
scan_ranges=[]
for i in range(len(val_begin)):
    scan_ranges.append(np.linspace(vals_begin[i],vals_end[i],nsteps))

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
    tag_continue=input('Type yes to continue')
    if tag_continue=='yes':
                
        # setup the dummy measurment
        DaqCache=DaqByBSCache(Channels2Listen)
        # add the requested channels to the cache
        for i in range(len(Channels2Listen)):
            DaqCache.connect(i)
                    
        # take the actual measurement
        print('Start measuring...',end=' ')
        raw_data=DaqCache.read_nshots(nshots)
        print('Done.')
                
        # save data
        print('Saving data...',end=' ')
        # get an output filename
        [output_foldernamem,output_filename]=setup_filename(scanname_comments)
        f=h5py.File(output_filename,'w')
        f.create_dataset('scan_type',data=['dummy'])
        f.create_dataset('nshots',data=nshots)
        for dataname in Channels2Listen:
            h5py.create_dataset(dataname,data=raw_data[dataname])
        f.close()
        print('Done. Data saved in:')
        print('output_filename')
    
elif tag_meas_type='scan':
    print('This is a parameter meter scan, with %d steps and %d shots for each step'%(nsteps,nshots))
    print('The scan channels are:'+str(scan_PVs))
    print('The scan values are:')
    for i in range(len(scan_ranges)):
        print(str(scan_ranges[i]))
    tag_continue=input('type yes to continue')
    if tag_continue=='yes':
                
        # setup the measurment
        DaqCache=DaqByBSCache(Channels2Listen)
        # add the requested channels to the cache
        for i in range(len(Channels2Listen)):
            DaqCache.connect(i)
                    
        # take the init values
        init_vals=[]
        for i in range(len(scan_PVs)):
            init_vals.append(epics.caget(scan_PVs[i]))
                    
        # do the scan
        if np.abs(init_vals[0]-scan_ranges[0][0])<np.abs(init_vals[0]-scan_ranges[0][1]): # we scan from the beginning in this case
            for i in range(len()):
                print('Start measuring %d, go to the set values')
                print('Done.')
                time.sleep(wait_time)
            init_vals
        else: # we scan from the end in this case
            init
                    
        # save data
        print('Saving data...')
        # get an output filename
        [output_foldernamem,output_filename]=setup_filename(scanname_comments)
                
        # return to initial value            
        if tag_return2init:
            print('Returning to initial values...')
            for i in range(len(scan_PVs)):
                epics.caput(scan_PVs[i],init_vals[i])
            print('Done.')

        
else:
    print('unable to configure the scan!')
