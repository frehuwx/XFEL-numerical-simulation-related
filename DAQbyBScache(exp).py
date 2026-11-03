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
import os


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
Laser_chs=[ #'SSL2-LSPC-SPEC1:SPECTRUM', # seed 2 spec meas
            #'SSL2-LSPC-SPEC1:WAVELENGTHS', # seed 2 spec meas (x axis)
            #'SSL1-LSPC-SPEC1:SPECTRUM', # seed 1 spec meas
            'SSL2-CPCW-SPEC01:SPECTRUM', # seed 2 spec meas
            'SSL-LADC-WL004:ADC1_CAL'] # seed 2 intensity

# slow channels (we only take one data per step)
slow_chs=[]

# other channels
def get_otherPV_chs(bunch2athos=Bunch2Athos):
    """Return the bunch-dependent list of other BS channels (bunch2athos = 1 or 2)."""
    if bunch2athos==1:
        otherPV_chs=[ "SINBC02-DBCM410:LM-AG-CH2-B1",
                      "S10MA01-DCDR080:LM-AG-CH1-B1",
                      "SATUN13-DBPM070:Q1",
                      "SATBD02-DBPM010:Q1",
                      #"SATBD02-DBPM010:Q1-VALID",
                      "SATMA02-RLLE-DSP:PHASE-VS",
                      "SATMA02-RLLE-DSP:AMPLT-VS",
                      "SATFE10-PEPG046:FCUP-INTENSITY-CAL",
                      "SATFE10-PEPG046-EVR0:CALCI",
                      "SATFE10-PEPG046:PHOTON-ENERGY-PER-PULSE-AVG"
                    ]
    else:
        otherPV_chs=[ "SINBC02-DBCM410:LM-AG-CH2-B2",
                      "S10MA01-DCDR080:LM-AG-CH1-B2",
                      "SATMA02-RLLE-DSP:PHASE-VS",
                      "SATMA02-RLLE-DSP:AMPLT-VS",
                      "SATFE10-PEPG046:FCUP-INTENSITY-CAL",
                      "SATFE10-PEPG046-EVR0:CALCI",
                      "SATFE10-PEPG046:PHOTON-ENERGY-PER-PULSE-AVG"
                    ]
    return otherPV_chs

otherPV_chs=get_otherPV_chs(Bunch2Athos)

# other channels, continue 
# these are the four camera parameters for the pointing and position of our seed lasers
# C1103 & C1104 : seed laser 1
# C1501 & C1502 : seed laser 2
# Slow BS channels
# SSL1-LTIM01-EVR0:DUMMY_PV1_NBS : Gecco Jiter Monitor
# SSL1-LTIM01-EVR0:DUMMY_PV2_NBS : Gecco Synch HV
# SSL1-LTIM01-EVR0:DUMMY_PV3_NBS : Lab Temperature
# SSL1-LTIM01-EVR0:DUMMY_PV4_NBS : Lab Humidity
# SSL1-LTIM01-EVR0:DUMMY_PV5_NBS : Hero Lab Energy
# SSL1-LTIM01-EVR0:DUMMY_PV6_NBS : EEHG Lab Energy
# SSL1-LTIM01-EVR0:DUMMY_PV7_NBS : HERO LAM
# SSL1-LTIM01-EVR0:DUMMY_PV8_NBS : EEHG LAM
# SSL1-LTIM01-EVR0:DUMMY_PV9_NBS : HERO Tunnel Temp
# SSL1-LTIM01-EVR0:DUMMY_PV10_NBS : HERO Tunnel humdiy
# SSL2-LTIM02-EVR0:DUMM_PV1_NBS : error seed laser 1
otherPV_chs_continue=['SSL-LCAM-C1103:intensity',
                      'SSL-LCAM-C1103:x_fit_mean',
                      'SSL-LCAM-C1103:x_fit_standard_deviation',
                      'SSL-LCAM-C1103:x_fwhm',
                      'SSL-LCAM-C1103:y_fit_mean',
                      'SSL-LCAM-C1103:y_fit_standard_deviation',
                      'SSL-LCAM-C1103:y_fwhm',
                      'SSL-LCAM-C1104:intensity',
                      'SSL-LCAM-C1104:x_fit_mean',
                      'SSL-LCAM-C1104:x_fit_standard_deviation',
                      'SSL-LCAM-C1104:x_fwhm',
                      'SSL-LCAM-C1104:y_fit_mean',
                      'SSL-LCAM-C1104:y_fit_standard_deviation',
                      'SSL-LCAM-C1104:y_fwhm',
                      'SSL-LCAM-C1501:intensity',
                      'SSL-LCAM-C1501:x_fit_mean',
                      'SSL-LCAM-C1501:x_fit_standard_deviation',
                      'SSL-LCAM-C1501:x_fwhm',
                      'SSL-LCAM-C1501:y_fit_mean',
                      'SSL-LCAM-C1501:y_fit_standard_deviation',
                      'SSL-LCAM-C1501:y_fwhm',
                      'SSL-LCAM-C1502:intensity',
                      'SSL-LCAM-C1502:x_fit_mean',
                      'SSL-LCAM-C1502:x_fit_standard_deviation',
                      'SSL-LCAM-C1502:x_fwhm',
                      'SSL-LCAM-C1502:y_fit_mean',
                      'SSL-LCAM-C1502:y_fit_standard_deviation',
                      'SSL-LCAM-C1502:y_fwhm',
                      'SSL1-LTIM01-EVR0:DUMMY_PV1_NBS',
                      'SSL1-LTIM01-EVR0:DUMMY_PV2_NBS',
                      'SSL1-LTIM01-EVR0:DUMMY_PV3_NBS',
                      'SSL1-LTIM01-EVR0:DUMMY_PV4_NBS',
                      'SSL1-LTIM01-EVR0:DUMMY_PV5_NBS',
                      'SSL1-LTIM01-EVR0:DUMMY_PV6_NBS',
                      'SSL1-LTIM01-EVR0:DUMMY_PV7_NBS',
                      'SSL1-LTIM01-EVR0:DUMMY_PV8_NBS',
                      'SSL1-LTIM01-EVR0:DUMMY_PV9_NBS',
                      'SSL1-LTIM01-EVR0:DUMMY_PV10_NBS',
                      'SSL2-LTIM02-EVR0:DUMMY_PV1_NBS',
                      'SSL2-LTIM02-EVR0:DUMMY_PV2_NBS',
                      'S10BC01-DBAM070:EOM1_T1',
                      'SINXB01-RLLE-DSP:PHASE-VS',
                      'SINXB01-RLLE-DSP:AMPLT-VS']
#otherPV_chs=np.concatenate((otherPV_chs,otherPV_chs_continue))

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
def get_BPM_chs(bunch2athos=Bunch2Athos):
    """Return the BPM X/Y channel names for the given Athos bunch (1 or 2)."""
    BPM_chs=[]
    for bpm in BPM_chs_raw:
        for dim in ['X', 'Y']:
            if bunch2athos==1:
                BPM_chs.append('%s:%s1' % (bpm, dim))
            else:
                BPM_chs.append('%s:%s2' % (bpm, dim))
    return BPM_chs

BPM_chs=get_BPM_chs(Bunch2Athos)

################################################################
# get the file name
################################################################
def setup_filename(comments=''): # setup the folder & file name
    # get the current date
    now=datetime.datetime.now()
    # get the directory (create the day folder if it does not exist yet)
    basefolder='/sf/data/measurements/%d/%02d/%02d/'%(now.year,now.month,now.day)+'spec_meas_by_bschannel/'
    os.makedirs(basefolder,exist_ok=True)
    # next free scan number
    n_scan=len(os.listdir(basefolder))+1
    while os.path.exists(basefolder+'scan%04d/'%(n_scan)):
        n_scan+=1
    foldername=basefolder+'scan%04d/'%(n_scan)
    os.mkdir(foldername)
    # get the filename
    timestamp='%04d_%02d_%02d_%02d_%02d_%02d'%(now.year,now.month,now.day,now.hour,now.minute,now.second)
    if len(comments)>0:
        filename=foldername+'scan%04d_'%(n_scan)+comments+'_'+timestamp+'.h5'
    else:
        filename=foldername+'scan%04d_'%(n_scan)+timestamp+'.h5'
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
        self.pv = None

    def connect_name(self, name):
        index = list(self.channels).index(name)
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

    def connect_all(self):
        print('Connecting to all the BS-Channels:')
        try:
            self.bs.get_vars(self.channels)
        except ValueError:
            print('Cannot find the channels!')
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
            if i%max(1,int(n_shots/10))==0:
                print('%d/%d'%(i,n_shots),end=' ')
        print('finish.')
        return data

    def sort_rawdata(self,rawdata):
        rawdata_sorted={}
        if len(rawdata)>0:
            for CHname in rawdata[0].keys():
                data_temp=[rawdata[i][CHname] for i in range(len(rawdata))]
                rawdata_sorted[CHname]=np.array(data_temp)
        return rawdata_sorted

################################################################
# Take the measurement
################################################################
# (guarded so that the GUI can import this module without running a measurement)
if __name__ == "__main__":

    # configure the measurement: 
    # measurement types: 'dummy' or 'scan'
    tag_meas_type='dummy'
    tag_return2init=False
    wait_time=10
    nshots=100

    # configure the scan name
    scanname_comments='MLSEQexp_'

    # configure the PVs
    scan_PVs=[]
    #scan_PVs=['SSL-LMOT-M1407:MOT']
    #scan_PVs=['SATUN05-MBND100:I-SET']
    #scan_PVs=['SLG-LMOT-M015:MOT']
    spec_PVs=[]
    spec_PVs=PSRD_Furka
    other_PVs=[]
    other_PVs=[BPM_chs]
    #other_PVs=[otherPV_chs,Laser_chs,BPM_chs]
    # other_PVs=[otherPV_chs,otherPV_chs_continue,Laser_chs,BPM_chs]
    # other_PVs=[otherPV_chs, Laser_chs, BPM_chs]
    #other_PVs=[Laser_chs,BPM_chs]

    # slow channels (i.e. we measure only once)
    slow_chs=['SATFE10-PEPG046:ENERGY-IN-USE',
              'SATFE10-PEPG046:PHOTONS-PER-SECOND-US.B',
              'SATFE10-PEPG046:PHOTONS-PER-SECOND-US.J',
              'SATFE10-PEPG046:PHOTONS-PER-SECOND-DS.B',
              'SATFE10-PEPG046:PHOTONS-PER-SECOND-DS.J',
              'SATFE10-PEPG046:PHOTONS-PER-SECOND-DS.G',
              'SATFE10-PEPG046:PHOTONS-PER-SECOND-DS.D']
              #'SLGBL-LENG-BIR2_IN:VAL_GET']

    # configure the scan range
    vals_begin=[3668.2]
    vals_end=[4068.2]
    nsteps=31
    scan_ranges=[]
    if tag_meas_type=='scan':
        if len(scan_PVs)==0 or len(scan_PVs)!=len(vals_begin) or len(scan_PVs)!=len(vals_end):
            sys.exit('Scan setup error: scan_PVs, vals_begin and vals_end must be non-empty and of the same length!')
        for i in range(len(scan_PVs)):
            slow_chs.append(scan_PVs[i])
        for i in range(len(vals_begin)):
            scan_ranges.append(np.linspace(vals_begin[i],vals_end[i],nsteps))
            #scan_ranges.append(np.array([0,5,10,20,30]))
        #scan_ranges=np.array(scan_ranges)
        #if len(scan_PVs)<2:
        #    scan_ranges=[scan_ranges]
        print('summary of the scan:')
        print('scan parameters:',end=' ')
        print(scan_PVs)
        for i in range(len(scan_PVs)):
            print('scan ranges for'+str(scan_PVs[i]),end=': ')
            print(scan_ranges[i])

    # get the channels
    Channels2Listen=spec_PVs
    for channel_list in other_PVs:
        Channels2Listen=np.concatenate((Channels2Listen,channel_list))
    Channels2Listen=list(dict.fromkeys(Channels2Listen)) # remove duplicates, keep the order
    print(Channels2Listen)

    # do the measurement
    if tag_meas_type=='dummy':
        print('This is a dummy scan, with %d shots'%(nshots))
        tag_continue=input('Type yes to continue: ')
        if tag_continue=='yes':
                
            # setup the dummy measurment
            DaqCache=DaqByBSCache(Channels2Listen)
            # add the requested channels to the cache
            DaqCache.connect_all()
            print('Connect finish')
       
            #for i in range(len(Channels2Listen)):
            #    DaqCache.connect(i)
            if nshots>500:
                print('Since the scan is longer than %d, will test the actual DAQ rate:'%nshots)
                raw_data=DaqCache.read_nshots(10)
                raw_data_sorted=DaqCache.sort_rawdata(raw_data)
                print('pids: ',end=' ')
                print(np.array(raw_data_sorted['pid']))
                print('difference: %d, equivalent to rep rate: %d Hz'%(np.mean(np.diff(raw_data_sorted['pid'])),100/np.mean(np.diff(raw_data_sorted['pid']))))

            # take the actual measurement
            print('Start real measuring...',end=' ')
            raw_data=DaqCache.read_nshots(nshots)
            raw_data_sorted=DaqCache.sort_rawdata(raw_data)

            # measure the slow parameters once
            rawdata_slow={}
            for CHname in slow_chs:
                data_temp=epics.caget(CHname)
                rawdata_slow[CHname]=np.array(data_temp)

            print('Done.')
                
            # save data
            print('Saving data...',end=' ')
            # get an output filename
            [output_foldernamem,output_filename]=setup_filename(scanname_comments)
            f=h5py.File(output_filename,'w')
            f.create_dataset('scan_type',data=['dummy'])
            f.create_dataset('nshots',data=nshots)
            for dataname in raw_data_sorted.keys():
                try:
                    f.create_dataset(dataname,data=raw_data_sorted[dataname])
                except:
                    print('ch exists: '+dataname,end=' ')
            env_data=f.create_group('environment data')
            for dataname in rawdata_slow.keys():
                env_data.create_dataset(dataname,data=rawdata_slow[dataname])
            f.close()
            print('Done. Data saved in:')
            print(output_filename)
            DaqCache.terminate()

    
    elif tag_meas_type=='scan':
        print('This is a parameter meter scan, with %d steps and %d shots for each step'%(nsteps,nshots))
        print('The scan channels are:'+str(scan_PVs))
        print('The scan values are:')
        for i in range(len(scan_ranges)):
            print(str(scan_ranges[i]))
        tag_continue=input('type yes to continue: ')
        if tag_continue=='yes':
                
            # setup the measurment
            DaqCache=DaqByBSCache(Channels2Listen)
            # add the requested channels to the cache
            DaqCache.connect_all()
            #for i in range(len(Channels2Listen)):
            #    DaqCache.connect(i)
            
            # take the init values
            init_vals=[]
            for i in range(len(scan_PVs)):
                init_vals.append(epics.caget(scan_PVs[i]))
            print(init_vals)
        
            # do the scan
            data_measured={}
            for dataname in Channels2Listen:
                data_measured[dataname]=[]
                for j in range(len(scan_ranges[0])):
                    data_measured[dataname].append([])
            for dataname in slow_chs:
                data_measured[dataname]=[]
                for j in range(len(scan_ranges[0])):
                    data_measured[dataname].append([])
                
            for i in range(len(scan_ranges[0])):
                #if np.abs(init_vals[0]-scan_ranges[0][0])>np.abs(init_vals[0]-scan_ranges[0][1]): # we scan from the end in this case
                #    i_scan=len(scan_ranges[0])-i-1
                #else: # we scan from the beginning in this case
                i_scan=i
                print('Step %d, go to the set values:'%(i+1), end=' ')

                for j in range(len(scan_PVs)):
                    print(scan_ranges[j][i_scan],end='')

                for j in range(len(scan_PVs)):
                    epics.caput(scan_PVs[j],scan_ranges[j][i_scan])

                time.sleep(wait_time)
                DaqCache.flush() # drop the data buffered while the PVs were moving
                print('Now take measurement...', end=' ')
                raw_data=DaqCache.read_nshots(nshots)
                raw_data_sorted=DaqCache.sort_rawdata(raw_data)
                for dataname in Channels2Listen:
                    data_measured[dataname][i_scan]=np.array(raw_data_sorted[dataname])
                print('Done.')

                # measure the slow parameters once
                print('Take environment data...', end=' ')
                rawdata_slow={}
                for CHname in slow_chs:
                    data_temp=epics.caget(CHname)
                    data_measured[CHname][i_scan]=np.array(data_temp)
                print('Done.')

            # save data
            print('Saving data...',end=' ')
            # get an output filename
            [output_foldernamem,output_filename]=setup_filename(scanname_comments)
            f=h5py.File(output_filename,'w')
            f.create_dataset('scan_type',data=['scan'])
            f.create_dataset('nsteps',data=nsteps)
            f.create_dataset('nshots',data=nshots)
            for i in range(len(scan_PVs)):
                try:
                    f.create_dataset(scan_PVs[i]+'_scan',data=scan_ranges[i]) # the scan values
                    f.create_dataset(scan_PVs[i]+'_init',data=init_vals[i])   # the value before the scan
                except:
                    print('ch exists: '+scan_PVs[i],end=' ')
            for dataname in data_measured.keys():
                try:
                    f.create_dataset(dataname,data=data_measured[dataname])
                except:
                    print('ch exists: '+dataname,end=' ')
            env_data=f.create_group('environment data')
            for dataname in slow_chs: # one value per step, shape: [nsteps]
                try:
                    env_data.create_dataset(dataname,data=np.array(data_measured[dataname]))
                except Exception as e:
                    print('could not save %s: %s'%(dataname,e),end=' ')
            f.close()
            print('Done. Data saved in:')
            print(output_filename)
                
            # return to initial value            
            if tag_return2init:
                print('Returning to initial values...')
                for i in range(len(scan_PVs)):
                    epics.caput(scan_PVs[i],init_vals[i])
                print('Done.')
            DaqCache.terminate()

    else:
        print('unable to configure the scan!')
