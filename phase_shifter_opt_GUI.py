import optuna
from phase_shifter_opt_noGUI import *

################################################################
# PV names
################################################################
phase_shifter_PVs=['SATUN06-UDLY060:PH-SHIFT-OP',
                   'SATUN07-UDLY060:PH-SHIFT-OP',
                   'SATUN08-UDLY060:PH-SHIFT-OP',
                   'SATUN09-UDLY060:PH-SHIFT-OP',
                   'SATUN10-UDLY060:PH-SHIFT-OP',
                   'SATUN11-UDLY060:PH-SHIFT-OP',
                   'SATUN12-UDLY060:PH-SHIFT-OP',
                   'SATUN13-UDLY060:PH-SHIFT-OP',
                   'SATUN15-UDLY060:PH-SHIFT-OP',
                   'SATUN16-UDLY060:PH-SHIFT-OP',
                   'SATUN17-UDLY060:PH-SHIFT-OP',
                   'SATUN18-UDLY060:PH-SHIFT-OP',
                   'SATUN19-UDLY060:PH-SHIFT-OP',
                   'SATUN20-UDLY060:PH-SHIFT-OP',
                   'SATUN21-UDLY060:PH-SHIFT-OP']
Athos_intensity_cali_PV='SATFE10-PEPG046:FCUP-INTENSITY-CAL'
Athos_intensity_uncali_PV='SATFE10-PEPG046-EVR0:CALCI'
PMOS_Maloja=['SATOP21-PMOS127-2D:SPECTRUM_X','SATOP21-PMOS127-2D:SPECTRUM_Y']
PSRD_Maloja=['SATOP31-PSRD132:SPECTRUM_X','SATOP31-PSRD132:SPECTRUM_Y']
PMOS_Furka=['SATOP31-PMOS132-2D:SPECTRUM_X','SATOP31-PMOS132-2D:SPECTRUM_Y']
PSSS=['SARFE10-PSSS059:SPECTRUM_X','SARFE10-PSSS059:SPECTRUM_Y']

################################################################
# analysis function
################################################################
def convert_data(data,channel_names=PMOS_Maloja):
    freq=np.array(data[0][channel_names[0]])
    spec_list=[]
    inten_list=[]
    for i in range(len(data)):
        spec_list.append(np.array(data[i][channel_names[1]]))
        inten_list.append(data[i][channel_names[2]])
    spec_list=np.array(spec_list)
    inten_list=np.array(inten_list)
    return [freq,spec_list,inten_list]

def ML_analysis(data,method='inten',channel_names=PMOS_Maloja):
    if method=='inten':
        inten_list=[]
        for i in range(len(data)):
            inten_list.append(data[i][channel_names[2]])
        val=np.nanmean(np.array(inten_list))
    else:
        freq=data[0][channel_names[0]] # we will assume that the x axis remains unchanged
        spec_list=[]
        for i in range(len(data)):
            spec_list.append(data[i][channel_names[1]])
        [time_fs,avg1,avg2]=FFT_avgs(freq,np.array(spec_list))
        p=Gaussian_fit(time_fs,avg1,xrange=[1,10])
        val=p[0]
    return val


################################################################
# GUI
################################################################
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        
        # initailize the setup
        self.n_shots=1
        self.n_iter=1
        self.idx_start=1
        self.idx_end=1
        self.optimizer=[] #BayesianOpt()
        self.init_freq=[]
        self.init_spec=[]
        self.Channels2opt=[]
        self.Channels2listen=[PMOS_Maloja[0],PMOS_Maloja[1],Athos_intensity_cali_PV]
        self.method='inten'
        
        #####################
        # window geometry
        #####################
        self.setWindowTitle("Phase shifter optimization")
        self.setGeometry(100, 100, 1200, 700)
        
        # main panel 
        self._main = QtWidgets.QWidget()
        layout_main = QtWidgets.QVBoxLayout(self._main)
        
        #####################
        # optimization initialization
        #####################
        layout_init = QtWidgets.QHBoxLayout()
        label1 = QtWidgets.QLabel('Optimization initialization:')
        label2 = QtWidgets.QLabel('Phase shifter start idx:')
        label3 = QtWidgets.QLabel('end idx:')
        label4 = QtWidgets.QLabel('Spectrometer')
        self.spectrometer_box=QtWidgets.QComboBox()
        self.spectrometer_box.addItems(["PMOS Maloja", "PSRD Maloja","PMOS Furka","PSSS"])
        label5 = QtWidgets.QLabel('number of shots:')
        self.start_line = QtWidgets.QLineEdit()
        self.start_line.setMaxLength(2)
        self.end_line = QtWidgets.QLineEdit()
        self.end_line.setMaxLength(2)
        self.numofshots_line= QtWidgets.QLineEdit()
        self.numofshots_line.setMaxLength(4)
        init_button=QtWidgets.QPushButton('Initialize!')
        init_button.clicked.connect(self.opt_initialize)
        
        layout_init.addWidget(label1)
        layout_init.addWidget(label2)
        layout_init.addWidget(self.start_line)
        layout_init.addWidget(label3)
        layout_init.addWidget(self.end_line)
        layout_init.addWidget(label4)
        layout_init.addWidget(self.spectrometer_box)
        layout_init.addWidget(label5)
        layout_init.addWidget(self.numofshots_line)
        layout_init.addWidget(init_button)
        layout_main.addLayout(layout_init)
        
        #####################
        # optimization setup
        #####################
        layout_setup = QtWidgets.QHBoxLayout()
        label5 = QtWidgets.QLabel('Optimization setup:')
        label6 = QtWidgets.QLabel('Number of iterations:')
        self.niter_line = QtWidgets.QLineEdit()
        self.niter_line.setMaxLength(4)

        self.method_box=QtWidgets.QComboBox()
        self.method_box.addItems(["Based on intensity", "Based on coherence"])
        
        start_button=QtWidgets.QPushButton('Start!')
        start_button.clicked.connect(self.opt_start)
        
        layout_setup.addWidget(label5)
        layout_setup.addWidget(label6)
        layout_setup.addWidget(self.niter_line)
        layout_setup.addWidget(self.method_box)
        layout_setup.addWidget(start_button)
        layout_main.addLayout(layout_setup)
        
        #####################
        # control buttons
        #####################
        layout_controls = QtWidgets.QHBoxLayout()
        go_button=QtWidgets.QPushButton('Go to the current optimial')
        revert_button=QtWidgets.QPushButton('Revert to initial')
        go_button.clicked.connect(self.go_optimial)
        revert_button.clicked.connect(self.revert_init)
        
        layout_controls.addWidget(go_button)
        layout_controls.addWidget(revert_button)
        layout_main.addLayout(layout_controls)
        
        #####################
        # main plot
        #####################
        plot_results = FigureCanvas(Figure(figsize=(5, 3),layout='constrained'))
        layout_main.addWidget(plot_results)
        self._ax = plot_results.figure.subplots(1,3)
        
        # plot main panel
        self._main.setLayout(layout_main)
        self.setCentralWidget(self._main)
        
    def opt_initialize(self):
        # update the parameters
        try:
            self.idx_start=int(self.start_line.text())
            self.idx_end=int(self.end_line.text())
        except:
            self.idx_start=1
            self.idx_end=1
            
        try:
            self.n_shots=int(self.numofshots_line.text())
        except:
            self.n_shots=1
            
        # setup the optimizer
        shifter_idx_list=np.arange(self.idx_start,self.idx_end)
        self.Channels2opt=[]
        bounds=[]
        for idx in shifter_idx_list:
            self.Channels2opt.append(phase_shifter_PVs[idx])
            bounds.append([-200,200])
        
        if "PMOS Furka" in self.spectrometer_box.currentText():
            self.Channels2listen=[PMOS_Furka[0],PMOS_Furka[1],Athos_intensity_cali_PV]
        elif "PSRD Maloja" in self.spectrometer_box.currentText():
            self.Channels2listen=[PSRD_Maloja[0],PSRD_Maloja[1],Athos_intensity_cali_PV]
        elif "PSSS" in self.spectrometer_box.currentText():
            self.Channels2listen=[PSSS[0],PSSS[1],Athos_intensity_cali_PV]
        else:
            self.Channels2listen=[PMOS_Maloja[0],PMOS_Maloja[1],Athos_intensity_cali_PV]
            
        if self.method_box.currentIndex()==0:
            self.method='inten'
        else:
            self.method='coherence'
            
        print('The channels to optimize:', end=' ')
        print(Channels2opt)
        print('The channels to listen:', end=' ')
        print(Channels2listen)
        print('method:'+ method)
        
        # create the optimizer
        self.optimizer=BayesianOpt(Channels2Opt=Channels2opt,Channels2Listen=Channels2listen,n_shots=self.n_shots,n_iter=self.n_iter,bounds=bounds)
        self.optimizer.set_analysis_func(self.ML_analysis_current)
        
        # get one initial measurement
        val_temp=self.optimizer.measure()
        if val_temp>-1e9:
            parameter_dict={}
            distribution_dict={}
            for i in range(len(self.Channels2Opt)):
                parameter_dict[self.Channels2Opt[i]]=self.optimizer.init_vals[i]
                distribution_dict[self.Channels2Opt[i]]=optuna.distributions.FloatDistribution(bounds[i][0],bounds[i][1])
            init_trial=optuna.trial.create_trial(params=parameter_dict, distributions=distribution_dict, value=val_temp)
            self.optimizer.Optimizer.add_trial(init_trial)
        
        # plot the initial measurment
        [freq,spec_list,inten_list]=convert_data(self.optimizer.rawdata)
        self.init_freq=np.array(freq)
        self.init_spec=np.array(spec_list)
        self.init_inten=np.array(inten_list)
        [init_time,init_avg1,init_avg2]=FFT_avgs(self.init_freq,self.init_spec)
        
        self._ax[0].clear()
        self._ax[1].clear()
        self._ax[2].clear()
        extent=[self.init_freq[0],self.init_freq[-1],0,len(self.init_spec)]
        ax0imag=self._ax[0].imshow(self.init_spec,aspect='auto',cmap='jet',extent=extent)
        self._ax[1].plot(self.init_freq,np.mean(self.init_spec,axis=0))
        self._ax[2].plot(init_time,init_avg1,label='|<A>|')
        self._ax[2].plot(init_time,init_avg2,label='<|A|>')
        self._ax[2].set_xlim([-1,np.max(init_time)])
        self._ax[2].legend(loc=1)
        
        # axis
        self._ax[0].set_xlabel('Photon energy (eV)')
        self._ax[0].set_ylabel('Number of shots')
        self._ax[1].set_xlabel('Photon energy (eV)')
        self._ax[1].set_ylabel('Intensity (a. u.)')
        self._ax[2].set_xlabel('Time (fs)')
        self._ax[2].set_ylabel('Intensity (a. u.)')
        
        # update
        ax0imag.figure.canvas.draw_idle()
        return
    
    def ML_analysis_current(self,data):
        return ML_analysis(data,self.method,self.Channels2listen)
    
    def go_optimial(self):
        self.optimizer.set2best()
    
    def revert_init(self):
        self.optimizer.revert()
    
    def opt_start(self):
        self.optimizer.run_optimization(n_iter=self.n_iter)
        return
        
        
        
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ApplicationWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
    