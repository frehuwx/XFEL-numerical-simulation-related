from phase_shifter_opt_noGUI import *


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
        shifter_PV_list=phase_shifter_PVs[self.idx_start:self.idx_end]
        self.optimizer=bayesian_opt(input_namelist=shifter_PV_list,freq_channel=[],spec_channel=[],n_shots=self.n_shots,n_iter=self.n_iter)
        # get one initial measurement
        self.optimizer.record_data()
        self.init_freq=np.array(self.optimizer.freq)
        self.init_spec=np.array(self.optimizer.spec)
        [init_time,init_avg1,init_avg2]=FFT_avgs(self.init_freq,self.init_spec)
        
        # plot the initial measurment
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
    
    def opt_start(self):
        return
        
        
        
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ApplicationWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()