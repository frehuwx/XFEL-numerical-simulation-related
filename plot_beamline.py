import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import SDDS_class as SDDS
import sys

################################################################
# data process part
################################################################

# manage the elements
element_types=['indicator','drift','solenoid','kicker','quad','sex','bend','rf-acc','rf-df','collimator','wake']
class elegant_element():
    # initialization
    def __init__(self,element_name,element_type,pos):
        # save results
        self.type_full=element_type
        self.name=element_name
        self.pos=pos
        # get the type of the element
        if element_type in ['MARK','CHARGE','CENTER','WATCH','MALIGN','MONI']:
            self.type='indicator'
        elif 'QUAD' in element_type:
            self.type='quad'
        elif element_type=='RFDF':
            self.type='rf-df'
        elif 'RF' in element_type:
            self.type='rf-acc'
        elif 'SEXT' in element_type:
            self.type='sex'
        elif 'BEND' in element_type:
            self.type='bend'
        elif 'KICK' in element_type:
            self.type='kicker'
        elif 'WAKE' in element_type:
            self.type='wake'
        elif element_type=='MAXAMP':
            self.type='collimator'
        elif 'DRIFT' in element_type:
            self.type='drift'
        elif element_type=='SOLE':
            self.type='solenoid'
        else:
            self.type='other'
            
# merge the elements
def merge_element(element1,element2):
    element_name=element1.name+' & '+element2.name
    element_merge=elegant_element(element_name,element1.type_full,element1.pos)
    return element_merge
            
# sort the elements: given a list of elements, we do the followings
# 1. remove the irrelavent ones (the ones that have the same name and the same position)
# 2. merge the indicators at the same position together
def sort_element(element_list):
    s_sorted=[]
    element_sorted=[]
    idx=0
    snow=0
    while idx<len(element_list):
        # get all the indices at the same position
        snow=element_list[idx].pos
        idx_end=idx
        while idx_end<len(element_list) and np.abs(element_list[idx_end].pos-snow)<1e-4:
            idx_end+=1
        # merge them together and append that into the sorted element
        element_now=element_list[idx]
        for i in np.arange(idx+1,idx_end):
            # merge them if both are indicators
            if element_now.type=='indicator' and element_list[i].type=='indicator':
                element_now=merge_element(element_now,element_list[i])
            # if the types are not the same, we need to seperate them!
            if element_now.type!=element_list[i].type:
                element_sorted.append(element_now)
                s_sorted.append(element_now.pos)
                element_now=element_list[i]
        s_sorted.append(element_now.pos)
        element_sorted.append(element_now) # append the last guy to the list
        # update the indice
        idx=idx_end+1
    return [s_sorted,element_sorted]

def plot_element(ax,element,length=0,name_tag=False,y0=0):
    # if it is a marker type, add an arrow
    if element.type=='indicator':
        ax.add_patch(mpatches.Arrow(element.pos,y0+1,0,-1,color=[0,0,0],width=0.1,zorder=5))
    elif element.type=='drift':
        ax.add_patch(mpatches.Rectangle((element.pos,y0-0.15),length,0.3,color=[0.7,0.7,0.7],alpha=0.7,linewidth=1,zorder=1))
    elif element.type=='quad':
        ax.add_patch(mpatches.Rectangle((element.pos,y0-1),length,2,color=[170/255,74/255,68/255],alpha=0.7,linewidth=1,zorder=4))
    elif element.type=='bend':
        ax.add_patch(mpatches.Rectangle((element.pos,y0-1),length,2,color=[0/255,191/255,255/255],alpha=0.7,linewidth=1,zorder=4))
    elif element.type=='rf-acc':
        ax.add_patch(mpatches.Rectangle((element.pos,y0-0.6),length,1.2,color=[127/255,255/255,212/255],alpha=0.7,linewidth=1,zorder=2))
    elif element.type=='rf-df':
        ax.add_patch(mpatches.Rectangle((element.pos,y0-0.6),length,1.2,color=[191/255,64/255,191/255],alpha=0.7,linewidth=1,zorder=2))
    elif element.type=='kicker':
        if length<0.1:
            ax.add_patch(mpatches.Wedge((element.pos,y0),1.2,268,272,color=[255/255,20/255,147/255],alpha=0.7,linewidth=1,zorder=3))
        else:
            ax.add_patch(mpatches.Rectangle((element.pos,y0-1.2),length,1.2,color=[255/255,20/255,147/255],alpha=0.7,linewidth=1,zorder=3))
    else:
        ax.add_patch(mpatches.Rectangle((element.pos,y0-0.6),length,1.2,color=[250/255,213/255,163/255],alpha=0.7,linewidth=1,zorder=2))
    return ax











################################################################
# GUI part
################################################################
from PyQt6.QtCore import *
from PyQt6.QtCore import Qt
from PyQt6.QtGui import *
#from PyQt6.QtWidgets import QApplication, QCheckBox, QLabel, QMainWindow, QStatusBar, QToolBar, QFileDialog

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # initailize the data
        self.s_sorted=[]
        self.element_sorted=[]
        #####################
        # window geometry
        #####################
        self.setWindowTitle("Elegant lattice viewer")
        self.setGeometry(100, 100, 1000, 900)
        
        #####################
        # toolbar
        #####################
        toolbar = QtWidgets.QToolBar("toolbar")
        self.addToolBar(toolbar)
        # open button
        button_open = QAction("Open", self)
        button_open.setStatusTip("open a .mag file")
        button_open.triggered.connect(self.selectDirectoryDialog)
        toolbar.addAction(button_open)
        
        # main panel 
        self._main = QtWidgets.QWidget()
        layout_main = QtWidgets.QVBoxLayout(self._main)
        #####################
        # plot range
        #####################
        layout_range = QtWidgets.QHBoxLayout()
        label1 = QtWidgets.QLabel('left:')
        label2 = QtWidgets.QLabel('right:')
        self.left_line = QtWidgets.QLineEdit()
        self.left_line.setValidator(QIntValidator())
        self.left_line.setMaxLength(5)
        self.right_line = QtWidgets.QLineEdit()
        self.right_line.setValidator(QIntValidator())
        self.right_line.setMaxLength(5)
        update_button=QtWidgets.QPushButton('update')
        update_button.clicked.connect(self.update_plot_withrange)
        layout_range.addWidget(label1)
        layout_range.addWidget(self.left_line)
        layout_range.addWidget(label2)
        layout_range.addWidget(self.right_line)
        layout_range.addWidget(update_button)
        layout_main.addLayout(layout_range)
        
        #####################
        # Slider
        #####################
        layout_slider = QtWidgets.QHBoxLayout()
        label3 = QtWidgets.QLabel('window:')
        self.window_line = QtWidgets.QLineEdit()
        self.window_line.setValidator(QIntValidator())
        self.window_line.setMaxLength(5)
        self.window_slider=QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.window_slider.sliderMoved.connect(self.update_plot_withslider)
        layout_slider.addWidget(label3)
        layout_slider.addWidget(self.window_line)
        layout_slider.addWidget(self.window_slider)
        layout_main.addLayout(layout_slider)
        
        #####################
        # access specific parts by name
        #####################
        self.number_of_interest=0
        self.idx_of_interest=0
        layout_accessbyname = QtWidgets.QHBoxLayout()
        label4 = QtWidgets.QLabel('channel name:')
        self.name_line = QtWidgets.QLineEdit()
        go_button=QtWidgets.QPushButton('Go to the first one!')
        go_button.clicked.connect(self.update_plot_withchannelname_first)
        next_button=QtWidgets.QPushButton('Next one')
        next_button.clicked.connect(self.update_plot_withchannelname_next)
        prev_button=QtWidgets.QPushButton('Last one')
        prev_button.clicked.connect(self.update_plot_withchannelname_prev)
        layout_accessbyname.addWidget(label4)
        layout_accessbyname.addWidget(self.name_line)
        layout_accessbyname.addWidget(go_button)
        layout_accessbyname.addWidget(next_button)
        layout_accessbyname.addWidget(prev_button)
        layout_main.addLayout(layout_accessbyname)
        
        #####################
        # check box
        #####################
        layout_boxes = QtWidgets.QHBoxLayout()
        self.box_showname = QtWidgets.QCheckBox("show main entrys")
        layout_boxes.addWidget(self.box_showname)
        self.box_showallname = QtWidgets.QCheckBox("incl. other entrys")
        layout_boxes.addWidget(self.box_showallname)
        self.box_showrepname = QtWidgets.QCheckBox("incl. repeated names")
        layout_boxes.addWidget(self.box_showrepname)
        layout_main.addLayout(layout_boxes)
        
        #####################
        # plot
        #####################
        lattice_layout = FigureCanvas(Figure(figsize=(5, 3),layout='constrained'))
        layout_main.addWidget(lattice_layout)
        self._ax = lattice_layout.figure.subplots()
        self.xleft=0
        self.xright=5
        self.s_sorted=[]
        self.element_sorted=[]
        # plot main panel
        self._main.setLayout(layout_main)
        self.setCentralWidget(self._main)

    def selectDirectoryDialog(self):
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setWindowTitle("Select file")
        file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
        file_dialog.setViewMode(QtWidgets.QFileDialog.ViewMode.List)

        if file_dialog.exec():
            # open file
            selected_directory = file_dialog.selectedFiles()[0]
            lattice_sdds=SDDS.SDDS(selected_directory)
            print('opening file: '+selected_directory)
            lattice_sdds.getinfo()
            s=lattice_sdds.getdata('s')
            ElementName=lattice_sdds.getdata('ElementName')
            ElementType=lattice_sdds.getdata('ElementType')
            # sort the result
            element_list=[]
            for i in range(len(s)):
                element_list.append(elegant_element(ElementName[i],ElementType[i],s[i]))
            [self.s_sorted, self.element_sorted] = sort_element(element_list)
            self.s_sorted=np.array(self.s_sorted)
            self.xleft=np.min(self.s_sorted)
            self.xright=np.max(self.s_sorted)
            self.update_plot()
            # limit the slider range
            self.window_slider.setMinimum(int(np.min(self.s_sorted)-2))
            self.window_slider.setMaximum(int(np.max(self.s_sorted)+2))
            
    def update_plot_withrange(self):
        try:
            self.xleft=int(self.left_line.text())
            self.xright=int(self.right_line.text())
            if self.xright>self.xleft:
                self.update_plot()
            else:
                self.xleft=0
                self.xright=2
                self.update_plot()
        except:
            self.xleft=0
            self.xright=2
            self.update_plot()
            
    def update_plot_withslider(self,slider_pos):
        # update the range of interest
        try:
            self.xleft=int(slider_pos)
            self.xright=self.xleft+int(self.window_line.text())
            self.update_plot()
        except:
            self.xleft=int(slider_pos)
            self.xright=self.xleft+5
            self.update_plot()
            
    def update_plot_withchannelname_first(self):
        channelname=self.name_line.text()
        # if it is the first time, we need to find the first indices
        self.number_of_interest=0
        idx_founded=0
        while idx_founded<len(self.element_sorted):
            if self.name_line.text() in self.element_sorted[idx_founded].name:
                self.number_of_interest=1
                self.idx_of_interest=idx_founded
                print('Founded! The name is:'+self.element_sorted[idx_founded].name+' at %.1f m'%(self.s_sorted[idx_founded]))
                break
            idx_founded+=1
            if idx_founded>=len(self.element_sorted):
                print('Cannot found the requested entry, go to the beginning...')
        # update the left and right limit
        try:
            if int(self.window_line.text())>0:
                self.xleft=self.s_sorted[idx_founded]-int(self.window_line.text())/2
                self.xright=self.s_sorted[idx_founded]+int(self.window_line.text())/2
                self.update_plot()
        except:
            try:
                self.xleft=self.s_sorted[idx_founded]-2.5
                self.xright=self.s_sorted[idx_founded]+2.5
                self.update_plot()
            except:
                self.xleft=0
                self.xright=5
        return
    def update_plot_withchannelname_next(self):
        channelname=self.name_line.text()
        # otherwise, we seek the next one
        idx_founded=self.idx_of_interest+1
        while idx_founded<len(self.element_sorted):
            if self.name_line.text() in self.element_sorted[idx_founded].name:
                self.number_of_interest+=1
                self.idx_of_interest=idx_founded
                print('Founded! The name is:'+self.element_sorted[idx_founded].name+' at %.1f m'%(self.s_sorted[idx_founded]))
                break
            idx_founded+=1
            if idx_founded>=len(self.element_sorted):
                print('Cannot found the requested entry, go to the beginning...')
        # update the left and right limit
        try:
            if int(self.window_line.text())>0:
                self.xleft=self.s_sorted[idx_founded]-int(self.window_line.text())/2
                self.xright=self.s_sorted[idx_founded]+int(self.window_line.text())/2
                self.update_plot()
        except:
            try:
                self.xleft=self.s_sorted[idx_founded]-2.5
                self.xright=self.s_sorted[idx_founded]+2.5
                self.update_plot()
            except:
                self.xleft=0
                self.xright=5
        return
    def update_plot_withchannelname_prev(self):
        channelname=self.name_line.text()
        # otherwise, we seek the previous one
        idx_founded=self.idx_of_interest-1
        while idx_founded>=0:
            if self.name_line.text() in self.element_sorted[idx_founded].name:
                self.number_of_interest-=1
                self.idx_of_interest=idx_founded
                print('Founded! The name is:'+self.element_sorted[idx_founded].name+' at %.1f m'%(self.s_sorted[idx_founded]))
                break
            idx_founded-=1
            if idx_founded<0:
                print('Cannot found the requested entry, go to the beginning...')
        # update the left and right limit
        try:
            if int(self.window_line.text())>0:
                self.xleft=self.s_sorted[idx_founded]-int(self.window_line.text())/2
                self.xright=self.s_sorted[idx_founded]+int(self.window_line.text())/2
                self.update_plot()
        except:
            try:
                self.xleft=self.s_sorted[idx_founded]-2.5
                self.xright=self.s_sorted[idx_founded]+2.5
                self.update_plot()
            except:
                self.xleft=0
                self.xright=5
        return
        
    def update_plot(self):
        self._ax.clear()
        #####################
        # plot all the elements within ROI
        #####################
        self._layoutplot, = self._ax.plot([np.min(self.s_sorted),np.max(self.s_sorted)],[0,0],'-',color=[0,0,0],linewidth=1)
        self._ax.set_xlim([self.xleft,self.xright])
        self._ax.set_ylim([-2.5,2.5])
        idx_left=np.argmin(np.abs(self.s_sorted-self.xleft))
        idx_right=np.argmin(np.abs(self.s_sorted-self.xright))
        xcen_list=[]
        for idx in np.arange(idx_left,idx_right):
            if idx+1>len(self.element_sorted):
                length=0
                xcen_list.append(self.element_sorted[idx].pos)
            else:
                length=self.element_sorted[idx+1].pos-self.element_sorted[idx].pos
                xcen_list.append(self.element_sorted[idx].pos+length/2)
            self._ax=plot_element(self._ax,self.element_sorted[idx],length=length,y0=0)
        self._ax.set_xlabel('along beamline (m)')
        #####################
        # plot the names
        #####################
        if self.box_showname.isChecked():
            #fontsize=int(5/(self.xright-self.xleft)*10)
            fontsize=9
            i=0
            displayed_channels=[]
            for idx in np.arange(idx_left,idx_right):
                if (self.element_sorted[idx].name not in displayed_channels) or self.box_showrepname.isChecked():
                    if self.element_sorted[idx].type=='indicator' or self.element_sorted[idx].type=='quad' or self.element_sorted[idx].type=='bend' :
                        self._ax.text(xcen_list[i],1.1,self.element_sorted[idx].name,rotation=90,horizontalalignment='center',verticalalignment='bottom',fontsize=fontsize)
                        displayed_channels.append(self.element_sorted[idx].name)
                    elif self.element_sorted[idx].type=='rf-acc' or self.element_sorted[idx].type=='rf-df':
                        self._ax.text(xcen_list[i],-0.7,self.element_sorted[idx].name,rotation=90,horizontalalignment='center',verticalalignment='top',fontsize=fontsize)
                        displayed_channels.append(self.element_sorted[idx].name)
                    elif self.element_sorted[idx].type=='kicker' :
                        self._ax.text(xcen_list[i],-1.3,self.element_sorted[idx].name,rotation=90,horizontalalignment='center',verticalalignment='top',fontsize=fontsize)
                        displayed_channels.append(self.element_sorted[idx].name)
                    elif self.element_sorted[idx].type=='other' or self.element_sorted[idx].type=='sex' or self.element_sorted[idx].type=='wake' or self.element_sorted[idx].type=='solenoid' :
                        self._ax.text(xcen_list[i],-0.7,self.element_sorted[idx].name,rotation=90,horizontalalignment='center',verticalalignment='top',fontsize=fontsize)
                        displayed_channels.append(self.element_sorted[idx].name)
                    elif self.box_showallname.isChecked():
                        if self.element_sorted[idx].type=='drift' :
                            self._ax.text(xcen_list[i],-0.4,self.element_sorted[idx].name,rotation=90,horizontalalignment='center',verticalalignment='top',fontsize=fontsize)
                            displayed_channels.append(self.element_sorted[idx].name)
                        else:
                            self._ax.text(xcen_list[i],-0.7,self.element_sorted[idx].name,rotation=90,horizontalalignment='center',verticalalignment='top',fontsize=fontsize)
                            displayed_channels.append(self.element_sorted[idx].name)
                i+=1
        #####################
        # plot the legends
        #####################
        delx=self.xright-self.xleft
        xbegin=self.xright-delx/2-delx/5
        delx_legend=delx/10
        self._ax.add_patch(mpatches.Rectangle((xbegin,2.2),delx/20,0.1,color=[0.7,0.7,0.7],alpha=0.7,linewidth=1,zorder=1))
        self._ax.add_patch(mpatches.Arrow(xbegin+delx_legend,2.25,delx/20,0,color=[0,0,0],width=0.1,zorder=1))
        self._ax.add_patch(mpatches.Rectangle((xbegin+delx_legend*2,2.2),delx/20,0.1,color=[170/255,74/255,68/255],alpha=0.7,linewidth=1,zorder=1))
        self._ax.add_patch(mpatches.Rectangle((xbegin+delx_legend*3,2.2),delx/20,0.1,color=[0/255,191/255,255/255],alpha=0.7,linewidth=1,zorder=1))
        self._ax.add_patch(mpatches.Rectangle((xbegin+delx_legend*4,2.2),delx/20,0.1,color=[127/255,255/255,212/255],alpha=0.7,linewidth=1,zorder=1))
        self._ax.add_patch(mpatches.Rectangle((xbegin+delx_legend*5,2.2),delx/20,0.1,color=[191/255,64/255,191/255],alpha=0.7,linewidth=1,zorder=2))
        self._ax.add_patch(mpatches.Rectangle((xbegin+delx_legend*6,2.2),delx/20,0.1,color=[255/255,20/255,147/255],alpha=0.7,linewidth=1,zorder=1))
        self._ax.text(xbegin+delx/40,2.35,'Drift',horizontalalignment='center',verticalalignment='bottom',fontsize=9)
        self._ax.text(xbegin+delx/40+delx_legend,2.35,'Marker & Screen',horizontalalignment='center',verticalalignment='bottom',fontsize=9)
        self._ax.text(xbegin+delx/40+delx_legend*2,2.35,'Quad',horizontalalignment='center',verticalalignment='bottom',fontsize=9)
        self._ax.text(xbegin+delx/40+delx_legend*3,2.35,'Bending',horizontalalignment='center',verticalalignment='bottom',fontsize=9)
        self._ax.text(xbegin+delx/40+delx_legend*4,2.35,'RF-accelerator',horizontalalignment='center',verticalalignment='bottom',fontsize=9)
        self._ax.text(xbegin+delx/40+delx_legend*5,2.35,'RF-deflector',horizontalalignment='center',verticalalignment='bottom',fontsize=9)
        self._ax.text(xbegin+delx/40+delx_legend*6,2.35,'Corrector',horizontalalignment='center',verticalalignment='bottom',fontsize=9)
        
        # draw
        self._layoutplot.figure.canvas.draw_idle()
        plt.show()
        
        
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ApplicationWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
