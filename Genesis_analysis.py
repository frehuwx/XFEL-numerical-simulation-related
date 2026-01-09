# deal with the output file of Genesis
import h5py as hdf5
import matplotlib.pyplot as plt
import numpy as np
import math
import ipywidgets as widgets
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter


# general interactive plot func
def update_plot(n_step, x_axis, slider_axis, data_list, xlabel, ylabel_list):
    plt.figure()
    for i in range(len(data_list)):
        data2plot=data_list[i][n_step]
        plt.plot(x_axis,data2plot,label=ylabel_list[i])
    plt.xlabel(xlabel)
    plt.title('%.2f'%slider_axis[n_step])
    plt.legend()
    plt.show()
    return

# a class for plotting animations
class aniplots2d():
    # initialize
    def __init__(self, ax, x_axis=[], y_axis=[], data_list=[], xlabel='', ylabel='', datalabel=[]):
        self.ax=ax
        self.x_axis=x_axis
        self.y_axis=y_axis
        self.data_list=data_list
        self.xlabel=xlabel
        self.ylabel=ylabel
        self.datalabel=datalabel
        lines=[]
        maxs=[]
        mins=[]
        for i in range(len(data_list)):
            line_temp=Line2D(x_axis, data_list[i][0],color='C'+str(i))
            maxs.append(np.max(data_list[i][0]))
            mins.append(np.min(data_list[i][0]))
            lines.append(line_temp)
            self.ax.add_line(line_temp)
            
        len_xaxis=np.max(x_axis)-np.min(x_axis)
        self.ax.set_xlim([np.min(x_axis)-len_xaxis*0.1,np.max(x_axis)+len_xaxis*0.1])
        
        ysmallest=min(mins)
        ybiggest=max(maxs)
        self.ax.set_ylim([ysmallest-(ybiggest-ysmallest)*0.05,ybiggest+(ybiggest-ysmallest)*0.05])
        
        self.lines=lines
        self.ax.legend(self.lines, self.datalabel)
        
    # update function
    def update_lines(self, n_step):
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        maxs=[]
        mins=[]
        for i in range(len(self.data_list)):
            self.lines[i].set_data(self.x_axis, self.data_list[i][n_step])
            maxs.append(np.nanmax(self.data_list[i][n_step]))
            mins.append(np.nanmin(self.data_list[i][n_step]))
            
        ysmallest=min(mins)
        ybiggest=max(maxs)
        self.ax.set_ylim([ysmallest-(ybiggest-ysmallest)*0.05,ybiggest+(ybiggest-ysmallest)*0.05])
        
        self.ax.legend(self.lines, self.datalabel)
        self.ax.set_title('%.2f'%(self.y_axis[n_step]))
        return iter(self.lines)


class genesis_output():
    
    # initialize base on FileName
    def __init__(self, FileName):
        
        f=hdf5.File(FileName,'r')
        # major groups
        beam=f.get('Beam')
        lattice=f.get('Lattice')
        field=f.get('Field')
        global_var=f.get('Global')
        meta_var=f.get('Meta')
        # number of dimension
        bunching=np.array(beam['bunching'])
        self.len_z=len(bunching)
        self.len_t=len(bunching[0])
        # z and t axis
        self.z_axis=np.array(lattice['zplot'])
        time=np.array(global_var['slen'])
        self.t_axis=np.linspace(0,time,self.len_t)
        # other para
        self.fileori=f
        self.beam=beam
        self.lattice=lattice
        self.field=field
        self.global_var=global_var
        self.meta_var=meta_var
        
    def clear(self):
        self.f.close()
        return
    
    def get_info(self,marconame):
        if marconame=='beam':
            print(self.beam.keys())
            try:
                print(self.beam['Global'].keys())
            except:
                print('Global doesn\'t exist')
        elif marconame=='lattice':
            print(self.lattice.keys())
        elif marconame=='field':
            print(self.field.keys())
            print(self.field['Global'].keys())
            try:
                print(self.beam['Global'].keys())
            except:
                print('Global doesn\'t exist')
        elif marconame=='global':
            print(self.global_var.keys())
        return
        
    def get_entry(self, entryname,marconame=''):
        if marconame=='beam':
            for keyName in self.beam.keys():
                if keyName==entryname:
                    founded=True
                    data=self.beam[keyName]
                elif keyName=='Global':
                    for keyName_Global in self.beam['Global'].keys():
                        if keyName_Global==entryname:
                            founded=True
                            data=self.beam['Global'][keyName_Global]
        elif marconame=='lattice':
            for keyName in self.lattice.keys():
                if keyName==entryname:
                    founded=True
                    data=self.lattice[keyName]
        elif marconame=='field':
            for keyName in self.field.keys():
                if keyName==entryname:
                    founded=True
                    data=self.field[keyName]
                elif keyName=='Global':
                    for keyName_Global in self.field['Global'].keys():
                        if keyName_Global==entryname:
                            founded=True
                            data=self.field['Global'][keyName_Global]
        elif marconame=='Global':
            for keyName in self.global_var.keys():
                if keyName==entryname:
                    founded=True
                    data=self.global_var[keyName]
        else:        
            founded=False
            for keyName in self.beam.keys():
                if keyName==entryname:
                    founded=True
                    data=self.beam[keyName]
            if not founded:
                for keyName in self.lattice.keys():
                    if keyName==entryname:
                        founded=True
                        data=self.lattice[keyName]
            if not founded:
                for keyName in self.field.keys():
                    if keyName==entryname:
                        founded=True
                        data=self.field[keyName]
            if not founded:
                print('cannot find the requested entry!')
                return []
        return data
    
    # plot 1D image
    def plot1D(self, entryname, marconame='', tag_plot=True):
        data = self.get_entry(entryname,marconame)
        
        x_axis=[]
        if len(data)>0:
            if len(data)==self.len_z:
                x_axis=self.z_axis
                xlabel='z (m)'
            else:
                x_axis=self.t_axis*1e6
                xlabel='bunch (um)'
                
            data=np.reshape(data,np.shape(x_axis))
        
        if tag_plot:
            plt.figure()
            plt.plot(x_axis,data)
            plt.title(entryname)
            plt.xlabel(xlabel)
            plt.show()
        
        return [x_axis,data]
        
    # plot 2D image    
    def plot2D(self, entryname, marconame='', tag_plot=True):
        data = self.get_entry(entryname,marconame)
        
        if tag_plot and len(data)>0:
            extent = [np.min(self.z_axis), np.max(self.z_axis), np.max(self.t_axis), np.min(self.t_axis)]
            plt.figure()
            plt.imshow(data,extent=extent,cmap='jet')
            plt.show()
        
        return [self.z_axis, self.t_axis, data]
    
    # plot 2D image in 1D animation
    def plotani(self, entryname_list, marconame_list=[], xaxis_type='z', tag_plot=True):
        
        data_list=[]
        for i in range(len(entryname_list)):
            if len(marconame_list)>0:
                data_temp=self.get_entry(entryname_list[i],marconame_list[i])
            else:
                data_temp=self.get_entry(entryname_list[i])
            if xaxis_type=='z':
                data_list.append(np.transpose(data_temp))
            else:
                 data_list.append(data_temp)
        
        if tag_plot:
            if xaxis_type=='z':
                widget = widgets.interactive(update_plot,
                                             n_step=widgets.IntSlider(min=0, max=len(self.t_axis)-1, value=0),
                                             x_axis=widgets.fixed(self.z_axis),
                                             slider_axis=widgets.fixed(self.t_axis*1e6),
                                             data_list=widgets.fixed(data_list),
                                             xlabel=widgets.fixed('z (m)'),
                                             ylabel_list=widgets.fixed(entryname_list))
            else:
                widget = widgets.interactive(update_plot,
                                             n_step=widgets.IntSlider(min=0, max=len(self.z_axis)-1, value=0),
                                             x_axis=widgets.fixed(self.t_axis*1e6),
                                             slider_axis=widgets.fixed(self.z_axis),
                                             data_list=widgets.fixed(data_list),
                                             xlabel=widgets.fixed('bunch (um)'), 
                                             ylabel_list=widgets.fixed(entryname_list))
            display(widget)
        
        return [self.z_axis, self.t_axis, data_list]
    
    # output 1d animation
    def outputani(self, entryname_list, marconame_list=[], xaxis_type='z', ylabel='', output_name='', total_time=15):
        [z_axis, t_axis, data_list]=self.plotani(entryname_list, marconame_list, xaxis_type, tag_plot=False)
        fig, ax = plt.subplots()
        if xaxis_type=='z':
            aniplot_z=aniplots2d(ax=ax, x_axis=self.z_axis, y_axis=self.t_axis*1e6, data_list=data_list, xlabel='z (m)', ylabel=ylabel, datalabel=entryname_list)
            ani = FuncAnimation(fig=fig, func=aniplot_z.update_lines, frames=len(self.t_axis), interval=10, blit=True)
            ani.save(output_name, writer=PillowWriter(fps=60))
        else:
            aniplot_t=aniplots2d(ax=ax, x_axis=self.t_axis*1e6, y_axis=self.z_axis, data_list=data_list, xlabel='bunch (um)', ylabel=ylabel, datalabel=entryname_list)
            ani = FuncAnimation(fig=fig, func=aniplot_t.update_lines, frames=len(self.z_axis), interval=10, blit=True)
            ani.save(output_name, writer=PillowWriter(fps=60))
        return
    
    # output power sum (used for nansum)
    def total_power(self):
        # get the power
        [z_axis,t_axis,power_2d]=self.plot2D('power','field',False)
        # t_axis delt
        delt=(t_axis[-1]-t_axis[0])/len(t_axis)/3e8
        # sum
        power_sum=np.nansum(power_2d*delt,1)
        return [z_axis,power_sum]
    
    # output power sum (sliced) 
    def sliced_power(self,trange=[],tag_um=True,tag_plot=False):
        # get the power
        [z_axis,t_axis,power_2d]=self.plot2D('power','field',False)
        # t_axis delt
        delt=(t_axis[-1]-t_axis[0])/(len(t_axis)-1)/3e8
        delx=(t_axis[-1]-t_axis[0])/(len(t_axis)-1)*1e6
        # get the index range
        idx_l=0
        idx_r=-1
        if len(trange)==2:
            if tag_um:
                idx_l=int((trange[0]-self.t_axis[0]*1e6)/delx)
                idx_r=int((trange[1]-self.t_axis[0]*1e6)/delx)
            else:
                idx_l=int((trange[0]-self.t_axis[0])/delt)
                idx_r=int((trange[1]-self.t_axis[0])/delt)
        # get the sum
        power_2d_trim=power_2d[:,idx_l:idx_r]
        power_sum=np.nansum(power_2d_trim*delt,1)
        # plot the end
        if tag_plot:
            plt.figure()
            plt.plot(t_axis,power_2d[-1],label='full')
            plt.plot(t_axis[idx_l:idx_r],power_2d_trim[-1],label='trimmed')
            plt.legend()
            plt.show()
        return [z_axis,power_sum]
    
    # output required spectrum along z axis
    def spectrum_single(self,trange=[],idx_z=-1,entry='farfield',padding=0,Hanns_window=False,debug=False,cali=True):
        # get the near / far field
        if entry=='farfield':
            [z_axis,t_axis,intensity]=self.plot2D('intensity-farfield','field',False)
            [z_axis,t_axis,phase]=self.plot2D('phase-farfield','field',False)
        else:
            [z_axis,t_axis,intensity]=self.plot2D('intensity-nearfield','field',False)
            [z_axis,t_axis,phase]=self.plot2D('phase-nearfield','field',False)
        intensity_tar=intensity[idx_z]
        phase_tar=phase[idx_z]
        time_axis=self.t_axis/3e8 # in s
        delt=(time_axis[-1]-time_axis[0])/(len(time_axis)-1)
        # get the range of interest
        if len(trange)>0:
            idx_start=int((trange[0]-self.t_axis[0])/(self.t_axis[1]-self.t_axis[0]))
            idx_end=int((trange[1]-self.t_axis[0])/(self.t_axis[1]-self.t_axis[0]))
            if idx_start>idx_end or idx_start<0 or idx_end>len(time_axis):
                print('warning: the input range could be wrong!')
            intensity_tar=intensity_tar[idx_start:idx_end]
            phase_tar=phase_tar[idx_start:idx_end]
            time_axis=time_axis[idx_start:idx_end]
        # build the complex field
        power_c=[]
        for i in range(len(intensity_tar)):
            E_temp=np.sqrt(intensity_tar[i])
            phase_temp=phase_tar[i]
            power_c.append(E_temp*np.cos(phase_temp)+E_temp*np.sin(phase_temp)*1j)
        power_c=np.array(power_c)
        # if hanns, add window
        if Hanns_window:
            hann_func=signal.windows.hann(len(power_c))
            power_c=power_c*hann_func
        # if padding, add zeros
        if padding>1:
            lent=len(time_axis)
            time_axis_tot=np.linspace(time_axis[0]-int(lent*padding)*delt,time_axis[-1]+int(lent*padding)*delt,int(lent*(2*padding+1)))
            power_zeros=np.zeros(int(padding*lent),dtype='complex')
            power_c_tot=np.concatenate((power_zeros,np.concatenate((power_c,power_zeros))))
        else:
            time_axis_tot=time_axis
            power_c_tot=power_c
        # switch to frequency domain
        numt=len(time_axis_tot)
        if numt%2==0:
            f_grid=1/delt/numt*np.linspace(-numt/2,numt/2-1,numt)
        else:
            f_grid=1/delt/numt*np.linspace(-numt/2,numt/2,numt)
        if not debug:
            sig_freq=np.fft.fftshift(np.fft.fft(np.fft.ifftshift(power_c_tot)))
        else:
            sig_freq=np.fft.fftshift(np.fft.fft((power_c_tot)))
        # calibrate x axis from 1/s to eV
        h=6.62607e-34
        e=1.60218e-19
        f_grid=f_grid*h/e
        # calibrate y aixs to sqrt{J/eV}
        if cali:
            power_freq=np.abs(sig_freq)**2
            power_freq_sum=np.sum(power_freq)*(f_grid[1]-f_grid[0])
            [z_axis,t_axis,power]=self.plot2D('power','field',False)
            power_time_sum=np.sum(np.abs(power[idx_z])*delt)
            cali_factor=np.sqrt(power_time_sum/power_freq_sum)
            sig_freq=sig_freq*cali_factor
        return [f_grid,sig_freq]
    
    
    # spectrum evolve through beamtime
    # returns [(1),(2),(3)], (1): z axis along beamline, (2): frequency, (3): 2d spectrum array
    def spectrum_evolve(self,trange=[]):
        # get the z axis
        lenz=len(self.z_axis)
        # loop
        sig_freq_list=[]
        n_finished=0
        for i in range(lenz):
            [f_grid,sig_freq_temp]=self.spectrum_single(trange=trange,idx_z=i)
            sig_freq_list.append(sig_freq_temp)
            if i%(int(lenz/10))==0:
                print('%d%%'%(n_finished*10), end=' ')
                n_finished+=1
        print('done')
        sig_freq_list=np.array(sig_freq_list)
        return [self.z_axis,np.array(f_grid),sig_freq_list]
        
    # for correlation study t1 - t1+delt , t2 - t2+delt
    # g(t1,t2)=sum(E(t1)E*(t2))/sqrt(sum|E(t1)|^2)/sqrt(sum|E(t2)|^2)
    def correlation_single(self,t1,t2,delt,idx_z=-1,tag_um=True,tag_plot=False):
        # get the correct bunch axis
        t_axis_cor=np.array(self.t_axis)
        if tag_um:
            t_axis_cor=t_axis_cor*1e6
        else:
            t_axis_cor=t_axis_cor/3e8
        # get the correct index range
        idx_t1_left=int((t1-t_axis_cor[0])*(len(t_axis_cor)-1)/(t_axis_cor[-1]-t_axis_cor[0]))
        idx_t1_right=int((t1+delt-t_axis_cor[0])*(len(t_axis_cor)-1)/(t_axis_cor[-1]-t_axis_cor[0]))
        idx_t2_left=int((t2-t_axis_cor[0])*(len(t_axis_cor)-1)/(t_axis_cor[-1]-t_axis_cor[0]))
        idx_t2_right=idx_t2_left+idx_t1_right-idx_t1_left
        # get the correct trace
        [z_axis,t_axis,intensity_far]=self.plot2D('intensity-farfield','field',False)
        [z_axis,t_axis,phase_far]=self.plot2D('phase-farfield','field',False)
        intensity_wanted=intensity_far[idx_z]
        phase_wanted=phase_far[idx_z]
        E1=intensity_wanted[idx_t1_left:idx_t1_right]*np.cos(phase_wanted[idx_t1_left:idx_t1_right])+intensity_wanted[idx_t1_left:idx_t1_right]*np.sin(phase_wanted[idx_t1_left:idx_t1_right])*1j
        E2=intensity_wanted[idx_t2_left:idx_t2_right]*np.cos(phase_wanted[idx_t2_left:idx_t2_right])+intensity_wanted[idx_t2_left:idx_t2_right]*np.sin(phase_wanted[idx_t2_left:idx_t2_right])*1j
        # plot
        if tag_plot:
            plt.figure()
            if tag_um:
                plt.plot(t_axis_cor[idx_t1_left:idx_t1_right],abs(E1),label='pulse 1')
                plt.plot(t_axis_cor[idx_t2_left:idx_t2_right],abs(E2),label='pulse 2')
                plt.xlabel('z along bunch (um)')
            else:
                plt.plot(t_axis_cor[idx_t1_left:idx_t1_right]*1e15,abs(E1),label='pulse 1')
                plt.plot(t_axis_cor[idx_t2_left:idx_t2_right]*1e15,abs(E2),label='pulse 2')
                plt.xlabel('time along bunch (fs)')
            plt.legend()
            plt.show()
        # get the correlation
        g12=np.sum(E1*np.conj(E2))/np.sqrt(np.sum(abs(E1*np.conj(E1)))*np.sum(abs(E2*np.conj(E2))))
        return g12
    def correlation_evolve(self,t1,t2,delt,tag_um=True):
        # get the z axis
        lenz=len(self.z_axis)
        # loop
        g12_list=[]
        n_finished=0
        for i in range(lenz):
            g12_list.append(self.correlation_single(t1,t2,delt,idx_z=i,tag_um=tag_um))
            if i%(int(lenz/10))==0:
                print('%d%%'%(n_finished*10), end=' ')
                n_finished+=1
        print('done')
        g12_list=np.array(g12_list)
        return [self.z_axis,g12_list]
        