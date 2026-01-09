# to process the data between SDDS, hdf5, and dict

# imports
import sys
import h5py as hdf5
import matplotlib.pyplot as plt
import numpy as np
import struct
from datetime import datetime

####################################################################################################################
# SDDS file
# list of components in the header:
# Header_list=['description','column','parameter','array','include','data']
# description_list=['text','contents']
# column_list=['name','symbol','units','description','format_string','type','field_length']
# parameter_list=['name','symbol','units','description','format_string','type','fixed_value']
# array_list=['name','symbol','units','description','format_string','type','group_name','field_length','dimensions']
# include_list=['filename']
# data_list=['mode','lines_per_row','no_row_counts','additional_header_lines']
####################################################################################################################


####################################################################################################################
# some helper function to phrase the header
####################################################################################################################

# change an binary array read by the file to a str
def biarray2str(input_arr):
    result=[]
    for i in range(len(input_arr)):
        result.append(chr(input_arr[i]))
    return ''.join(result)
# change a str into a binary array
def str2biarray(input_str):
    result=[]
    for i in range(len(input_str)):
        result.append(ord(input_str[i]))
    return result

# seperate the string of the header
def separate_str(input_str):
    whitespace_list=[ord(','),ord('\n'),ord(' '),ord('='),ord('!')]
    str_list=[]
    tag_start=False
    tag_quote=False
    idx_str=0
    while idx_str<len(input_str):
        char_temp=input_str[idx_str]
        if char_temp==ord('"') and tag_quote==False:
            tag_quote=True
            idx_ref=idx_str+1
        elif tag_quote:
            if char_temp==ord('"'):
                str_list.append(biarray2str(input_str[idx_ref:idx_str]))
                tag_quote=False
        else:
            if tag_start:
                if char_temp in whitespace_list:
                    str_list.append(biarray2str(input_str[idx_ref:idx_str]))
                    tag_start=False
            else:
                if not char_temp in whitespace_list:
                    tag_start=True
                    idx_ref=idx_str
            if char_temp==ord('!'): # remove the comment
                idx_str=len(input_str)
        idx_str+=1
    return str_list

# a function that phrases components in the header
def phrase_component(pfile):
    tag_start=False
    tag_end=False
    component_type=[]
    component_dict=dict()
    while not tag_end:
        # read line by line (one component >=1 lines)
        line_temp=pfile.readline()
        strlist_temp=separate_str(line_temp)
        i=0
        while i<len(strlist_temp):
            if strlist_temp[i]=='&end':
                tag_end=True
            elif tag_start==False:
                tag_start=True
                component_type=strlist_temp[i][1:]
            else:
                component_dict[strlist_temp[i]]=strlist_temp[i+1]
                i+=1
            i+=1
    return [component_type,component_dict]

# a function that convert the data
def data_convert(data_type, data_val):
    if data_type=='long' or data_type=='short':
        return int(data_val)
    #else, if it is float or double, convert into float
    elif data_type=='float' or data_type=='double':
        return float(data_val)
    #else, it is a string (get rid of \n)
    else:
        return data_val

# a function that reads the next binary data
# alway use it to read one data
# if the type is string, there will be a long signed value before the string to indicate the number of bytes
def data_read_binary(pfile, data_type, char_endian):
    num2read=0
    if data_type=='string':
        # read the number first
        format_str=char_endian+'l'
        num2read=4
        lenstr=pfile.read(num2read)
        lenstr=struct.unpack(format_str,lenstr)
        lenstr=lenstr[0]
        # read the actual data
        array_readed=pfile.read(lenstr)
        return biarray2str(array_readed)
    elif data_type=='long':
        format_str=char_endian+'l'
        num2read=4
    elif data_type=='short':
        format_str=char_endian+'h'
        num2read=2
    elif data_type=='float':
        format_str=char_endian+'f'
        num2read=4
    elif data_type=='double':
        format_str=char_endian+'d'
        num2read=8
    elif data_type=='long64':
        format_str=char_endian+'q'
        num2read=8
    elif data_type=='ulong64':
        format_str=char_endian+'Q'
        num2read=8
    data_temp=pfile.read(num2read)
    data_unpacked=struct.unpack(format_str,data_temp)
    return data_unpacked[0]

# a function that writes the next binary data
# if the type is string, there will be a long signed value before the string to indicate the number of bytes
def data_write_binary(pfile, data_type, data, char_endian):
    num2write=0
    if data_type=='string':
        # write the number first
        lenstr=len(data)
        format_str=char_endian+'l'
        data_packed=struct.pack(format_str,lenstr)
        pfile.write(data_packed)
        # write the string
        pfile.write(bytearray(str2biarray(data)))
        return
    elif data_type=='long':
        format_str=char_endian+'l'
    elif data_type=='short':
        format_str=char_endian+'h'
    elif data_type=='float':
        format_str=char_endian+'f'
    elif data_type=='double':
        format_str=char_endian+'d'
    elif data_type=='long64':
        format_str=char_endian+'q'
    elif data_type=='ulong64':
        format_str=char_endian+'Q'
    data_packed=struct.pack(format_str,data)
    pfile.write(data_packed)
    return

# a function that writes one dict for the header
def writeheader_dict(pfile,dict_type,tar_dict):
    # head
    pfile.write('&'+dict_type+' ')
    # entries
    for keyName in tar_dict.keys():
        if keyName != 'value': # exclude the actual values
            data2write=tar_dict[keyName]
            data2write=str(data2write)
            tag_quote=False
            if ' ' in data2write or '!' in data2write: # we need quotation in this case
                tag_quote=True
            if tag_quote:
                pfile.write(keyName+'=\"'+data2write+'\", ')
            else:
                pfile.write(keyName+'='+data2write+', ')
    # end
    pfile.write('&end\n')
    

class SDDS(dict):
    
####################################################################################################################
# SDDS class:
# input: direct initialize(file), build_from_file(file), phrase_header_from_file(file)
# output: write2file(file), write2hdf5_std(file), write2hdf5_fmt(file,list_datanames,list_entries)
# info: getinfo(), getentryinfo(name)
# data access: getdict(name), getdata(name), setdata(name,data)
# others: clear()
####################################################################################################################
    
    # setup SDDS data (each component is a dict)
    def __init__(self, filename='', description=dict(), data=dict(), parameter=[], array=[], column=[]):
        self.clear()
        self['description']=description
        self.parameter=parameter
        self.array=array
        self.column=column
        for keyName in data.keys():
            self['data'][keyName]=data[keyName]
        if len(filename)>0:
            self.build_from_file(filename)
            
    # clear data, setting everything to default
    def clear(self):
        self['description']=dict()
        self['data']=dict()
        self.parameter=[]
        self.array=[]
        self.column=[]
        self['data']['mode']='binary'
        self['data']['lines_per_row']=1
        self['data']['no_row_counts']=0
        self['data']['additional_header_lines']=0
        self['data']['endian']='little'
    
    # setup the SDDS data based on existing file
    def build_from_file(self, filename=''):
        # use the header to assign the data first
        f_open=self.phrase_header_from_file(filename)
        # collect the data
        ###############################################################
        # 1. ascii case
        ###############################################################
        if self['data']['mode']=='ascii':
            # 1.1 remove extra comment lines
            if self['data']['additional_header_lines']>0: # case 1
                for i in range(self['data']['additional_header_lines']):
                    f_open.readline()
            else: # case 2 search for '!' lines
                marker_0=f_open.tell() # intial pointer
                line_offset=0
                tag_start=False
                while not tag_start:
                    str_temp=f_open.readline()
                    idx_str=0
                    while str_temp[idx_str]==ord(' '):
                        idx_str+=1
                    if str_temp[idx_str]==ord('!'):
                        line_offset+=1
                    else:
                        tag_start=True
                f_open.seek(marker_0,0)
                while line_offset>0:
                    f_open.readline()
                    line_offset-=1
            # 1.2 parameters: each parameter is stored in a line
            # note: if there is fixed_value, then won't appear here
            for i in range(len(self.parameter)):
                parameter_temp=self.parameter[i]
                list_keyName=parameter_temp.keys()
                if 'fixed_value' not in list_keyName:
                    line_temp=f_open.readline()
                    self.parameter[i]['value']=data_convert(parameter_temp['type'], line_temp)
                else:
                    self.parameter[i]['value']=data_convert(parameter_temp['type'], self.parameter[i]['fixed_value'])
            # 1.3 array: dimension line + each member is stored in a line
            for i in range(len(self.array)):
                array_temp=self.array[i]
                # read the dimension line
                line_temp=f_open.readline()
                str_list=separate_str(line_temp)
                if len(str_list)!=int(array_temp['dimensions']):
                    print('warning: array dimensions don\'t match')
                array_dim=[]
                total_num=1
                for j in range(len(str_list)):
                    len_dim=int(str_list[j])
                    array_dim.append(len_dim)
                    total_num=total_num*len_dim
                # get the data (1D)
                data_raw=[]
                for j in range(total_num):
                    line_temp=f_open.readline()
                    data_raw.append(data_convert(array_temp['type'], line_temp))
                self.array[i]['value']=np.reshape(data_raw,array_dim)
            # 1.4 column: 
            # if no_row_counts<=0, then there is a line indicating the number of lines
            # if no_row_counts>0, then there is an empty line for end indication
            if len(self.column)>0:
                for i in range(len(self.column)):
                    self.column[i]['value']=[]
                if self['data']['no_row_counts']<=0:
                    # first line: number of points
                    line_temp=f_open.readline()
                    str_list=separate_str(line_temp)
                    num_rows=int(str_list[0])
                    # read the rest data
                    for i in range(num_rows):
                        line_temp=f_open.readline()
                        str_list=separate_str(line_temp)
                        for j in range(len(self.column)):
                            self.column[j]['value'].append(data_convert(self.column[j]['type'], str_list[j]))
                else:
                    line_temp=f_open.readline()
                    str_list=separate_str(line_temp)
                    while len(str_list)==len(self.column):
                        for j in range(len(self.column)):
                            self.column[j]['value'].append(data_convert(self.column[j]['type'], str_list[j]))
                        line_temp=f_open.readline()
                        str_list=separate_str(line_temp)
                
        ###############################################################
        # 2. binary case
        ###############################################################
        else:
            char_endian='<' # default little
            if self['data']['endian']=='big':
                char_endian='>'
            # use struct for every data
            # 2.1 remove extra comment lines
            for i in range(self['data']['additional_header_lines']):
                f_open.readline()
            # 2.2 read first data (number of rows for the column data)
            num_rows=f_open.read(4)
            num_rows=struct.unpack(char_endian+'l',num_rows)
            num_rows=num_rows[0]
            # 2.3 read parameters
            # note: if there is fixed_value, then won't appear here
            for i in range(len(self.parameter)):
                parameter_temp=self.parameter[i]
                list_keyName=parameter_temp.keys()
                if 'fixed_value' not in list_keyName:
                    self.parameter[i]['value']=data_read_binary(f_open,parameter_temp['type'],char_endian)
                else:
                    self.parameter[i]['value']=data_convert(parameter_temp['type'], self.parameter[i]['fixed_value'])
            # 2.4 read array
            for i in range(len(self.array)):
                # first there will n long integers based on the dimension
                array_temp=self.array[i]
                n_dim=int(array_temp['dimensions'])
                array_dim=[]
                total_num=1
                for j in range(n_dim):
                    len_dim=f_open.read(4)
                    len_dim=struct.unpack(char_endian+'l',len_dim)
                    len_dim=len_dim[0]
                    array_dim.append(len_dim)
                    total_num=total_num*array_dim
                # next read total_num items and form 1D array
                data_raw=[]
                for j in range(total_num):
                    data_raw.append(data_read_binary(f_open,array_temp['type'],char_endian))
                # finally resize the data
                self.array[i]['value']=np.reshape(data_raw,array_dim)
            # 2.5 read column
            if len(self.column)>0:
                for i in range(len(self.column)):
                        self.column[i]['value']=[]
                for i in range(num_rows):
                    for j in range(len(self.column)):
                        column_temp=self.column[j]
                        self.column[j]['value'].append(data_read_binary(f_open,column_temp['type'],char_endian))
        
    # a function that reads the header
    def phrase_header_from_file(self, filename=''):
        # clear the data first for safty
        self.clear()
        # open the file
        f_open=open(filename,'rb')
        # 1st line should always be SDDSx
        line0=str(f_open.readline())
        if 'SDDS' not in line0:
            print('error: only SDDS file is spported, while the file is '+line0)
            return
        # now phrasing all the components:
        tag_header_end=False
        while not tag_header_end:
            # if there is 'include' component, then phrase the header in the corresponding file
            [component_type_temp, component_dict_temp]=phrase_component(f_open)
            if component_type_temp=='include':
                [f_include,description,data,parameter,array,column]=phrase_header(component_dict_temp['filename'])
                tag_header_end=True
            else:
                if component_type_temp=='description':
                    self['description']=component_dict_temp
                elif component_type_temp=='data': # data always marks the end of header
                    for keyName in component_dict_temp.keys():
                        self['data'][keyName]=component_dict_temp[keyName]
                    # change to int data for convenience    
                    self['data']['lines_per_row']=int(self['data']['lines_per_row'])
                    self['data']['no_row_counts']=int(self['data']['no_row_counts'])
                    self['data']['additional_header_lines']=int(self['data']['additional_header_lines'])
                    tag_header_end=True
                elif component_type_temp=='parameter':
                    self.parameter.append(component_dict_temp)
                elif component_type_temp=='array':
                    self.array.append(component_dict_temp)
                elif component_type_temp=='column':
                    self.column.append(component_dict_temp)
        return f_open
    
    # output the SDDS data to a SDDS file
    def write2file(self, filename='', fileformat=''):
        if len(filename)<=0:
            print('warning: no filename is specified')
            return
        else:
            # if not specified, use the stored format
            if fileformat!='ascii' and fileformat!='binary':
                    fileformat=self['data']['mode']
            # create the header:
            f_write=open(filename,'w')
            # write SDDS5
            f_write.write('SDDS5\n')
            # write some more info
            time_now=datetime.now()
            f_write.write('! created at '+ time_now.strftime("%m/%d/%Y, %H:%M:%S") +' using Python.\n')
            # write description
            writeheader_dict(f_write,'description',self['description'])
            # write parameters
            for i in range(len(self.parameter)):
                writeheader_dict(f_write,'parameter',self.parameter[i])
            # write array
            for i in range(len(self.array)):
                writeheader_dict(f_write,'array',self.array[i])
            # write column
            for i in range(len(self.column)):
                writeheader_dict(f_write,'column',self.column[i])
            # write data (change the mode accordingly)
            datadict2write=self['data']
            datadict2write['mode']=fileformat
            writeheader_dict(f_write,'data',self['data'])
            # save the header
            f_write.close()
            ############################################################
            # 1. ascii case
            ############################################################
            if fileformat=='ascii':
                f_write=open(filename,'a')
                # write data
                # 1.1 write parameters
                # note: fixed_value paras won't appear here
                for i in range(len(self.parameter)):
                    parameter_temp=self.parameter[i]
                    list_keyName=parameter_temp.keys()
                    if 'fixed_value' not in list_keyName:
                        data2write=parameter_temp['value']
                        f_write.write(str(data2write)+'\n')
                # 1.2 write array
                for i in range(len(self.array)):
                    array_temp=self.array[i]
                    # write the shape first
                    data2write=array_temp['value']
                    shape2write=np.shape(data2write)
                    total_num=1
                    for j in range(len(shape2write)):
                        f_write.write(str(shape2write[j]))
                        total_num=total_num*shape2write[j]
                        if j+1<len(shape2write):
                            f_write.write(' ')
                        else:
                            f_write.write('\n')
                    # reshape the data to 1d
                    data2write=np.reshape(data2write,total_num)
                    for j in range(len(data2write)):
                        f_write.write(str(data2write[j])+'\n')
                # 1.3 write column
                if len(self.column)>0:
                    num_rows=len(self.column[0]['value'])
                    # with row counts: add one row
                    if self['data']['no_row_counts']<=0:
                        f_write.write(str(num_rows)+'\n')
                    for i in range(num_rows):
                        for j in range(len(self.column)):
                            data2write=self.column[j]['value']
                            datatype=self.column[j]['type']
                            data2write=data2write[i]
                            if datatype=='string':
                                f_write.write('"'+str(data2write)+'"')
                            else:
                                f_write.write(str(data2write))
                            if j+1<len(self.column):
                                f_write.write(' ')
                            else:
                                f_write.write('\n')
                    # without: add an empty row at the end
                    if self['data']['no_row_counts']>0:
                        f_write.write('\n')
                f_write.close()
                print(filename+' saved')
                return
            
            ############################################################
            # 2. binary case
            ############################################################
            elif fileformat=='binary':
                char_endian='<' # default little
                if self['data']['endian']=='big':
                    char_endian='>'
                f_write=open(filename,'ab')
                # write data
                # 2.1 write the number of rows for the column set
                if len(self.column)<=0:
                    num_rows=0
                else:
                    num_rows=len(self.column[0]['value'])
                f_write.write(struct.pack(char_endian+'l',num_rows))
                # 2.2 write parameters
                # note: fixed_value paras won't appear here
                for i in range(len(self.parameter)):
                    parameter_temp=self.parameter[i]
                    list_keyName=parameter_temp.keys()
                    if 'fixed_value' not in list_keyName:
                        data2write=parameter_temp['value']
                        datatype=parameter_temp['type']
                        data_write_binary(f_write,datatype,data2write,char_endian)
                # 2.3 write array
                for i in range(len(self.array)):
                    array_temp=self.array[i]
                    # write the shape first
                    data2write=array_temp['value']
                    datatype=array_temp['type']
                    shape2write=np.shape(data2write)
                    total_num=1
                    for j in range(len(shape2write)):
                        f_write.write(struct.pack(char_endian+'l',shape2write[j]))
                        total_num=total_num*shape2write[j]
                    # reshape the data to 1d
                    data2write=np.reshape(data2write,total_num)
                    for j in range(len(data2write)):
                        data_write_binary(f_write,datatype,data2write[j],char_endian)
                # 2.4 write column
                if len(self.column)>0:
                    num_rows=len(self.column[0]['value'])
                    for i in range(num_rows):
                        for j in range(len(self.column)):
                            data2write=self.column[j]['value']
                            datatype=self.column[j]['type']
                            data2write=data2write[i]
                            data_write_binary(f_write,datatype,data2write,char_endian)
                print(filename+' saved')
                return
        print('warning: nothing is saved')
        return
    
    # output the SDDS data to a hdf5 file
    def write2hdf5_std(self, filename=''):
        if len(filename)>0:
            # create an empty one first
            f=hdf5.File(fileName,'w')
            grp_description = f.create_group('description')
            grp_column = f.create_group('column s')
            grp_parameter = f.create_group('parameters')
            grp_data = f.create_group('data')
            # assign description
            for keyName in self['description'].keys():
                grp_description.create_dataset(keyName,data=self['description'][keyName])
            # assign parameter
            for i in range(len(self.parameter)):
                parameter_temp=self.parameter[i]
                grp_para_temp=grp_parameter.create_group(parameter_temp['name'])
                for keyName in parameter_temp.keys():
                    grp_para_temp.create_dataset(keyName,data=parameter_temp[keyName])
            # assign array
            for i in range(len(self.array)):
                array_temp=self.array[i]
                grp_arr_temp=array.create_group(array_temp['name'])
                for keyName in array_temp.keys():
                    grp_arr_temp.create_dataset(keyName,data=array_temp[keyName])
            # assign column
            for i in range(len(self.column)):
                column_temp=self.column[i]
                grp_col_temp=array.create_group(column_temp['name'])
                for keyName in column_temp.keys():
                    grp_col_temp.create_dataset(keyName,data=column_temp[keyName])
            f.close()
            return
        else:
            print('warning: no filename is specified')
            return
        return
    
    # output the SDDS data to a hdf5 file (only with value, and with fmt)
    def write2hdf5_fmt(self, filename='', data_list=[], fmt_list=[]):
        # if data list is empty, save everything in a big file
        # warning: if there are entries with same name, it won't work
        if len(data_list)<=0:
            f=hdf5.File(filename,'w')
            # assign description
            for keyName in self['description'].keys():
                f.create_dataset(keyName,data=self['description'][keyName])
            # assign parameter
            for i in range(len(self.parameter)):
                parameter_temp=self.parameter[i]
                f.create_dataset(parameter_temp['name'],parameter_temp['value']) 
            # assign array
            for i in range(len(self.array)):
                array_temp=self.array[i]
                f.create_dataset(array_temp['name'],array_temp['value']) 
            # assign column
            for i in range(len(self.column)):
                column_temp=self.column[i]
                f.create_dataset(column_temp['name'],column_temp['value'])
            f.close()
            return
        else:
            f=hdf5.File(filname,'w')
            for i in range(len(data_list)):
                data2save=self.getdict(data_list[i])
                pFile_temp=f
                for j in range(len(fmt_list[i])):
                    if fmt_list[i][j] not in pFile_temp.keys():
                        pFile_temp=pFile_temp.create_group(fmt_list[i][j])
                    else:
                        pFile_temp=pFile_temp[fmt_list[i][j]]
                pFile_temp.create_dataset(data2save['name'],data2save['value'])
            f.close()
            return
        return        
                
    
    # get the list of data names 
    def getinfo(self):
        print('Summary of the file:')
        print('description:')
        for keyName in self['description'].keys():
            print(keyName+'-- \''+self['description'][keyName],end='\' ')
        print('')
        print('Parameters: [', end=' ')
        for i in range(len(self.parameter)):
            print(self.parameter[i]['name'], end=', ')
        print(']')
        print('Arrays: [', end=' ')
        for i in range(len(self.array)):
            print(self.array[i]['name'], end=', ')
        print(']')
        print('Columns: [', end=' ')
        for i in range(len(self.column)):
            print(self.column[i]['name'], end=', ')
        print(']')
        
    # get the info of a specific entry
    def getentryinfo(self,name):
        founded=False
        for i in range(len(self.parameter)):
            if self.parameter[i]['name']==name:
                founded=True
                print(name+' founded in parameter:')
                for keyName in self.parameter[i].keys():
                    if keyName != 'name':
                        print(keyName+':', end=' ')
                        print(self.parameter[i][keyName])
        for i in range(len(self.array)):
            if self.array[i]['name']==name:
                founded=True
                print(name+' founded in array:')
                for keyName in self.array[i].keys():
                    if keyName != 'name':
                        if keyName == 'value':
                            print(keyName+' (shape):', end=' ')
                            print(np.shape(self.array[i][keyName]))
                        else:
                            print(keyName+':', end=' ')
                            print(self.array[i][keyName])
        for i in range(len(self.column)):
            if self.column[i]['name']==name:
                founded=True
                print(name+' founded in column:')
                for keyName in self.column[i].keys():
                    if keyName != 'name':
                        if keyName == 'value':
                            print(keyName+' (shape):', end=' ')
                            print(np.shape(self.column[i][keyName]))
                        else:
                            print(keyName+':', end=' ')
                            print(self.column[i][keyName])
        if not founded:
            print('cannot find the data!')
        return
            
        
    # get a wanted dict (only valid when there is no same name)    
    def getdict(self, Dictname):
        # search in parameter
        for i in range(len(self.parameter)):
            if Dictname==self.parameter[i]['name']:
                return self.parameter[i]
        # search in array
        for i in range(len(self.array)):
            if Dictname==self.array[i]['name']:
                return self.array[i]
        # search in column
        for i in range(len(self.column)):
            if Dictname==self.column[i]['name']:
                return self.column[i]
        return dict()
    
    # get a data (only valid when there is no same name)
    def getdata(self, Dataname):
        dict_temp=self.getdict(Dataname)
        if 'value' in dict_temp.keys():
            return dict_temp['value']
        else:
            print('warning: there is no data in the required entry')
            return []
        return []
    
    # set a data
    def setdata(self, Dataname, Data):
        # search in parameter
        for i in range(len(self.parameter)):
            if Dataname==self.parameter[i]['name']:
                self.parameter[i]['value']=Data
        # search in array
        for i in range(len(self.array)):
            if Dataname==self.array[i]['name']:
                self.array[i]['value']=Data
        # search in column
        for i in range(len(self.column)):
            if Dataname==self.column[i]['name']:
                self.column[i]['value']=Data

####################################################################################################################
# SDDS class-related functions:
# SDDSconvert(file1,file2) # convert the mode of a given file
# ...
####################################################################################################################                                              

# switch the SDDS file
# can be slow because it reads in all the data first
def SDDSconvert(filename_ori,filename_tar):
    SDDS_ori=SDDS(filname_ori)
    if SDDS_ori['data']['mode']=='ascii':
        SDDS_ori.write2file(filename_tar,'binary')
    elif SDDS_ori['data']['mode']=='binary':
        SDDS_ori.write2file(filename_tar,'ascii')
    else:
        print('warning: mode error')
    return SDDS_ori