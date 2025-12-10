from mne import channels
import numpy as np
import os
from pyedflib import highlevel
import logging
logger= logging.getLogger()

class EdfUtil:
    """
    channel_names: A list with labels for each channel.
    sample_rate: raw sampling raw
    subject_name: the name of the subject
    file_path: path to save the edf file
    file_name: name of the file to be saved
    """
    def __init__(self,channel_names,sample_rate,subject_name,gender,file_path,file_name):
        self.channel_names = channel_names
        self.sample_rate = sample_rate
        self.subject_name = subject_name
        self.gender = gender
        self.file_path = file_path
        self.file_name = file_name
        self.signal_headers = []
        self.header = highlevel.make_header(patientname=subject_name, gender=gender)
        self.file_path_name = os.path.join(file_path, file_name)

    def make_annotations(self, conductor):
        rst = []
        for timepoint, description in conductor:
            if description == 16:
                continue
            rst.append([timepoint, -1, str(description)])
        return rst

    def write_edf(self, signals, conductor):
        self.header['annotations'] = self.make_annotations(conductor)
        
        for index, channel in enumerate(self.channel_names):
            p_min = signals[index].min()
            p_max = signals[index].max()
            header = highlevel.make_signal_header(channel, dimension='uV', sample_rate=100, physical_min=p_min, physical_max=p_max)
            self.signal_headers.append(header)

        if os.path.exists(self.file_path) ==False:
            os.makedirs(self.file_path)
        logger.info("===signal_headers==={}".format(self.signal_headers))
        highlevel.write_edf(self.file_path_name, signals, self.signal_headers, self.header)

    def write_edf_without_anno(self, signals):
        highlevel.write_edf(self.file_path_name, signals, self.signal_headers, self.header)

def read_edf(file_name):
    signals, signal_headers, header = highlevel.read_edf(file_name)
    # highlevel.change_polarity()
    return {}

# read_edf(../data/raw_edf_tmp/SL04-T03.edf')
