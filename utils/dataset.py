from inspect import stack
import torch
import random
import numpy as np
import scipy.io as sio

from scipy.signal.filter_design import butter
from scipy.signal import filtfilt

from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler

class MobiDataSet():
    def __init__(self, subject_path, time_step, sep_fraction=0.8):
        self.data_path = subject_path
        self.time_step = time_step
        self.sep_fraction = sep_fraction
        self.data = sio.loadmat(self.data_path)

    def get_data(self, standard=True, cha_order='origin', type='raw', remove_EMG=False, ratio=1):
        # data: time_points * chan
        self.data['trainEEG'] = self.adjust_chan_order(self.data['trainEEG'], cha_order)
        self.data['testEEG'] = self.adjust_chan_order(self.data['testEEG'], cha_order)

        trainEEG, valEEG = self.train_valid_separation(self.sep_fraction, self.data['trainEEG'])
        trainJoints, valJoints = self.train_valid_separation(self.sep_fraction, self.data['trainJoints'][:,:6])
        testEEG, testJoints = self.data['testEEG'], self.data['testJoints'][:,:6]
        scJoints = None
        if remove_EMG:
            trainEEG = self.remove_periphery_EMG(trainEEG)
            valEEG = self.remove_periphery_EMG(valEEG)
            testEEG = self.remove_periphery_EMG(testEEG)
        if standard:
            _, trainEEG, \
                valEEG, testEEG = self.standardize_dataset(trainEEG, valEEG, testEEG)
            scJoints, trainJoints, valJoints, \
                testJoints = self.standardize_dataset(trainJoints, valJoints, testJoints)

        trainEEG_make, trainJoints_make = \
                self.make_dataset(trainEEG, trainJoints, self.time_step, type, ratio)
            
        valEEG_make, valJoints_make = \
            self.make_dataset(valEEG, valJoints, self.time_step, type, ratio=1)
            
        testEEG_make, testJoints_make = \
            self.make_dataset(testEEG, testJoints, self.time_step, type, ratio=1)
        
        return_dic = {'trainEEG':trainEEG_make, 'trainJoints':trainJoints_make, 'valEEG':valEEG_make,
                          'valJoints':valJoints_make, 'testEEG':testEEG_make, 'testJoints':testJoints_make,
                          'scJoints':scJoints, 'jointsColumns':self.data['jointsChannels'][:6], 'eegChannels': self.data['eegChannels']}
        return return_dic

    def make_dataset(self, eeg, joints, time_step, type, ratio):
        '''
        create dataset
        :param eeg:(numpy, [n_samples, n_chans]) eeg matrix 
        :param joints:(numpy, [n_samples, n_chans]) joints matrix
        :param time_step:(int) window size
        '''
        data_len = eeg.shape[0]-time_step+1
        chans_num = eeg.shape[1]
        joints_cor = joints[time_step-1:, :]

        if type == 'raw':
            if time_step == 0:
                return eeg, joints
            data_index_list = list(range(data_len))
            if ratio == 1:
                fat_EEG = np.full((data_len, time_step, chans_num), np.nan)
            else:
                random.shuffle(data_index_list)
                data_index_list = data_index_list[:int(ratio*data_len)]
                data_index_list.sort()
                fat_EEG = np.full((int(ratio*data_len), time_step, chans_num), np.nan)
                joints_cor = joints_cor[data_index_list]

            for i, idx in enumerate(data_index_list):
                fat_EEG[i] = eeg[idx:idx+time_step,:]

            return fat_EEG, joints_cor

        elif type=='power':
            bands = [[0.1, 4.0],[4.0, 8.0],[8.0, 12.0],[12.0,35.0],[35.0, 49.9]]
            concated = False
            for band in bands:
                eeg_filter = self.bandpass_filter(eeg, band[0], band[1])
                if not concated:
                    eeg_ = eeg_filter
                    concated = True
                else:
                    eeg_ = np.hstack([eeg_, eeg_filter])
            
            eeg_ = eeg_**2
            concated = False
            for idx in range(data_len):
                step_EEG = np.sum(eeg_[idx:idx+time_step,:], axis=0, keepdims=True)
                eeg_[idx] = step_EEG
               
            return eeg_[:data_len], joints_cor

    def train_valid_separation(self, sep_fraction, data_to_sep):
        """
        A function to separate the data into training and validation. 
        based on https://github.com/shonaka/EEG-neural-decoding

        :param sep_fraction:(float) How much portion of the data you want to use for train. e.g. 0.8 for 80%, float
        :param data_to_sep:(numpy, [n_samples, n_chans])  The data you want to separate.
        """
        t_samp = int(data_to_sep.shape[0]*sep_fraction)
        t_data = data_to_sep[:t_samp, :]
        v_data = data_to_sep[t_samp+1:, :]

        return t_data, v_data

    def standardize_dataset(self, train_data, validation_data, test_data):
        """
        Standardize the dataset. 
        based on https://github.com/shonaka/EEG-neural-decoding
        :param sc: standardize class from sklearn "StandardScaler()"
        :param train_data: the train data you want to standardize
        :param validation_data: the validation data you want to standardize based on the train data
        :param test_data: the test data you want to standardize based on the train data

        Returns:
        :param sc: standardize class fit to the data. Used later for transformation.
        :param train_stan: standardized train data
        :param validation_stan:  standardized validation data
        :param test_stan:  standardized test data
        """
        sc = StandardScaler()
        train_stan = sc.fit_transform(train_data)
        validation_stan = sc.transform(validation_data)
        test_stan = sc.transform(test_data)

        return sc, train_stan, validation_stan, test_stan
    
    def bandpass_filter(self, eeg, low, high, order=4, sr=100):
        '''
        Butterworth bandpass filter 
        '''
        wn = np.array([low, high]) / sr * 2
        b, a = butter(order, wn, btype='bandpass')
        eeg_ = filtfilt(b, a, eeg, axis=0)

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(figsize=(20, 10))
        # ax.plot(eeg[0:500, 3], label='Original')
        # ax.plot(eeg_[0:500, 3], label='filtfilt')
        # plt.show()
        # input()
        return eeg_

    def adjust_chan_order(self, eeg, cha_order='origin'):
        origin_list = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                        'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 
                        'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'C5', 'C3', 'C1', 'Cz', 'C2', 
                        'C4', 'C6', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 
                        'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 
                        'PO9', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'PO10', 'O1', 'Oz', 'O2']

        hemi_list = ['Fp1', 'AF7', 'AF3', 'F7', 'F5', 'F3', 'F1', 'FT7', 'FC5',
                    'FC3', 'FC1', 'C5', 'C3', 'C1', 'TP7', 'CP5', 'CP3', 'CP1',
                    'P7', 'P5', 'P3', 'P1', 'PO9', 'PO7', 'PO3', 'O1', 'AFz', 
                    'Fz', 'FCz', 'Cz', 'Fp2', 'AF8', 'AF4', 'F8', 'F6', 'F4',
                    'F2', 'FT8', 'FC6', 'FC4', 'FC2', 'C6', 'C4', 'C2', 'TP8',
                    'CP6', 'CP4', 'CP2', 'P8', 'P6', 'P4', 'P2', 'PO10', 'PO8',
                    'PO4', 'O2', 'CPz', 'Pz', 'POz', 'Oz']
        
        hemi_gcn_overlap_list = [
            'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', # left frontal [0, 10]
            'AFz', 'Fz', 'FCz',                                                      # middle frontal [11, 13]
            'AFz', 'Fz', 'FCz',                                                      # middle frontal [14, 16]
            'Fp2', 'AF8', 'AF4', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', # right frontal [17, 27]

            'C5', 'C3', 'C1', 'CP1', 'CP3', 'CP5', 'TP7', # left central [28, 34]
            'Cz', 'CPz',                                  # middle central [35, 36]
            'Cz', 'CPz',                                  # middle central [37, 38]
            'C6', 'C4', 'C2', 'CP2', 'CP4', 'CP6', 'TP8', # right central [39, 45]
            
            'P7', 'P5', 'P3', 'P1', 'PO3', 'PO7', 'PO9', 'O1', # left posterior/occipital [46, 53]
            'Pz', 'POz', 'Oz',                                 # middle posterior/occipital [54, 56]
            'Pz', 'POz', 'Oz',                                 # middle posterior/occipital [57, 59]
            'P8', 'P6', 'P4', 'P2', 'PO4',  'PO8', 'PO10', 'O2'# right posterior/occipital [60, 67]
            ]
        
        emg_sen_list = ['AF7', 'F7', 'FT7', 'TP7', 'P7', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'P8', 'TP8', 'FT8', 'F8', 'AF8']
        
        if cha_order == 'origin':
            return eeg

        if cha_order == 'Hemi':
            mask_list = [0, 2, 3, 7, 8, 9, 10, 16, 17, 18, 19, 25, 26, 27, 32, 33, 
                        34, 35, 41, 42, 43, 44, 50, 51, 52, 57, 4, 11, 20, 28, 1, 
                        6, 5, 15, 14, 13, 12, 24, 23, 22, 21, 31, 30, 29, 40, 39,
                        38, 37, 49, 48, 47, 46, 56, 55, 54, 59, 36, 45, 53, 58]

            # mask_list = [origin_list.index(cha) for cha in hemi_list]
            return eeg[:, mask_list]

        if cha_order == 'Hemi_GCN_Overlap':
            mask_list = [
            0, 2, 3, 10, 9, 8, 7, 16, 17, 18, 19, 
            4, 11, 20, 
            4, 11, 20, 
            1, 6, 5, 12, 13, 14, 15, 24, 23, 22, 21, 
            
            25, 26, 27, 35, 34, 33, 32,
            28, 36, 
            28, 36, 
            31, 30, 29, 37, 38, 39, 40, 
            
            41, 42, 43, 44, 52, 51, 50, 57, 
            45, 53, 58, 
            45, 53, 58, 
            49, 48, 47, 46, 54, 55, 56, 59]
            # mask_list = [origin_list.index(cha) for cha in hemi_gcn_overlap_list]
            return eeg[:, mask_list]
            
        if cha_order == 'de_emg':
            de_emg_sen_mask_list = []
            for idx, cha in enumerate(origin_list):
                if cha not in emg_sen_list:
                    de_emg_sen_mask_list.append(idx)
            return eeg[:, de_emg_sen_mask_list]

    def remove_periphery_EMG(self, eeg):
        """
        :param eeg: T x num_of_Channels
        :return: new_eeg: T x num_of_Channels
        """
        periphery = [2, 7, 16, 32, 41, 50, 57, 58, 59, 56, 49, 40, 24, 15, 6]
        internal = [0, 1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47, 48, 51, 52, 53, 54, 55]
        eeg_tmp1 = eeg[:, internal]
        eeg_tmp2 = eeg[:, periphery]
        for i in range (eeg.shape[0]):
            if (i == 0 or i == eeg.shape[0] - 1):
                for j in range (15):
                    eeg_tmp2[i,j] = eeg[i, j-1] + eeg[i, j+1] - 2 * eeg[i, j]
            else:
                for j in range(15):
                    eeg_tmp2[i, j] = eeg[i, j - 1] + eeg[i, j + 1] + eeg[i-1, j] + eeg[i+1, j] - 4 * eeg[i, j]
        new_eeg = np.concatenate((eeg_tmp1, eeg_tmp2), 1)
        return new_eeg

def get_dataloader(X, Y, batch_size, shuffle=True):
    X_torch = torch.from_numpy(X).float()
    Y_torch = torch.from_numpy(Y).float()
    
    dataset = TensorDataset(X_torch, Y_torch)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader

def get_edge_index(edge_range, time_step, n_chans=60):
    """
    return edges within one time window
    """
    start, end = edge_range[0], edge_range[1]
    edge_index_list = []
    for i in range(start, end+1):
        for j in range(i+1, end+1):
            for t in range(time_step):
                #todo: what is the purpose of i+(t*n_chans),j+(t*n_chans)?
                edge_index_list.append([i+(t*n_chans),j+(t*n_chans)])
                edge_index_list.append([j+(t*n_chans),i+(t*n_chans)])
    return edge_index_list

def get_gcn_dataloader(X, Y, config, shuffle=True, type='1point'):
    """
    return a  dataloader
    almost same as the nomal one
    but the data includes data.x, data.y, data.edge_index
    """
    # X [n_sample, time_step, n_chans]
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data

    time_step = X.shape[1]
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()

    if type=='1point':
        #X is reshaped to be [n_samples, time_step * n_chans]
        X = X.reshape((X.shape[0], -1))
        edge_index_list = []
        for one_region in config.gcn.brain_regions:
            edge_index_list += get_edge_index(one_region, time_step, n_chans=config.gcn.brain_regions[-1][1]+1)
        edge_index = torch.tensor(edge_index_list, dtype=torch.long)
        data_list = []
        for idx in range(X.shape[0]):
            data = Data(x=X[idx:idx+1].t(), 
                        edge_index=edge_index.t().contiguous(),
                        y=Y[idx])
            data_list.append(data)
        data_loader = DataLoader(data_list, batch_size=config.batch_size, shuffle=shuffle)
    elif type=='rawtime':
        X = torch.permute(X, [0,2,1])# X [n_sample, n_chans, time_step]
        edge_index_list = []
        for one_region in config.gcn.brain_regions:
            edge_index_list += get_edge_index(one_region, 1, n_chans=config.gcn.brain_regions[-1][1]+1)
        edge_index = torch.tensor(edge_index_list, dtype=torch.long)
        
        data_list = []
        for idx in range(X.shape[0]):
            data = Data(x=X[idx], 
                        edge_index=edge_index.t().contiguous(),
                        y=Y[idx])
            data_list.append(data)
        data_loader = DataLoader(data_list, batch_size=config.batch_size, shuffle=shuffle)
    return data_loader