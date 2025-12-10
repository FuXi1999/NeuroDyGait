from matplotlib.pyplot import axis
import numpy as np
import pandas as pd

from .experiment import Experiment
from .EdfUtil import EdfUtil

def txt2edf(exp_path, subject_name, save_path):
    exp = Experiment(path=exp_path)
    eegDf = exp.eegData

    # remove 'Time', 'HEOL', 'HEOR', 'VEOL', 'VEOU'
    # add 'HEOG', 'VEOG' 
    columns = list(eegDf.columns)
    usedColumns = columns[4:-1][::-1]+['HEOG', 'VEOG']

    # calculate HEOG and VEOG
    eegDf['HEOG'] = eegDf['HEOL'] - eegDf['HEOR']
    eegDf['VEOG'] = eegDf['VEOL'] - eegDf['VEOU']

    # interpolate raw EEG data
    timeColumn = eegDf.Time.values
    timeIndex = np.round(timeColumn*100).astype(int)-1
    fullLength = timeIndex[-1]+1

    fullEeg = np.full((fullLength, 62), np.nan)
    fullEeg[timeIndex] = eegDf[usedColumns].values

    fullEeg_df = pd.DataFrame(fullEeg, columns=usedColumns)
    fullEegInter_df = fullEeg_df.interpolate(method='polynomial', order=2)
    
    assert np.where(np.isnan(fullEegInter_df))[0].max() < 30
    
    fullEegInter_df.fillna(0, inplace=True)
    fullEegInter = fullEegInter_df[usedColumns].values

    # get the eeg data[channels*samples]
    usedEEGData = fullEegInter.T

    # transfer to edf file
    EDFFile = EdfUtil(usedColumns, 100, subject_name, 'X', save_path, subject_name+'.edf')
    conductor = exp.conductorData.values # add triggers
    EDFFile.write_edf(usedEEGData, conductor)