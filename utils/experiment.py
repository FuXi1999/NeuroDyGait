import os
import pandas as pd
import mne

class Experiment:
    'class of one experimemt'
    def __init__(self, path) -> None:
        self.channelOrder = ['Time', 'Fp1', 'Fp2', 'AF7', 'AF3', 'T8', 'AF4', 'AF8',  'F7', 
       'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'T7', 'FC2',
       'FC4', 'FC6', 'FT8',  'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'TP7', 'CP5', 'CP3',
       'CP1', 'CPz', 'CP2', 'CP4', 'CP6',  'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2',
       'P4',  'P6' , 'P8', 'PO9', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'PO10', 'O1', 'Oz', 'O2',
       'TP9', 'TP10', 'FT9', 'FT10']
        self.changeChannelDic = {'T7':'FCz', 'T8':'AFz', 'TP9':'VEOU', 'TP10':'VEOL', 'FT9':'HEOL', 'FT10':'HEOR'}
        
        self.channelOrder.reverse()
        self.experimentName = os.path.basename(path)
        self.experimentPath = path
        self.impedanceBefore = self.loadImpedances('before')
        self.impedanceAfter = self.loadImpedances('after')
        self.loadEEG()
        self.loadJoints()
        self.loadConductor()
        # self.loadDigitizer()

    def loadEEG(self):
        eegFilePath = os.path.join(self.experimentPath,'eeg.txt')
        self.eegData = pd.read_table(eegFilePath, sep='\t', skiprows=[0], header=None)
        if self.eegData.shape[1]>65:
            del self.eegData[65]
        columnName = ['Time'] + self.impedanceBefore.loc['Name'].tolist()[:64]
        self.eegData.columns = columnName
        self.eegData = self.eegData[self.channelOrder]
        self.eegData.rename(columns=self.changeChannelDic, inplace=True)
    
    def loadImpedances(self, flag):
        impedancesFilePath = os.path.join(self.experimentPath, 'impedances-'+flag+'.txt')
        skiprows = list(range(21))
        impedancesData = pd.read_csv(impedancesFilePath, sep='\s*\t\s*', skiprows=skiprows, engine='python', header=None)
        impedancesData.columns = ['PhysChn', 'Name', 'LabelSet', 'ImpedanceValue']
        impedancesData.replace('---', 1000, inplace=True)
        impedancesData.ImpedanceValue = impedancesData.ImpedanceValue.astype(int)
        channelList = impedancesData.Name.tolist()
        impedancesData = impedancesData.T
        impedancesData.columns = channelList
        impedancesData.rename(columns=self.changeChannelDic, inplace=True)
        return impedancesData

    def loadJoints(self):
        jointsFilePath = os.path.join(self.experimentPath, 'joints.txt')
        self.jointsData = pd.read_table(jointsFilePath, skiprows=[0,1], header=None)
        if self.jointsData.shape[1]>13:
            del self.jointsData[13]
        columnName = ['Time', 'GHR', 'GKR', 'GAR', 'GHL', 'GKL', 'GAL', 'PHR', 'PKR', 'PAR', 'PHL', 'PKL', 'PAL']
        self.jointsData.columns = columnName 
    
    def loadConductor(self):
        conductorFilePath = os.path.join(self.experimentPath, 'conductor.txt')
        self.conductorData = pd.read_table(conductorFilePath, sep='\t', skiprows=[0,1], header=None)
        columnName = ['Time', 'Event']
        self.conductorData.columns = columnName
    
    def loadDigitizer(self):
        # KeyError
        digitizerFilePath = os.path.join(self.experimentPath, 'digitizer.bvct')
        self.digitizerData = mne.channels.read_dig_captrak(digitizerFilePath)

# SL01_T01 = Experiment(path='../data/RepositoryData/SL01-T01')
# eeg = SL01_T01.eegData