import os
import scipy.io
import numpy as np
import mne
from scipy.interpolate import interp1d
import pickle

dataset_base = '/home/fuxi/FBM/raw/'
channel_names = {}

standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'T1', 'T2', 'I1', 'I2', 'HEO', 'VEO', 'ECG', 'EMG', \
    'F3-C3', 'F4-C4', 'F7-T3', 'F8-T4', 'FP1-F3', 'P3-O1', 'P4-O2', 'T3-T5', 'T4-T6', \
    'C3-P3', 'C4-A1', 'C4-P4', 'F2-F4'
]


GPP_ch_names = ['FP1', 'FZ', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
                'PZ', 'P3', 'P7', 'O1', 'OZ', 'O2', 'P4', 'P8', 'CP6', 'CP2',
                'CZ', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'FP2', 'AF7', 'AF3',
                'AFZ', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1',
                'P5', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'P6', 'P2', 'CPZ', 'CP4',
                'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2']

FBM_ch_names = ['FP1', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 'F7', 'F5', 'F3', \
               'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', \
                'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', \
                'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', \
                'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', \
                'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2']

EEG_order = ['FP1', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 'F7', 'F5', 'F3', \
               'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', \
                'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', \
                'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', \
                'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', \
                'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2'
            ]
# joint_names = ['jL5S1', 'jL4L3', 'jL1T12', 'jT9T8', 'jT1C7', 'jC1Head', \
#                     'jRightC7Shoulder', 'jRightShoulder', 'jRightElbow', 'jRightWrist', \
#                     'jLeftC7Shoulder', 'jLeftShoulder', 'jLeftElbow', 'jLeftWrist', \
#                     'jRightHip', 'jRightKnee', 'jRightAnkle', 'jRightBallFoot', \
#                     'jLeftHip', 'jLeftKnee', 'jLeftAnkle', 'jLeftBallFoot']
joint_names = ['jRightHip', 'jRightKnee', 'jRightAnkle', 'jRightBallFoot', \
                    'jLeftHip', 'jLeftKnee', 'jLeftAnkle', 'jLeftBallFoot']
EOG_channels = ['VEO', 'HEO']
for dim in ['X', 'Y', 'Z']:
    for name in joint_names:
        standard_1020.append(name + dim)
eeg_l_freq = 0.1
eeg_h_freq = 75.0
eeg_rsfreq = 200
eeg_samp_rate = 1000

emg_l_freq = 5
emg_h_freq = 200
emg_rsfreq = 500
emg_samp_rate = 1000

eog_l_freq = 0.1
eog_h_freq = 75.0
eog_rsfreq = 200
eog_samp_rate = 1000

kin_rsfreq = 200

seg_freq = 20
window_time = 2

for file in os.listdir(dataset_base):
    if not file.endswith('.bvct'):
        continue
    sbj = file.split('-')[0].split('_')[-1]
    channel_names[sbj] = []
    with open(dataset_base + file, 'r') as f:
        lines = f.readlines()[:640]
        for i in range(len(lines)):
            lines[i] = lines[i].replace(' ', ',')
            if not '<Name>' in lines[i]:
                continue
            channel_name = lines[i].split('<Name>')[1].split('</Name>')[0]
            if (not channel_name.upper() in standard_1020) or channel_name.upper() == 'A1' or channel_name.upper() == 'A2':
                # print(channel_name.upper())
                continue
            if channel_name.upper() == 'FT9':
                channel_name = 'AFZ'
            if channel_name.upper() == 'FT10':
                channel_name = 'FCZ'
            channel_names[sbj].append(channel_name.upper())

info = mne.create_info(ch_names=GPP_ch_names, sfreq=1000, ch_types='eeg')    


def sample_generation(data, samp_freq, seg_freq, t):
    """
    将数据切割成训练样本。

    参数：
    - data: numpy.ndarray, 形状为 (C, T)，输入数据，其中 C 是通道数，T 是时间步数。
    - samp_freq: int，data 的采样率。
    - seg_freq: int，切割频率（每秒切割的次数）。
    - t: int，样本时间窗的长度，单位为秒。

    返回：
    - numpy.ndarray, 形状为 (num_samples, C, samp_freq * t)，切割后的样本。
    """
    C, T = data.shape

    # 每个样本的长度
    sample_length = samp_freq * t

    # 计算总的样本数量，考虑到最后一个样本可能不足完整时间窗
    num_samples = (T - sample_length ) // int(samp_freq / seg_freq) + 1

    # 初始化存储切片的数组
    samples = np.zeros((num_samples, C, sample_length))

    # 填充切片数据
    for i in range(num_samples):
        start_idx = int(i * (samp_freq / seg_freq))
        end_idx = start_idx + sample_length

        samples[i] = data[:, start_idx:end_idx]

    return samples

def process_fbm_eeg(eeg_data, fbm_order, target_order):
    """
    处理 FBM 数据集 EEG，删除 FCZ 并按 GPP 顺序重新排列。

    Parameters:
        eeg_data: np.ndarray, shape (60, T) 原始 EEG 数据
        fbm_order: list[str], FBM 通道名列表（顺序应与 eeg_data 匹配）
        target_order: list[str], 目标通道顺序 (GPP_ch_names)

    Returns:
        eeg_out: np.ndarray, shape (59, T)，重排后的 EEG 数据
    """
    # Step 1: 删除 FCZ
    if 'FCZ' in fbm_order:
        idx_fcz = fbm_order.index('FCZ')
        eeg_data = np.delete(eeg_data, idx_fcz, axis=0)
        fbm_order = fbm_order[:idx_fcz] + fbm_order[idx_fcz+1:]

    # Step 2: 创建通道名到索引的映射
    ch_to_idx = {ch: i for i, ch in enumerate(fbm_order)}

    # Step 3: 重排为 target_order 顺序
    eeg_out = np.stack([eeg_data[ch_to_idx[ch]] for ch in target_order], axis=0)

    return eeg_out

def preprocessing_EEG(eeg_file, eeg_l_freq=0.1, eeg_h_freq=75.0, eeg_rsfreq=200, eeg_samp_rate=1000, seg_freq=10, t=2):
    """
    对 EEG 数据进行预处理。

    参数：
    - data: numpy.ndarray, 形状为 (C, T)，输入数据，其中 C 是通道数，T 是时间步数。
    - samp_freq: int，data 的采样率。
    - l_freq: float，数据的低通滤波频率。
    - h_freq: float，数据的高通滤波频率。
    - rs_freq: int，数据的下采样频率。
    - seg_freq: int，切割频率（每秒切割的次数）。
    - t: int，样本时间窗的长度，单位为秒。

    返回：
    - numpy.ndarray, 形状为 (num_samples, C, samp_freq * t)，切割后的样本。
    """
    mat = scipy.io.loadmat(dataset_base + eeg_file)
    eeg_data = mat['eeg'][0][0]['rawdata']  # (60, T)
    eeg_data = process_fbm_eeg(eeg_data, channel_names[sbj], GPP_ch_names)
    # channel_order = channel_names[sbj]
    # ordered_data = []
    # for channel in standard_1020:
    #     if channel in channel_order:
    #         idx = channel_order.index(channel)
    #         ordered_data.append(eeg_data[idx])
    # eeg_data = np.array(ordered_data)
    # print(eeg_data.shape)
    eeg_mne_data = mne.io.RawArray(eeg_data, info)
    # Band-pass filter the EEG signal between 0.1Hz and 75Hz
    eeg_mne_data.filter(l_freq=eeg_l_freq, h_freq=eeg_h_freq, method='iir')
    # Apply a notch filter at 50Hz to remove power line noise
    eeg_mne_data.notch_filter(freqs=50, method='iir')
    # 设置平均参考
    eeg_mne_data.set_eeg_reference('average', projection=False)


    # Resample the EEG data
    eeg_mne_data.resample(eeg_rsfreq, npad="auto")

    eeg_data = eeg_mne_data.get_data()
    del eeg_mne_data
    eeg_samples = sample_generation(eeg_data, eeg_rsfreq, seg_freq, 2)
    return eeg_samples


def preprocessing_KIN(kin_file, kin_rsfreq=10, seg_freq=10, t=2):
    """
    对 KIN 数据进行预处理。

    参数：
    - data: numpy.ndarray, 形状为 (C, T)，输入数据，其中 C 是通道数，T 是时间步数。
    - samp_freq: int，data 的采样率。
    - rs_freq: int，数据的下采样频率。
    - seg_freq: int，切割频率（每秒切割的次数）。
    - t: int，样本时间窗的长度，单位为秒。

    返回：
    - numpy.ndarray, 形状为 (num_samples, C, samp_freq * t)，切割后的样本。
    """
    kin_data = scipy.io.loadmat(dataset_base + kin_file)['kin']['data'][0][0][0][0]['jointAngle']
    # print(joint_names)
    # print(np.array(kin_data[name][0][0][:, [2]]).shape)
    kin_data = np.concatenate(([np.array(kin_data[name][0][0][:, [2]]) for name in joint_names]), axis=1)
    # Normalize KIN data to have zero mean and unit variance for each channel
    kin_data = (kin_data - np.mean(kin_data, axis=0)) / np.std(kin_data, axis=0)
    kin_samp_rate = int(scipy.io.loadmat(dataset_base + kin_file)['kin']['srate'])
    T = kin_data.shape[0]
    time_original = np.linspace(0, T / kin_samp_rate, T).flatten()
    num_new_samples = int(T * kin_rsfreq / kin_samp_rate)
    time_new = np.linspace(0, T / kin_samp_rate, num_new_samples)
    kin_data_resampled = np.zeros((num_new_samples, kin_data.shape[1]))
    for i in range(kin_data.shape[1]):
        y = kin_data[:, i].flatten()
        interp_func = interp1d(time_original, y, kind='linear', fill_value="extrapolate")
        kin_data_resampled[:, i] = interp_func(time_new)
    kin_data = kin_data_resampled.T
    del kin_data_resampled
    kin_samples = sample_generation(kin_data, kin_rsfreq, seg_freq, 2)
    label = kin_samples
    return label

num_trials = [0, 11, 9, 10, 9, 10, 11, 11, 10, 11]
flag = {}
for file in os.listdir(dataset_base):
    if not file.endswith('-eeg.mat'):
        continue

    trial = file.split('-')[1][1:]
    if trial == '00':
        continue
    sbj = file.split('-')[0].split('_')[-1]
    if sbj < '02':
        continue
    if sbj not in flag:
        flag[sbj] = 0
    emg_file = file.replace('-eeg.mat', '-emg.mat')
    if not os.path.exists(dataset_base + emg_file):
        print('\nNo EMG file for ' + file)
        continue

    kin_file = file.replace('-eeg.mat', '-kin.mat')
    if not os.path.exists(dataset_base + kin_file):
        print('\nNo KIN file for ' + file)
        continue
    label = preprocessing_KIN(kin_file, kin_rsfreq, seg_freq, window_time)
    label = np.transpose(label, (0, 2, 1))

    eeg_samples = preprocessing_EEG(file, eeg_l_freq, eeg_h_freq, eeg_rsfreq, eeg_samp_rate, seg_freq, window_time)

    # print(eeg_samples.shape[0], label.shape[0])
    if not (max(eeg_samples.shape[0], label.shape[0]) - min(eeg_samples.shape[0], label.shape[0])) < 2:
        continue
    emg_labels = [''] + [str(i) for i in range(2, 13)]
    save_path = dataset_base.replace('raw', 'FBM_59ch') + sbj + '_01' + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(eeg_samples.shape[0]):
        dump_path = save_path + str(i + flag[sbj]) + '.pkl'
        # if not os.path.exists(dump_path):
        # print(label[i].shape)
        with open(dump_path, 'wb') as f:
            pickle.dump(
                {
                    "EEG": eeg_samples[i],
                    # "EEG_ch_names": EEG_order,
                    # "EMG": emg_samples[i],
                    # "EMG_ch_names": ['EMG' + str(j) for j in emg_labels],
                    # "EOG": eog_samples[i],
                    # "EOG_ch_names": EOG_channels,
                    "Y": label[i],
                },
                f,
            )
    flag[sbj] += eeg_samples.shape[0]

    
    
    
