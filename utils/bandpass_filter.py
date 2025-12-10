import numpy as np
import torch
from scipy.signal import butter, lfilter
import scipy.signal as signal
from scipy.signal import sosfiltfilt, cheb2ord, cheby2
import matplotlib.pyplot as plt


def bandpass_filter_pytorch(data, lowcut, highcut, fs=100, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    # 将输入数据转换为numpy数组并进行滤波
    data_numpy = data.detach().cpu().numpy()
    filtered_data_numpy = lfilter(b, a, data_numpy, axis=-1)

    # 将滤波后的数据转换回PyTorch张量
    filtered_data = torch.tensor(filtered_data_numpy, dtype=torch.float32).to('cuda:0')
    filtered_data.requires_grad_()

    return filtered_data
def bandpassfilter_cheby2_sos(data, bandFiltCutF, fs=100, filtAllowance=[0.2, 5], axis=2):
    '''
    Band-pass filter the EEG signal of one subject using cheby2 IIR filtering
    and implemented as a series of second-order filters with direct-form II transposed structure.

    Settings are based on Prof. Guan's suggestions

    Param:
        data: nparray, size [trials x channels x times], original EEG signal
        bandFiltCutF: list, len: 2, low and high cut off frequency (Hz).
                If any value is None then only one-side filtering is performed.
        fs: sampling frequency (Hz)
        filtAllowance: list, len: 2, transition bandwidth (Hz) of low-pass and high-pass f
        axis: the axis along which apply the filter.
    Returns:
        data_out: nparray, size [trials x channels x times], filtered EEG signal
    '''

    aStop = 40  # stopband attenuation
    aPass = 1  # passband attenuation
    nFreq = fs / 2  # Nyquist frequency

    if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (bandFiltCutF[1] == None or bandFiltCutF[1] >= fs / 2.0):
        # no filter
        print("Not doing any filtering. Invalid cut-off specifications")
        return data

    elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
        # low-pass filter
        print("Using lowpass filter since low cut hz is 0 or None")
        fPass = bandFiltCutF[1] / nFreq
        fStop = (bandFiltCutF[1] + filtAllowance[1]) / nFreq
        # find the order
        [N, wn] = signal.cheb2ord(fPass, fStop, aPass, aStop)
        sos = signal.cheby2(N, aStop, wn, 'lowpass', output='sos')

    elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
        # high-pass filter
        print("Using highpass filter since high cut hz is None or nyquist freq")
        fPass = bandFiltCutF[0] / nFreq
        fStop = (bandFiltCutF[0] - filtAllowance[0]) / nFreq
        # find the order
        [N, wn] = signal.cheb2ord(fPass, fStop, aPass, aStop)
        sos = signal.cheby2(N, aStop, wn, 'highpass', output='sos')

    else:
        # band-pass filter
        # print("Using bandpass filter")
        fPass = (np.array(bandFiltCutF) / nFreq).tolist()
        fStop = [(bandFiltCutF[0] - filtAllowance[0]) / nFreq, (bandFiltCutF[1] + filtAllowance[1]) / nFreq]
        # find the order
        [N, wn] = signal.cheb2ord(fPass, fStop, aPass, aStop)
        sos = signal.cheby2(N, aStop, wn, 'bandpass', output='sos')

    dataOut = signal.sosfilt(sos, data, axis=axis)
    # dataOut = signal.sosfiltfilt(sos, data, axis=axis)

    return dataOut





if __name__ == '__main__':

    fs = 100 # sampling frequency
    bandFiltCutF = [8, 13] # filtering frequency
    C = 16
    T = 180
    eeg_data = np.random.randn(100, C, T)
    #using cheby2_sos to filter the signal
    x = bandpassfilter_cheby2_sos_torch(eeg_data, bandFiltCutF, fs)
    # 绘制提取的Alpha信号
    t = x[0][0]
    plt.figure(figsize=(10, 6))
    plt.plot(x[0][0])
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Extracted Alpha EEG Signal')
    plt.show()


