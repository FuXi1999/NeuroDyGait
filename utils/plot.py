import os
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib   

from scipy.signal import find_peaks

def plot_joint(sub, data_list, lable_list):
    for idx, joint_data in enumerate(data_list):
        sub.plot(joint_data, label=lable_list[idx])
        if lable_list[idx] == 'KL':
            peaks, _ = find_peaks(joint_data, height=20, distance=150)
            print(peaks)
            sub.plot(peaks, joint_data[peaks], 'x',  c = "r")
            sub.vlines(peaks, 10,  joint_data[peaks],  colors = "r", linestyles = "dashed")
            range_ = []
            for idx in range(len(peaks)-1): 
                range_.append(peaks[idx+1]-peaks[idx])
                plt.annotate(s='', xy=(peaks[idx]-1, 20), xytext=(peaks[idx+1]+1, 20),
                 arrowprops=dict(arrowstyle='<->', edgecolor="red", linestyle='--', shrinkA=1, shrinkB=1))
                plt.text(peaks[idx]+(peaks[idx+1]-peaks[idx])/2, 22, str((peaks[idx+1]-peaks[idx])/100)+'s', ha='center')
            print(range_, np.mean(range_))
    # sub.title(joint_name)
    sub.legend()
    
    return sub


def make_joints_figure(mat_path, start, last):
    start *= 100
    last *= 100
    data = sio.loadmat(mat_path)
    train_joints = data['trainJoints'][:,:6]

    used_joints = train_joints[start:start+last, :]
    joints_label = ['HR', 'KR', 'AR', 'HL', 'KL', 'AL']
    # GHR GKR GAR GHL GKL GAL
    fig = plt.figure(figsize=[16,4])
    for idx in range(3):
        sub = fig.add_subplot(1,3,idx+1)
        plot_joint(sub, [used_joints[:, idx], used_joints[:, idx+3]], [joints_label[idx], joints_label[idx+3]])
    # plt.show()
    plt.savefig(os.path.join('../results', 'joints.png'))

def plot_one(sub, csv_df_dic, matrics, joint, time_l=[20, 220, 20]):
    marker_list = ['^', '+', 'o', 'x', 'd', '2', '1']
    
    for i, csv_key in enumerate(csv_df_dic.keys()):
        csv_df = csv_df_dic[csv_key]
        used_col = []
        time_step_list = list(range(time_l[0], time_l[1], time_l[2]))
        used_time_step = []
        for time_step in time_step_list:
            col_name = matrics+'-TS'+str(time_step)+'-'+joint
            if col_name in csv_df.columns:
                used_time_step.append(time_step)
                used_col.append(col_name)
        
        mean_std = csv_df.loc[['Avg.', 'Std.'], used_col].values
        sub.errorbar(used_time_step, mean_std[0], mean_std[1], marker=marker_list[i], capsize=6, label=csv_key)
        sub.set_xlabel('Time Step [x10 ms]')
        sub.set_ylabel(matrics)
        sub.set_xticks(list(range(20,220,20)))
        sub.set_xticklabels([str(idx) for idx in range(20, 220, 20)])
        # sub.text(100, mean_std[0,0]-mean_std[1,0], joint, ha="center")
    sub.legend(loc='lower right')
    return sub


def plot(csv_file_list, joints_list, matrics):
    csv_df_dic = {}
    for csv_file in csv_file_list:
        if csv_file.split('/')[-1] in ['rdgeRegression.csv', 'lnearRegression.csv', 'lasso.csv']:
            csv_df_dic[csv_file.split('/')[-2]] = pd.read_csv(csv_file, index_col=0, header=0).fillna(0)
        else:
            csv_df_dic[csv_file.split('/')[-2]] = pd.read_csv(csv_file, index_col=0, header=0).T.fillna(0)
    fig = plt.figure(figsize=[16,8])
    for i, joint in enumerate(joints_list):
        sub = fig.add_subplot(2,3,i+1)
        plot_one(sub, csv_df_dic, matrics, joint)
    plt.savefig(os.path.join('../results', matrics+'.png'))

def make_table(csv_file_list, joints_list, matrics_list, time_l=[20,220,20]):
    csv_df_dic = {}
    for csv_file in csv_file_list:
        if csv_file.split('/')[-1] in ['ridgeRegresson.csv', 'linearegression.csv', 'lasso.csv']:
            csv_df_dic[csv_file.split('/')[-2]] = pd.read_csv(csv_file, index_col=0, header=0).fillna(0)
        else:
            csv_df_dic[csv_file.split('/')[-2]] = pd.read_csv(csv_file, index_col=0, header=0).T.fillna(0)
    writer = pd.ExcelWriter(os.path.join('../results', 'table.xlsx'),engine='openpyxl', mode='a')
    for matrics in matrics_list:
        for joint in joints_list:
            mean_std_list = []
            for i, csv_key in enumerate(csv_df_dic.keys()):
                csv_df = csv_df_dic[csv_key]
                used_col = []
                time_step_list = list(range(time_l[0], time_l[1], time_l[2]))
                used_time_step = []
                for time_step in time_step_list:
                        col_name = matrics+'-TS'+str(time_step)+'-'+joint
                        if col_name in csv_df.columns:
                            used_time_step.append(time_step)
                            used_col.append(col_name)
                    
                mean_std = csv_df.loc[['Avg.', 'Std.'], used_col]
                mean_std.index = [csv_key+'-mean', csv_key+'-std']
                mean_std_list.append(mean_std)
            
            pd.concat(mean_std_list, sort=False).to_excel(writer, sheet_name=matrics+'-'+joint)
    writer.save()

def get_average_pace(mat_path):
    mat_list = os.listdir(mat_path)
    mat_list.sort()
    pace_list = []
    fig, ax = plt.subplots(figsize=(20, 10))
    for subject in mat_list:
        data = sio.loadmat(os.path.join(mat_path, subject))
        train_joints = data['trainJoints'][:,4]
        peaks, _ = find_peaks(train_joints, height=20, distance=150)
        range_ = []
        for idx in range(len(peaks)-1): 
            range_.append(peaks[idx+1]-peaks[idx])
        print(subject, np.mean(range_))
        pace_list.append(np.mean(range_)/100)
    pace_list.append(np.mean(pace_list))
    plt.xticks(rotation=280, fontsize=10)
    plt.ylabel('Walking Cycle Time [s]')

    norm = plt.Normalize(1.2, 3)
    norm_values = norm(pace_list)
    map_vir = plt.cm.Purples
    colors = map_vir(norm_values)
    x_label = [item[:-4] for item in mat_list]
    x_label.append('Average')
    plt.bar(np.arange(len(mat_list)), pace_list[:-1], color=colors, edgecolor='black')
    plt.bar(24, pace_list[-1], color=colors, edgecolor='red')
    plt.text(24, pace_list[-1],'%.2fs'%pace_list[-1],ha='center',va='bottom',fontsize=10)
    ax.set_xticks(list(range(25)))
    ax.set_xticklabels(x_label)

    sm = matplotlib.cm.ScalarMappable(cmap=map_vir, norm=norm)
    sm.set_array([])

    fig.colorbar(sm)
    plt.savefig('../results/pace.png')


# file_list = ['../results/eegnet/eegnet.csv', '../results/LR/linearRegression.csv','../results/cnnlstm/cnnlstm.csv', '../results/lstm/lstm.csv', '../results/tcn/tcn.csv', ]
# file_list = ['../results/eegnet/eegnet.csv', '../results/eegnet_deemg/eegnet_deemg.csv']
file_list = ['../results/LR/linearRegression.csv', '../results/lr_deemg_remove/lr_deemg_remove.csv', '../results/lr_deemg_lapl/lr_deemg_lapl.csv']
# file_list = ['../results/LR/linearRegression.csv', '../results/lrp/lrp.csv',]
joints = ['HL', 'KL', 'AL', 'HR', 'KR', 'AR']
plot(file_list, joints, 'R2')

# matrics_list = ['COR', 'R2']
# make_table(file_list, joints, matrics_list)

# mat_path = '../data/raw_mat/SL04-T02.mat'
# make_joints_figure(mat_path, 400, 10)

# get_average_pace('../data/raw_mat')
