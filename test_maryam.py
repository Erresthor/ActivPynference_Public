import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

def DecodePreStim(SUBJ):
    # my function here
    data = sio.loadmat('/mnt/data/Maryam/output/' + str(SUBJ) +'/MeanPreStim/Mean_AllCond_Vis.mat')

    AllData1 = data['Mean']
    Label1 = data['Label']

    ACC1 = []
    Y1 = Label1[:, 0]
    CI = []

    model = SVC(kernel='linear', C=10, max_iter=100000)
    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=10)
    MAX_NUMBER_OF_TIMESTEPS = AllData1[0].shape[0]
    for i in range(MAX_NUMBER_OF_TIMESTEPS):         # Easy:161,  Diff:161
        print(i)
        data_input1 = np.transpose(AllData1[0][i])
        X1 = scaler.fit_transform(data_input1)
        # data_input2 = np.transpose(AllData2[0][i])
        # X2 = scaler.fit_transform(data_input2)
        skf.get_n_splits(X1, Y1)
        acc1 = []
        # acc2 = []
        for train_index, test_index in skf.split(X1, Y1):
            model.fit(X1[train_index], Y1[train_index])
            score1 = model.score(X1[test_index], Y1[test_index])
            # model.fit(X2[train_index], Y2[train_index])
            # score2 = model.score(X2[test_index], Y2[test_index])
            acc1.append(score1)
            # acc2.append(score2)
        ACC1.append(np.mean(acc1))
        # ACC2.append(np.mean(acc2))
        CI.append(np.percentile(acc1, [2.5, 97.5]))
        print(ACC1)
    
    with open("/mnt/data/Maryam/output/" + str(SUBJ) +"/ACC_Vis", "wb") as fp:   #Pickling
        pickle.dump(ACC1, fp)
    with open("/mnt/data/Maryam/output/" + str(SUBJ) +"/CI_Vis", "wb") as fp:   #Pickling
        pickle.dump(CI, fp)

    output_path = '/mnt/data/Maryam/output/' + str(SUBJ) +'/DecodingV100.svg'
    save_figure_from_data(ACC1,CI,output_path)
    print("Wouhou, I've done my job !")

def save_figure_from_data(data_ACC1,data_CI,saveto):
    lowlimit = -500
    highlimit = 5
    timeinterval = 5
    CI1 = [pair[0] for pair in data_CI] 
    CI2 = [pair[1] for pair in data_CI] 

    data_ACC1_arr = np.array(data_ACC1)
    data_CI1_arr = np.array(CI1)
    data_CI2_arr = np.array(CI2)
    t = np.arange(lowlimit, highlimit, timeinterval)
    t = t[:len(data_ACC1)]
    ## Confidence Interval
    #ci1 = 1.96 * np.std(ACC1)/np.sqrt(len(t))    ##ACC_EasyFast, ACC_DiffFast
    # ci2 = 1.96 * np.std(ACC3)/np.sqrt(len(t))    ##ACC_EasySlow, ACC_DiffSlow

    plt.plot(t, data_ACC1, 'r', label='V100 Trials')      ##143, 140
    plt.fill_between(t, (data_ACC1_arr-data_CI1_arr), (data_ACC1_arr+data_CI2_arr), color='r', alpha=.3)    ##color='r'
    plt.plot(t, data_ACC1_arr, color='black') 

    # plt.plot(t, chancelevel1, 'r--', label='ChanceLevel')     ##ChanceL_EasyFast
    # # plt.plot(t, chancelevel2, 'r--',  label='ChanceSlow')    ##ChanceL_EasySlow

    plt.tick_params(axis='both', labelsize=12)
    # plt.ylim(0.4, 0.1)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Time(ms)', fontsize=14)
    plt.title('Decoding PreStim: V100 Visual Channels Subj106', fontsize=14)
    plt.legend(fontsize=14)
    plt.text(-90, 0.85, 'Target onset',  fontsize=14)   ##Target onset
    plt.axvline(x = 0, ymin=0, ymax=0.88, color = 'black', ls='--', label = 'vline_multiple - full height')
    
    # Save the figure in SVG format
    plt.savefig(saveto, format='svg')
    print("Figure has been saved to " + saveto)

if __name__ == '__main__':
    input_arguments = sys.argv

    name_of_script = input_arguments[0]
    subject_id = int(input_arguments[1])
    print("Now processing subject " + str(subject_id)+".")
    DecodePreStim(subject_id)
