## define a dataset loader
import torch
from torch.utils.data import Dataset
import pandas as pd
import wfdb
import os
import numpy as np

import neurokit2 as nk
from sklearn.preprocessing import MultiLabelBinarizer
import ast
from multi_modal_heart.ECG.ecg_utils import get_median_wave_value, standardize_multi_lead_ecg, detect_r_peaks_from_multiple_lead,resample_ECG_signal, mask_signal, distort_multi_lead_signals
from multi_modal_heart.ECG.utils import get_age_group_code
class ECGDataset(Dataset):
    def __init__(self, data_root = "/home/engs2522/project/lib/ECG/ecg_ptbxl_benchmarking/data/ptbxl/",
                       label_csv_path ="/home/engs2522/project/lib/ECG/ptbxl_feature_benchmark/data/organized/high_quality/train_statement.csv",
                       n_leads =12,
                       max_seq_len = 1024,
                       sampling_rate = 500,
                       augmentation = True,
                       use_median_wave= False,
                       data_proc_config={
                           "if_clean":False,
                           "reverse_avR":False,
                       },
                       data_aug_config={
                            "noise_frequency_list":[5,20,100,150,175],
                            "noise_amplitude_range":[0.,0.2],
                            "powerline_amplitude_range":[0.,0.05],
                            "artifacts_amplitude_range":[0.,0.3],
                            "artifacts_number_range":[0,3],
                            "powerline_frequency_list":[5,10],
                            "artifacts_frequency_list":[5,10],
                            "linear_drift_range":[0.,0.5],
                            "random_prob":0.5,
                            "if_mask_signal":False,
                            "mask_whole_lead_prob":0.1,
                            "lead_mask_prob":0.2,
                            "region_mask_prob":0.15,
                            "mask_length_range":[0.08, 0.18],
                            "mask_value":0.0,
                            "random_drop_half_prob":0.0
                       }
                       ):
        self.data_root = data_root
        self.label_csv_df = pd.read_csv(label_csv_path)
        # missing_indexes = self.scan_median_wave_files()
        # self.label_csv_df = self.label_csv_df.drop(missing_indexes)
        ## remove rows with missing files:

        self.use_median_wave = use_median_wave
        if self.use_median_wave:
            # filter out those without median beats
            self.label_csv_df = self.label_csv_df[self.label_csv_df["median_exist"]]
        ## initialization:
        self.augmentation=augmentation
        self.pid =0
        self.n_leads = n_leads
        self.max_seq_len = max_seq_len
        self.sampling_rate = sampling_rate
        self.data_proc_config = data_proc_config
        self.data_aug_config = data_aug_config

        ## get binarizer for class encoding
        mlb = MultiLabelBinarizer()
        result = mlb.fit_transform([["NORM", "MI", "HYP","STTC","CD"]])
# print (binarizer.classes_)
        self.super_class_label_converter = mlb
        self.super_classes_labels = mlb.classes_

    
    def __len__(self):
        return len(self.label_csv_df)
    
  
    def get_age_code(self, idx):
        '''
        return a 9-dim one hot vector for age group
        '''
        age = self.label_csv_df.iloc[idx]["age"]
        age_code = get_age_group_code(age) ## label 1-8, categorical age label, label=0, unknown age
        ## make as one hot vector
        age_code = torch.nn.functional.one_hot(torch.tensor(age_code),num_classes=9)
        return age_code

    def get_sex(self, idx):
        '''
        return sex code (0: male, 1: female, 0.5: unknown, prefer not to say)
        '''
        sex = self.label_csv_df.iloc[idx]["sex"]
        if sex==0 or sex==1:
            sex_binary_code = torch.from_numpy(sex)
        else:
            ## for unknown sex, make it as 0.5
            sex_binary_code =torch.float(0.5)
        return sex_binary_code ## 0 or 1

    def get_raw_data(self, idx):
        '''
        get raw ECG data from the local file system
        return a 12*5000 numpy array if the sampling rate is 500 Hz or 12*1000 numpy array if the sampling rate is 100 Hz
        '''

        if self.use_median_wave:
            find_median  = False
            f = self.label_csv_df.iloc[idx]["filename_median"]
            try:
                find_median = True
                recording,meta = wfdb.rdsamp(self.data_root+f)
                if recording.shape[1]!=12:
                    find_median = False
                    Warning(f"{f} lead missing")

            except:
                find_median = False
                raise FileNotFoundError(f"{f} file not found")
                ## search for another file
           
            #     ## missing value
            #     recording = np.zeros_like((600,12))
            #     meta = {}
            #     meta["fs"] = 500
            #     # if idx < len(self.label_csv_df)-1:
                #     signal 
                ## load 500 hz data and compute the median wave using automatic algorithm
                # f = self.label_csv_df.iloc[idx]["filename_hr"]
                # recording,meta = wfdb.rdsamp(self.data_root+f)
                # recording = get_median_wave_value(recording,sampling_rate=meta["fs"],lead_axis=1,output_length=600)
        else:
            if self.sampling_rate==500:
                ## load data from local file system;
                f = self.label_csv_df.iloc[idx]["filename_hr"]
                recording,meta = wfdb.rdsamp(self.data_root+f)
            elif self.sampling_rate==100:
                f = self.label_csv_df.iloc[idx]["filename_lr"]
                recording,meta = wfdb.rdsamp(self.data_root+f)
            else:
                raise ValueError("sampling rate should be 500 or 100")
        ## transpose the recording to 12*5000
        original_recording = recording.T
        current_sampling_rate = meta["fs"]
        if current_sampling_rate==self.sampling_rate:
            pass
        else:
            ## resample the signal
            original_recording = resample_ECG_signal(original_recording, current_sampling_rate, self.sampling_rate)
        return original_recording


    def __getitem__(self, idx):
        self.pid = idx
        original_recording = self.get_raw_data(idx)
        reference_sample =self.preprocess_data(original_recording, if_clean=True) ## use clean signal as reference

        ## add data augmentation:
        if self.augmentation:
            augmented_signals, mask = self.augment_data(original_recording)
        else:
            augmented_signals = original_recording
            mask = np.ones_like(original_recording)
            
        input_sample = self.preprocess_data(augmented_signals)# 12*5000

        ## augmented version for input
        signal_length = input_sample.shape[1]
     
        if self.max_seq_len*2<signal_length:
            start_pos = np.random.randint(0, signal_length-self.max_seq_len*2)
            current_seq = input_sample[:,start_pos:start_pos+self.max_seq_len]
            cleaned_seq = reference_sample[:,start_pos:start_pos+self.max_seq_len]
            mask_seq = mask[:,start_pos:start_pos+self.max_seq_len]
            ## cleaned version for reference
            next_seq = reference_sample[:,start_pos+self.max_seq_len:start_pos+2*self.max_seq_len]
        elif self.max_seq_len*2>signal_length and self.max_seq_len<signal_length:
            Warning("max_seq_len <signal length< 2*max_seq_len, will only return a random section of signal in the length of <max_seq_len>")
            start_pos = np.random.randint(0, signal_length-self.max_seq_len)
            current_seq = input_sample[:,start_pos:start_pos+self.max_seq_len]
            cleaned_seq = reference_sample[:,start_pos:start_pos+self.max_seq_len]
            mask_seq = mask[:,start_pos:start_pos+self.max_seq_len]
            next_seq = None
        else:
            Warning("max_seq_len > signal length, will return pad the whole signal to match the dim")
            ## return the whole signal
            current_seq = np.zeros((input_sample.shape[0],self.max_seq_len))
            width = (self.max_seq_len-signal_length)//2
            current_seq[:,width:width+signal_length] = input_sample

             ## return the whole signal
            cleaned_seq = np.zeros((input_sample.shape[0],self.max_seq_len))
            cleaned_seq[:,width:width+signal_length] = reference_sample

            ## mask length
            mask_seq = np.zeros((input_sample.shape[0],self.max_seq_len))
            mask_seq[:,width:width+signal_length] = mask
            next_seq = None

        ## lead selection:
        if self.n_leads<12:
            if self.n_leads ==8:
                ## use only 1,2, V1-V6.
                current_seq = current_seq[[0,1,6,7,8,9,10,11]]
                cleaned_seq = cleaned_seq[[0,1,6,7,8,9,10,11]]
                mask_seq = mask_seq[[0,1,6,7,8,9,10,11]]
            else: raise NotImplementedError("n_leads should be 8 or 12")
        elif self.n_leads==12:
            pass
        else:
            raise ValueError("n_leads should be less than 12")

        sample = torch.from_numpy(np.array(current_seq)).float()
        cleaned_sample = torch.from_numpy(np.array(cleaned_seq)).float()
        mask_sample = torch.from_numpy(np.array(mask_seq)).float() ## 12*800
        
   
     
        super_class, super_class_encoding = self.get_superclass_label(idx)
        super_class_encoding = torch.from_numpy(np.array(super_class_encoding)).long()

      

        if next_seq is not None:
            next_sample = torch.from_numpy(np.array(next_seq)).float()
            # next_sample = next_sample.unsqueeze(1) ##
            data_dict = {
                    "id":idx,
                    "input_seq":sample,
                    "cleaned_seq":cleaned_sample,
                    "mask":mask_sample, ## lead*seq_len
                    "next_seq":next_sample,
                    # "meta":meta,
                    # "super_class":super_class,
                    "super_class_encoding":super_class_encoding
                } 
        else:
            data_dict =  {
                    "id":idx,
                    "input_seq":sample,
                    "cleaned_seq":cleaned_sample,
                    "mask":mask_sample,
                    "super_class_encoding":super_class_encoding
                } 
        row = self.label_csv_df.iloc[idx]
        ## extended information
        if "ecg_id" in self.label_csv_df.columns:
            ecg_Id = row["ecg_id"]
            data_dict["ecg_id"] = ecg_Id

        if "report" in self.label_csv_df.columns:
            report = row["report"]
            data_dict["report"] = report
        return data_dict

        
    def get_superclass_label(self,idx):
        if idx is None:
            idx = self.pid
        try:
            self.label_csv_df['diagnostic_superclass'] =self.label_csv_df['diagnostic_superclass'].apply(ast.literal_eval)
        except:
            pass
        row = self.label_csv_df.iloc[idx]
        class_strings = row["diagnostic_superclass"]
        super_class_encoding = self.super_class_label_converter.transform([class_strings])[0]
        super_class_encoding = np.array(list(super_class_encoding)).astype(np.int32)
        ## map super class to integer
        return class_strings,super_class_encoding
    
    def preprocess_data(self,recording, if_clean=None):
        '''
        perform basic preprocessing on the ECG data including: bandpass filtering (if clean is true), z-score standardization
        '''
        assert self.data_proc_config is not None, "data_proc_config should be specified"
        if if_clean is None:
            if_clean = self.data_proc_config["if_clean"]
        if abs(np.sum(recording)-0)<1e-5:
            return recording
        recording = standardize_multi_lead_ecg(recording, sampling_rate=self.sampling_rate,if_clean=if_clean)# 12*5000
        return recording
    
    def augment_data(self,recording):
        '''
        augment ECG data with a series of data augmentation methods
        Input:
            recording: 12*L
            Output: 12*L, mask 12*L indicating the masked lead (note, region is not indicated in the mask, as we simulate the scenario that in real world this is not available)
        '''
        assert self.data_aug_config is not None, "data_aug_config should be specified"
        # print ("augmented")
        # if abs(np.sum(recording)-0)<1e-5:
        #     return recording, np.zeros_like(recording)
        noise_frequency_list = self.data_aug_config["noise_frequency_list"]
        noise_amplitude = np.random.random()*(self.data_aug_config["noise_amplitude_range"][1]-(self.data_aug_config["noise_amplitude_range"][0]))+self.data_aug_config["noise_amplitude_range"][0]
        powerline_amplitude = np.random.random()*(self.data_aug_config["powerline_amplitude_range"][1]-(self.data_aug_config["powerline_amplitude_range"][0]))+self.data_aug_config["powerline_amplitude_range"][0]
        powerline_frequency = np.random.choice(self.data_aug_config["powerline_frequency_list"])
        artifacts_amplitude = np.random.random()*(self.data_aug_config["artifacts_amplitude_range"][1]-(self.data_aug_config["artifacts_amplitude_range"][0]))+self.data_aug_config["artifacts_amplitude_range"][0]
        artifacts_number = np.random.randint(self.data_aug_config["artifacts_number_range"][0], self.data_aug_config["artifacts_number_range"][1]+1) 
        linear_drift = np.random.random()*(self.data_aug_config["linear_drift_range"][1]-(self.data_aug_config["linear_drift_range"][0]))+self.data_aug_config["linear_drift_range"][0]
        random_prob = self.data_aug_config["random_prob"]
        artifacts_frequency_list = self.data_aug_config["artifacts_frequency_list"]
        
        artifacts_frequency = np.random.choice(artifacts_frequency_list)
        distorted_recording = recording.copy()
        distorted_recording = distort_multi_lead_signals(distorted_recording, self.sampling_rate, noise_frequency=noise_frequency_list, 
                                                         noise_amplitude=noise_amplitude,powerline_amplitude=powerline_amplitude,
                                                         artifacts_amplitude=artifacts_amplitude,artifacts_number=artifacts_number,linear_drift=linear_drift,
                                                         random_prob=random_prob,
                                                         artifacts_frequency=artifacts_frequency)
        if_mask_signal = self.data_aug_config["if_mask_signal"]
    
         
        if if_mask_signal:
                # print ("masked")
                if "random_drop_half_prob" in self.data_aug_config.keys():
                    random_drop_half_prob = self.data_aug_config["random_drop_half_prob"]
                else: random_drop_half_prob = 0.0
                region_mask_prob = self.data_aug_config["region_mask_prob"]
                lead_mask_prob = self.data_aug_config["lead_mask_prob"]
                mask_whole_lead_prob = self.data_aug_config["mask_whole_lead_prob"]
                mask_length_range_start = self.data_aug_config["mask_length_range"][0]
                mask_length_range_end = self.data_aug_config["mask_length_range"][1]

                if not abs(region_mask_prob-0.0)<1e-5:
                    critical_points = detect_r_peaks_from_multiple_lead(recording, self.sampling_rate,method="promac",use_lead_index=1) ## use lead II to detect R peaks
                else:
                    critical_points = None
                masked_signals, mask= mask_signal(distorted_recording, critical_points, 
                                                mask_whole_lead_prob = mask_whole_lead_prob,
                                                lead_mask_prob = lead_mask_prob,
                                                region_mask_prob=region_mask_prob, 
                                                mask_length_range=[mask_length_range_start, mask_length_range_end], sampling_rate=self.sampling_rate, 
                                                mask_value=0.0,random_drop_half_prob=random_drop_half_prob)
                augmented_signals = masked_signals
        else:
                augmented_signals = distorted_recording
                mask = np.ones_like(recording)
        return augmented_signals, mask
