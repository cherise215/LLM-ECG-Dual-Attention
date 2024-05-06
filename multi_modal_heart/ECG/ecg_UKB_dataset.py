## define a dataset loader
import torch
from torch.utils.data import Dataset
import pandas as pd
import wfdb
import os
import numpy as np

import neurokit2 as nk
from sklearn.preprocessing import MultiLabelBinarizer
from multi_modal_heart.ECG.ecg_dataset import ECGDataset
from multi_modal_heart.ECG.ecg_utils import extract_ECG_data_from_xml_file, convert_lead_dict_to_array, resample_ECG_signal

class ECGUKBDataset(ECGDataset):
    def __init__(self, data_root = "/home/engs2522/project/lib/ECG/ecg_ptbxl_benchmarking/data/ptbxl/",
                       label_csv_path ="/home/engs2522/project/lib/ECG/ptbxl_feature_benchmark/data/organized/high_quality/train_statement.csv",
                       n_leads =12,
                       max_seq_len = 1024,
                       sampling_rate = 500,
                       use_median_wave= False,
                       augmentation = True,
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
        super().__init__(data_root = data_root,
                          label_csv_path =label_csv_path,
                            n_leads =n_leads,
                            max_seq_len = max_seq_len,
                            sampling_rate = sampling_rate,
                            augmentation = augmentation,
                            data_proc_config=data_proc_config,
                            data_aug_config=data_aug_config)
  
         ## get binarizer for class encoding
        mlb = MultiLabelBinarizer()
        self.use_median_wave = use_median_wave
        result = mlb.fit_transform([["NORM", "HF"]])
        # print (binarizer.classes_)
        self.super_class_label_converter = mlb
        self.super_classes_labels = mlb.classes_
    
    def get_raw_data(self, idx):
        '''
        get raw ECG data from the local file system
        return a 12*5000 numpy array if the sampling rate is 500 Hz or 12*1000 numpy array if the sampling rate is 100 Hz
        '''
        f = self.statement_info.iloc[idx]["ECG_xmlfile_path"]
        meta_dict, wave_dict = extract_ECG_data_from_xml_file(f,if_median=self.use_median_wave)
        original_sample_rate = float(meta_dict["sample_rate"])
        if self.sampling_rate==original_sample_rate:
            new_wave = resample_ECG_signal(wave_dict, original_sample_rate, new_sample_rate=float(self.sampling_rate))
        else:
            new_wave = wave_dict
        recording = convert_lead_dict_to_array(new_wave)
        return recording

    def get_superclass_label(self,idx):
        if idx is None:
            idx = self.pid
        row = self.label_csv_df.iloc[idx]
        class_strings = row["diagnostic_superclass"]
        super_class_encoding = self.super_class_label_converter.transform([class_strings])[0]
        print (super_class_encoding)
        # super_class_encoding = np.array(list(super_class_encoding)).astype(np.float32)
        ## map super class to integer
        return class_strings,super_class_encoding
   