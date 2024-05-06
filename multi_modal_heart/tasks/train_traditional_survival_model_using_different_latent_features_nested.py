
# prepare data for MI-> HF prediction
## load data from model features of subjects which have been correctly predicted as MI classes (true positive)
## get the HF status and time of these subjects [MI-> diagnosis of HF]
## run the Cox regression model and visualize the cph coefficients
import argparse
import os, sys, time, pickle, copy, h5py
import numpy as np, pandas as pd

import optunity
from lifelines.plotting import plot_lifetimes
import lifelines.utils
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from sklearn.model_selection import KFold

sns.set_style("whitegrid")

os.environ["CUDA_AVAILABLE_DEVICES"] = "1"

def coxreg_single_run(xtr, ytr, penalty):
    '''
    xtr: training data, shape (n_samples, n_features)
    ytr: training label, shape (n_samples, 2), 1st column is status, 2nd column is time
    penalty: penalizer for Cox regression
    # l1_ratio: l1 ratio for Cox regression
    see: https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html
    return: Cox regression model, training data
    '''
    df_tr = pd.DataFrame(np.concatenate((ytr, xtr),axis=1))
    df_tr.columns = ['status','time'] + ['X'+str(i+1) for i in range(xtr.shape[1])]
    cph = CoxPHFitter(penalizer=penalty)
    cph.fit(df_tr, duration_col='time', event_col='status')
    
    return cph,df_tr

def hypersearch_cox(x_data, y_data, method, nfolds, nevals, penalty_range):
    '''
    x_data: training data, shape (n_samples, n_features)
    y_data: training label, shape (n_samples, 2), 1st column is status, 2nd column is time
    method: optimization method, e.g. 'particle swarm'
    nfolds: number of folds for cross validation
    nevals: number of evaluations for hyperparameter search
    penalty_range: range of penalizer for Cox regression
    return: optimal hyperparameters, search log
    '''
    @optunity.cross_validated(x=x_data, y=y_data, num_folds=nfolds, num_iter=2)
    def modelrun(x_train, y_train, x_test, y_test, penalty):
        cvmod,df_tr = coxreg_single_run(xtr=x_train, ytr=y_train, penalty=10 ** penalty)
        cv_preds = cvmod.predict_partial_hazard(x_test)
        cv_C = concordance_index(y_test[:, 1], -cv_preds, y_test[:, 0])
        return cv_C
    optimal_pars, searchlog, _ = optunity.maximize(modelrun, num_evals=nevals,
                                                   solver_name=method, penalty=penalty_range)
    print('Optimal hyperparameters : ' + str(optimal_pars))
    print('Cross-validated C after tuning: %1.3f' % searchlog.optimum)
    return optimal_pars, searchlog


def load_MI_HF_event_database(prevalent_MI_csv_path="/home/engs2522/project/multi-modal-heart/multi_modal_heart/toolkits/ukb/non_imaging_information/MI/prevalent_MI.csv",
                              hf_csv_path="/home/engs2522/project/multi-modal-heart/multi_modal_heart/toolkits/ukb/non_imaging_information/HF/HF_record_found_in_algorithmly_defined_I50_ICD10_ICD9_with_source_dates_ecg_cmr_paired.csv",
                              censor_time ='2023-01-06',
                              path_to_save_csv = '/home/engs2522/project/multi-modal-heart/multi_modal_heart/toolkits/ukb/non_imaging_information/MI/MI_HF_coxreg_eid.csv'
                            ):
    '''
    load the MI and HF information from csv files, and merge them together
    prevalent_MI_csv_path: path to the csv file of prevalent MI:[eid, MI_date,ecg_date]
    hf_csv_path: path to the csv file of HF: [eid, HF-date, ecg_date]
    return: dataframe with eid, HF_status, time_to_HF
    '''
    prevalent_MI_information_df = pd.read_csv(prevalent_MI_csv_path)
    hf_df = pd.read_csv(hf_csv_path)
    MI_HF_df = pd.merge(prevalent_MI_information_df, hf_df, on='eid', how='inner')

    ## eids of subjects with HF already before MI
    MI_HF_df['MI_date'] = MI_HF_df['MI_date'].apply(pd.to_datetime).dt.date
    MI_HF_df['HF-date'] = MI_HF_df['HF-date'].apply(pd.to_datetime).dt.date
    MI_HF_df['ecg_date'] = MI_HF_df['ecg_date'].apply(pd.to_datetime).dt.date

    HF_after_ECG = MI_HF_df[MI_HF_df['HF-date'] >MI_HF_df['ecg_date']]
    HF_before_ECG = MI_HF_df[MI_HF_df['HF-date']<=MI_HF_df['ecg_date']]

    print ('find ', len(HF_after_ECG), 'HF after ECG (incident) records')
    print ('find ', len(HF_before_ECG), 'HF before ECG (prevalent) records')
    assert len(HF_after_ECG)>0, "no HF after ECG records found"
    MI_HF_df['time_to_HF'] = MI_HF_df['HF-date'] - MI_HF_df['ecg_date']
    HF_after_ECG['time_to_HF'] = HF_after_ECG['HF-date'] - HF_after_ECG['ecg_date']
    # HF_after_ECG['time_to_HF_(weeks)'] = HF_after_ECG['time_to_HF'].dt.days/7


    HF_after_ECG['time_to_HF'] = HF_after_ECG['HF-date'] - HF_after_ECG['ecg_date']
    HF_after_ECG['time_to_HF'] = HF_after_ECG['time_to_HF'].apply(lambda x: x.days)
    prevalent_MI_information_df_filtered = prevalent_MI_information_df[prevalent_MI_information_df.eid.isin(HF_before_ECG.eid)==False]
    # compute duration time of follow-up study, censor time is 2023-01-06
    ## convert to datetime
    prevalent_MI_information_df_filtered['censor_time'] = censor_time
    prevalent_MI_information_df_filtered['censor_time'] = prevalent_MI_information_df_filtered['censor_time'].apply(pd.to_datetime).dt.date
    prevalent_MI_information_df_filtered['ecg_date'] =prevalent_MI_information_df_filtered['ecg_date'].apply(pd.to_datetime).dt.date
    # convert to datetime
    prevalent_MI_information_df_filtered['HF_status'] = 0
    prevalent_MI_information_df_filtered['time_to_HF'] = prevalent_MI_information_df_filtered["censor_time"]-prevalent_MI_information_df_filtered['ecg_date']
    ## for those subjects who have HF after MI, set HF_status to 1, and recalculate the time to HF
    prevalent_MI_information_df_filtered.loc[prevalent_MI_information_df_filtered.eid.isin(HF_after_ECG.eid), 'HF_status'] = 1
    prevalent_MI_information_df_filtered.loc[prevalent_MI_information_df_filtered.eid.isin(HF_after_ECG.eid), 'time_to_HF'] = HF_after_ECG['time_to_HF'].values    # %%
    ## create a new dataframe for Cox regression
    ## eid, HF_status, time_to_HF
    MI_HF_coxreg_df = prevalent_MI_information_df_filtered[['eid','HF_status','time_to_HF']]
    MI_HF_coxreg_df.time_to_HF = MI_HF_coxreg_df.time_to_HF.apply(lambda x: int(np.ceil(x.days/7)))
    MI_HF_coxreg_df.to_csv(path_to_save_csv, index=False)
    return MI_HF_coxreg_df

def load_features_from_disk(model_name,feature_file_name="test_last_hidden_feature.npy"):
    feature = np.load(f'/home/engs2522/project/multi-modal-heart/multi_modal_heart/tasks/finetune_on_MI/{model_name}/feature/{feature_file_name}.npy')
    eid_list = np.load(f'/home/engs2522/project/multi-modal-heart/multi_modal_heart/tasks/finetune_on_MI/{model_name}/feature/test_eid.npy')
    # print("feature size", feature.shape)
    # print("eid size", len(eid_list))
    assert feature.shape[0] == len(eid_list), "feature and eid size not match, but got {} and {}".format(feature.shape[0], len(eid_list))
    return feature, eid_list

if __name__=='__main__':

    ## args set up
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_hyperparam",action="store_true")
    ## hyperparameters for survival model training
    parser.add_argument("--penalty",type=float,default=1e-2)
    parser.add_argument("--nfolds",type=int,default=2)
    parser.add_argument("--seed",type=int,default=42)
    args = parser.parse_args()

    def run_model(model_name,feature_file_name, nfolds, seed=42): 
        save_dir = f'/home/engs2522/project/multi-modal-heart/multi_modal_heart/tasks/finetune_on_MI/{model_name}/feature/{feature_file_name}/{seed}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        feature, eid_list = load_features_from_disk(model_name, feature_file_name=feature_file_name)
        MI_HF_coxreg_df = pd.read_csv('/home/engs2522/project/multi-modal-heart/multi_modal_heart/toolkits/ukb/non_imaging_information/MI/MI_HF_coxreg_df.csv')
        mi_hf_eid_list = MI_HF_coxreg_df.eid.apply(int).values.tolist()
        ## find the features of subjects with eids in the MI_HF_coxreg_df
        filtered_feature ={}
        unique_eid_list = list(set(eid_list))
        assert len(unique_eid_list) == len(eid_list), "duplicate eid found"
        for eid,feature_i in zip(eid_list, feature):
            if int(eid) in mi_hf_eid_list:
                filtered_feature[int(eid)] = feature_i
            else:
                continue
                # print(f'{eid_list[i]} not found')
        print(len(filtered_feature))
        filtered_feature_df = pd.DataFrame.from_dict(filtered_feature, orient='index')
        ## set eid column as index
        filtered_feature_df.index.name = 'eid'  
        filtered_feature_df.reset_index(inplace=True)    
        MI_HF_coxreg_df_feature = pd.merge(MI_HF_coxreg_df, filtered_feature_df, on='eid', how='inner')

        ## extract items with column 0:127 as features
        x = MI_HF_coxreg_df_feature.iloc[:,3:].values
        ## get the state and time to HF as label
        y = MI_HF_coxreg_df_feature.iloc[:,1:3].values

        random_seed = seed
        y_status = y[:,0]
        ## shuffle the data (x,y) together

        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        # Initialize lists to store predictions
        val_c_index_list = []
        i =0
   
        for cval, (train_indices, test_indices) in enumerate(kf.split(x, y_status)):
            
            x_train, y_train = x[train_indices], y[train_indices]
            x_test, y_test = x[test_indices], y[test_indices]
            print("Step 1a")
            cval_dir = f"{save_dir}/cval_{str(cval)}"
            if not os.path.exists(cval_dir):os.makedirs(cval_dir)
            if args.search_hyperparam:
                #---------------------start hyperparam search----------------#
                opars, osummary = hypersearch_cox(x_data=x_train, y_data=y_train, method='particle swarm', 
                                             nfolds=nfolds, nevals=50, penalty_range=[-7,-2])
                 # using the optimal hyperparameters to train the model again on the whole dataset
                with open(f'{cval_dir}/hyperparam.text', "w") as f:
                ## opars dict is the optimal hyperparameters and save it to the txt file
                    for k, v in opars.items():
                        f.write(str(k))
                        f.write(":")
                        f.write(str(v))
                        f.write("\n")
                penalty=10**opars['penalty']
            else:
                penalty = args.penalty
            cph,df = coxreg_single_run(x_train, y_train, penalty=penalty)
            cph.print_summary()
             # # compute harrel's concordance index
            pred_test = cph.predict_partial_hazard(x_test)
            C_app = concordance_index(x_test[:,1], -pred_test, x_test[:,0])
            print('\n\n==================================================')
            print('Apparent concordance index = {0:.4f} for cval {1}'.format(C_app,cval))
            print('==================================================\n\n')
             ##save the summary
            if args.search_hyperparam:
                pd.DataFrame(cph.summary).to_csv(f'{cval_dir}/MI_HF_coxreg_summary_cindex{np.round(C_app,4)}_optimi_penalty_{str(np.round(penalty,4))}.csv')
            else:
                pd.DataFrame(cph.summary).to_csv(f'{cval_dir}/MI_HF_coxreg_summary_cindex{np.round(C_app,4)}_fixed_penalty_{str(np.round(penalty,4))}.csv')

            plt.figure(figsize=(10,10), dpi=500)
            cph.plot(hazard_ratios=True)
            ##save the figure
            plt.savefig(f'{cval_dir}/MI_HF_coxreg.png', dpi=500, bbox_inches='tight')
            plt.close()


            ## find the feature with the largest coefficient
            id = cph.summary["coef"].sort_values(ascending=False).head(1).index.values[0]
            print(id)
            cph.plot_partial_effects_on_outcome(covariates=id, values=[-3,-1.5,-1, -0.5, -0.25,0, 0.25,0.5, 1,1.5,3], cmap='coolwarm')
            ## save the figure
            plt.savefig(f'{cval_dir}/partial_effect_on_{id}.png', dpi=300, bbox_inches='tight')    
            plt.close()

            ## save the model
            with open(f'{cval_dir}/MI_HF_coxreg_model.pkl', 'wb') as f:
                pickle.dump(cph, f)
            val_c_index_list.append(C_app)
        return val_c_index_list
      
   
    opars = None

    model_name_list=[
     
        "ECG_attention",
        "ECG_attention_pretrained_on_recon",
        "ECG_attention_pretrained_on_recon_ECG2Text",
        "ECG_attention_pretrained_on_classification+ECG2Text"
        # "resnet1d101_512",
        # "resnet1d101_512_pretrained_recon",
        # "resnet1d101_512_pretrained_recon+ECG2Text",

    ]
    feature_file_name_list = [
        "test_last_hidden_feature",
        "train_bottleneck_feature",
    ]
    result = []
    failed_list=[]
    seed_list = [42, 2021, 2022, 2023, 2024]
    for model_name in model_name_list:
        for feature_file_name in feature_file_name_list:
            seed_mean_cval_list = []
            for seed in seed_list:
                pl.seed_everything(seed)
                try:    
                    val_c_index_list= run_model(model_name, feature_file_name,nfolds=args.nfolds, seed=seed)
                    mean_cval =np.mean(val_c_index_list)
                    std_cval = np.std(val_c_index_list)
                    print(f"{model_name}_{feature_file_name}: {mean_cval}, {std_cval}")
                    result.append([f'{model_name}_{feature_file_name}',seed,mean_cval,std_cval])
                    seed_mean_cval_list.append(mean_cval)
                except Exception as e:
                    print(e)
                    failed_list.append([model_name, feature_file_name,seed])
            mean_cindex = np.mean(seed_mean_cval_list)
            std_cindex = np.std(seed_mean_cval_list)
            result.append([f'{model_name}_{feature_file_name}','AVG',mean_cindex,std_cindex])
    ## save the result as csv
    if not args.search_hyperparam:
        pd.DataFrame(result, columns=['model_name', 'seed', 'C_mean', 'C_std']).to_csv(f'/home/engs2522/project/multi-modal-heart/multi_modal_heart/tasks/finetune_on_MI/MI_HF_coxreg_result_penalty={str(np.round(args.penalty,4))}.csv', index=False)
    else:
        pd.DataFrame(result, columns=['model_name', 'seed','C_mean', 'C_std']).to_csv(f'/home/engs2522/project/multi-modal-heart/multi_modal_heart/tasks/finetune_on_MI/MI_HF_coxreg_result_optimized.csv', index=False)
    pd.DataFrame(failed_list, columns=['model_name', 'feature_file_name','seed']).to_csv(f'/home/engs2522/project/multi-modal-heart/multi_modal_heart/tasks/finetune_on_MI/MI_HF_coxreg_failed_list.csv', index=False)





