
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
    @optunity.cross_validated(x=x_data, y=y_data, num_folds=nfolds)
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

def bootstrap_survival_model(x_full, y_full, B,args=None):
    '''
    This function performs bootstrap resampling on a Cox regression model, to estimate the optimism of the model's C-index.
    Input: 
    x_full: full sample of features, shape (n_samples, n_features)
    y_full: full sample of labels, shape (n_samples, 2), 1st column is status, 2nd column is time
    B: number of bootstrap samples to generate
    Output:
    preds_bootfull: list of predictions on full sample, for each bootstrap sample
    inds_inbag: list of indices of samples in each bootstrap sample
    '''

    nsmp = len(x_full)
    rowids = [_ for _ in range(nsmp)]
    preds_bootfull = []
    inds_inbag = []
    Cb_opts = []
    for b in range(B):
        print('\n-------------------------------------')
        print('Current bootstrap sample:', (b+1), 'of', B)
        print('-------------------------------------')

        #STEP 2: Generate a bootstrap sample by doing n random selections with replacement (where n is the sample size)
        
        #Note, we make a stratified bootstrap sample, to ensure that the proportion of events is the same in the bootstrap sample as in the full sample
        # b_inds = np.random.choice(rowids, size=nsmp, replace=True)
        original_index = np.arange(nsmp)
        xboot, yboot,boot_index = resample(x_full, y_full, original_index,stratify=y_full, replace=True, random_state=b, n_samples=nsmp)
        #(2a) find optimal hyperparameters
        if args.search_hyperparam:
            bpars, bsummary = hypersearch_cox(x_data=xboot, y_data=yboot, method='particle swarm', nfolds=5, nevals=50, penalty_range=[-2,0])
        
            #(2b) using optimal hyperparameters, train a model on bootstrap sample
            bmod, df = coxreg_single_run(xtr=xboot, ytr=yboot, penalty=10**bpars['penalty'])
        else:
            bmod, df = coxreg_single_run(xtr=xboot, ytr=yboot, penalty=args.penalty)
            
        #(2c[i])  Using bootstrap-trained model, compute predictions on bootstrap sample. Evaluate accuracy of predictions (Harrell's Concordance index)
        predboot = bmod.predict_partial_hazard(xboot)
        Cb_boot = concordance_index(yboot[:,1], -predboot, yboot[:,0])
        
        #(2c[ii]) Using bootstrap-trained model, compute predictions on FULL sample.     Evaluate accuracy of predictions (Harrell's Concordance index)
        predbootfull = bmod.predict_partial_hazard(x_full)
        Cb_full = concordance_index(y_full[:,1], -predbootfull, y_full[:,0])

        #STEP 3: Compute optimism for bth bootstrap sample, as difference between results from 2c[i] and 2c[ii]
        Cb_opt = Cb_boot - Cb_full
    
        #store data on current bootstrap sample (predictions, C-indices)
        preds_bootfull.append(predbootfull)
        inds_inbag.append(original_index)
        Cb_opts.append(Cb_opt)

        del bpars, bmod
    ## averge the B bootstrap samples
    C_opt = np.mean(Cb_opts)

    return C_opt, Cb_opts, preds_bootfull, inds_inbag

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
    parser.add_argument("--if_bootstrap",action="store_true")
    parser.add_argument("--seed",type=int,default=42)
    args = parser.parse_args()

    def run_model(model_name,feature_file_name, if_bootstrap=False, seed=42): 
        pl.seed_everything(seed)
        save_dir = f'/home/engs2522/project/multi-modal-heart/multi_modal_heart/tasks/finetune_on_MI/{model_name}/feature/{feature_file_name}'
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
        MI_HF_coxreg_df_feature_only = MI_HF_coxreg_df_feature.iloc[:,3:].values
        ## get the state and time to HF as label
        MI_HF_coxreg_df_label = MI_HF_coxreg_df_feature.iloc[:,1:3].values
        #---------------------start hyperparam search----------------#
        if args.search_hyperparam:
            opars, summary = hypersearch_cox(x_data=MI_HF_coxreg_df_feature_only, y_data=MI_HF_coxreg_df_label, method='particle swarm', 
                                             nfolds=5, nevals=50, penalty_range=[-2,0])
        # using the optimal hyperparameters to train the model again on the whole dataset
            with open(f'{save_dir}/hyperparam.text', "w") as f:
            ## opars dict is the optimal hyperparameters and save it to the txt file
                for k, v in opars.items():
                    f.write(str(k))
                    f.write(":")
                    f.write(str(v))
                    f.write("\n")
            penalty=10**opars['penalty']
            cph,df = coxreg_single_run(MI_HF_coxreg_df_feature_only, MI_HF_coxreg_df_label, penalty=penalty)
        else:
            penalty = args.penalty
            cph,df = coxreg_single_run(MI_HF_coxreg_df_feature_only, MI_HF_coxreg_df_label, penalty=args.penalty)

        # # compute harrel's concordance index
        predfull = cph.predict_partial_hazard(MI_HF_coxreg_df_feature_only)
        C_app = concordance_index(MI_HF_coxreg_df_label[:,1], -predfull, MI_HF_coxreg_df_label[:,0])
        print('\n\n==================================================')
        print('Apparent concordance index = {0:.4f}'.format(C_app))
        print('==================================================\n\n')
        cph.print_summary()
        ##save the summary
        if args.search_hyperparam:
            pd.DataFrame(cph.summary).to_csv(f'{save_dir}/MI_HF_coxreg_summary_cindex{np.round(C_app,4)}_penalty_{str(np.round(penalty,4))}.csv')
        else:
            pd.DataFrame(cph.summary).to_csv(f'{save_dir}/MI_HF_coxreg_summary_cindex{np.round(C_app,4)}_optimi_penalty_{str(np.round(penalty,4))}.csv')

        plt.figure(figsize=(10,20))
        cph.plot(hazard_ratios=True)
        ##save the figure
        plt.savefig(f'{save_dir}/MI_HF_coxreg.png', dpi=500, bbox_inches='tight')
        plt.close()
        # # X = df.loc[0]
        # cph.predict_survival_function(X).rename(columns={0:'CoxPHFitter'}).plot()
        # ## save the figure
        # plt.savefig(f'/home/engs2522/project/multi-modal-heart/multi_modal_heart/tasks/finetune_on_MI/{model_name}/feature/MI_HF_coxreg_survival.png', dpi=300, bbox_inches='tight')
        # print("predicted  ",cph.predict_partial_hazard(X))
        plt.figure(figsize=(10,10), dpi=500)
        plot_lifetimes(
            df["time"],
            event_observed=df["status"],
            event_observed_color="#383838",
            event_censored_color="#383838",
            left_truncated=True,
        )
        plt.ylabel("Patient Number")
        plt.xlabel("weeks from MI diagnosed")
        plt.title("MI-> HF survival curve")
        plt.savefig(f'{save_dir}/MI_HF_coxreg_lifetimes.png', dpi=500, bbox_inches='tight')
        plt.close()

        ## find the feature with the largest coefficient
        id = cph.summary["coef"].sort_values(ascending=False).head(1).index.values[0]
        print(id)
        cph.plot_partial_effects_on_outcome(covariates=id, values=[-3,-1.5,-1, -0.5, -0.25,0, 0.25,0.5, 1,1.5,3], cmap='coolwarm')
        ## save the figure
        plt.savefig(f'{save_dir}/partial_effect_on_{id}.png', dpi=300, bbox_inches='tight')    
        plt.close()

        ## save the model
        with open(f'{save_dir}/MI_HF_coxreg_model.pkl', 'wb') as f:
            pickle.dump(cph, f)
        if if_bootstrap:
            ## perform bootstrap for B times
            C_opt,  Cb_opts, preds_bootfull, inds_inbag = bootstrap_survival_model(x_full=MI_HF_coxreg_df_feature_only, y_full=MI_HF_coxreg_df_label, 
                                                          B=10, args=args)    # ## print C_app-C_opt
            
            C_opt_95confint = np.percentile([C_app - o for o in Cb_opts], q=[2.5, 97.5])
            c_optimied = C_app-C_opt 
            print(f"C_app, Copt, Cadj, 95% CI: {C_app}, {C_opt}, {str(c_optimied)}, {str(C_opt_95confint)}")
            ## save those values, in together with largest coef
            with open(f'{save_dir}/MI_HF_coxreg_C_app_C_opt_C_adj.txt', 'w') as f:
                f.write(f"C_app, Copt, Cadj, 95% CI: {str(C_app)}, {str(C_opt)}, {str(c_optimied)}, {str(C_opt_95confint)}")
                f.write(f"largest coef: {id}")
            ## save the bootstrap results
            with open(f'{save_dir}/MI_HF_coxreg_bootstrap_results.pkl', 'wb') as f:
                pickle.dump(Cb_opts, f)
            return C_app, C_opt, c_optimied, C_opt_95confint
        else:
            print (f"C_app: {C_app}")
            with open(f'{save_dir}/MI_HF_coxreg_C_app_C_opt_C_adj.txt', 'w') as f:
                f.write(f"C_app, Copt, Cadj: {C_app}, -, -")
                f.write(f"largest coef: {id}")
            return C_app, '-', '-',['_',"-"]
    
        ## load the MI features
    model_name ="ECG_attention_pretrained_on_recon_ECG2Text"
    # model_name = 'resnet1d101_512+benchmark_classifier_ms_resnet'
    model_name = 'resnet1d101_512+benchmark_classifier_ms_resnet_finetune_ECG2Text'
    model_name = 'ECG_attention_512_finetuned_no_attention_pool_no_linear_ms_resnet_ECG2Text'
    model_name ="resnet1d101_512_pretrained_recon+ECG2Text"
    feature_file_name = "test_last_hidden_feature"
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
        # "train_bottleneck_feature",
    ]
    result = []
    failed_list=[]
    for model_name in model_name_list:
        for feature_file_name in feature_file_name_list:
            try:    
                C_app, C_opt, C_app_opp,C_opt_95confint= run_model(model_name, feature_file_name,if_bootstrap=args.if_bootstrap, seed=args.seed)
                print(f"{model_name}_{feature_file_name}: C_app, Copt, Cadj,C95: {str(C_app)}, {str(C_opt)}, {str(C_app_opp)} {str(C_opt_95confint)}")
                result.append([f'{model_name}_{feature_file_name}',C_app, C_opt, C_app_opp, C_opt_95confint])
                print(result)
            except Exception as e:
                print(e)
                ## 
                failed_list.append([model_name, feature_file_name])
    ## save the result as csv
    if not args.search_hyperparam:
        pd.DataFrame(result, columns=['model_name', 'C_app', 'C_opt', 'C_app_opp', 'C95']).to_csv(f'/home/engs2522/project/multi-modal-heart/multi_modal_heart/tasks/finetune_on_MI/MI_HF_coxreg_result_penalty={str(np.round(args.penalty,4))}.csv', index=False)
    else:
        pd.DataFrame(result, columns=['model_name', 'C_app', 'C_opt', 'C_app_opp', 'C95']).to_csv(f'/home/engs2522/project/multi-modal-heart/multi_modal_heart/tasks/finetune_on_MI/MI_HF_coxreg_result.csv', index=False)
    pd.DataFrame(failed_list, columns=['model_name', 'feature_file_name']).to_csv(f'/home/engs2522/project/multi-modal-heart/multi_modal_heart/tasks/finetune_on_MI/MI_HF_coxreg_failed_list.csv', index=False)





