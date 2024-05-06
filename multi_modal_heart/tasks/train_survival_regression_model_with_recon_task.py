
# %% code is partly adapted from survival4D project in github: https://github.com/UK-Digital-Heart-Project/4Dsurvival
## this script is used to train the survival regression model with the reconstruction task (with dropout) as originally proposed in the paper
## 1. load the model
## 2. train the survival model with the reconstruction task
## 3. test the model
## 4. save the model
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import os
import  random
import sys
import numpy as np
from tqdm import tqdm
from scipy.stats import zscore
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from lifelines.utils import concordance_index
import optunity
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split,StratifiedShuffleSplit
import operator as op
import itertools as it
from lightning.pytorch.tuner import Tuner
from torch import Tensor

from pytorch_metric_learning import samplers## tenrin to torch tensor dataset
sys.path.append("./")
from multi_modal_heart.model.ecg_net_attention import ECGEncoder,ECGAttentionAE
from multi_modal_heart.model.ecg_net import ECGAE
from multi_modal_heart.model.ecg_net import BenchmarkClassifier
from lightning.pytorch.callbacks import LearningRateFinder
from multi_modal_heart.common.scheduler import CosineWarmupScheduler,get_cosine_schedule_with_warmup
from multi_modal_heart.ECG.ecg_utils import batch_lead_mask
from multi_modal_heart.model.marcel_ECG_network import ECGMarcelVAE

## set os environment CUBLAS_WORKSPACE_CONFIG=:4096:8
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

## define survival loss function here
def _negative_log_likelihood(E, risk): 
    '''
    E: 1 if uncensored, 0 if censored
    risk: predicted risk

    ''' 
    hazard_ratio = torch.exp(risk)
    log_risk = torch.log(torch.cumsum(hazard_ratio,dim=0))
    uncensored_likelihood = risk - log_risk
    censored_likelihood = uncensored_likelihood * E
    neg_likelihood = -torch.sum(censored_likelihood)
    return neg_likelihood

def cox_loss(theta: Tensor, delta: Tensor, time: Tensor) -> Tensor:
    """
    Compute Cox proportional hazards loss.

    Args:
        theta (Tensor): Theta tensor, shape (batch_size, 1).
        delta (Tensor): Event indicator tensor, shape (batch_size, 1).
        time (Tensor): Time tensor, shape (batch_size, 1). Event time if uncensored,
                                censoring time if censored.

    Returns:
        Tensor: Cox loss.
    """
    if not is_valid_cox_batch(delta, time):
        return theta.sum() * 0

    time = time.reshape(-1)
    theta = theta.reshape(-1, 1)
    risk_mat = (time >= time[:, None]).float()
    loss_cox = (
        theta.reshape(-1) - logsumexp(theta.T, mask=risk_mat, dim=1)
    ) * delta.reshape(-1)
    loss_cox = loss_cox.sum() / delta.sum()
    return -loss_cox

def is_valid_cox_batch(delta: Tensor, time: Tensor) -> bool:
    """
    Check if the batch is valid for computing Cox loss.
    In the cases below, loss is not defined
    1. If there is no uncensored data
    2. If there are uncensored samples but the risk matrix is empty.
      I.e., no censored patients survived more that the event time of any
        uncensored patient.

    Args:
        delta (Tensor): Event indicator tensor, shape (batch_size, 1).
        time (Tensor): Survival time tensor, shape (batch_size, 1).

    Returns:
        bool: True if valid, False otherwise.
    """
    risk_matrix = (time >= time[:, None]).float()
    return ((risk_matrix.sum(dim=1) * delta.reshape(-1)) > 1).any()

def logsumexp(input_tensor: Tensor, mask: Tensor = None, dim: int = None, keepdim: bool = False) -> Tensor:
    """
    Compute the log of the sum of exponentials of input elements (masked).

    Args:
        input_tensor (Tensor): Input tensor.
        mask (Tensor, optional): Mask tensor, same shape as x.
        dim (int, optional): Dimension to reduce.
        keepdim (bool, optional): Keep dimension.

    Returns:
        Tensor: Result tensor.
    """
    if dim is None:
        input_tensor, dim = input_tensor.view(-1), 0
    max_value, _ = torch.max(input_tensor, dim=dim, keepdim=True)
    input_tensor = input_tensor - max_value
    res = torch.exp(input_tensor)
    if mask is not None:
        res = res * mask
    res = torch.log(torch.sum(res, dim=dim, keepdim=keepdim) + 1e-8)
    return res + max_value.squeeze(dim) if not keepdim else res + max_value

class FineTuneLearningRateFinder(pl.callbacks.LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)

class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides same representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True,random_state=None):
        if torch.is_tensor(y):
            y = y.cpu().numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        assert n_batches >= 1, 'number of batches must be at least 2'
        if n_batches>=2:
            self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        else: self.skf = None
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.random_state = random_state

    def __iter__(self):
        if self.shuffle and self.skf is not None:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        if self.skf is not None:
            for train_idx, test_idx in self.skf.split(self.X, self.y):
                yield test_idx
        else:
            a_list = torch.arange(len(self.y)).tolist()
            ## shuffle the list
            if self.shuffle:
                random.shuffle(a_list)
            yield a_list

    def __len__(self):
        return len(self.y) // self.batch_size

class LitSurvivalModel(pl.LightningModule):
    def __init__(self,encoder,input_dim,lr=1e-4,alpha = 0.5,
                 wd=1e-2,dropout=0.5,freeze_encoder=False, decoder=None,
                 max_iters=20000, warm_up=False, add_VAE_loss=False):
        super().__init__()
    
        self.save_hyperparameters(ignore=["encoder", "decoder","freeze_encoder"])
        self.encoder = encoder
        self.decoder = decoder
        self.dropout_input = nn.Dropout(p=dropout)
        self.freeze_encoder = freeze_encoder
        self.lr_scheduler = None
        self.warm_up=warm_up
        self.add_VAE_loss = add_VAE_loss
        # self.args = args
        if self.freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        #### add a branch for survival prediction regression
        self.downsteam_net = BenchmarkClassifier(input_dim,hidden_size=3,output_size=1,batchwise_dropout=True)

        self.test_score_list = []
        self.y_status_list = []
        self.y_duration_list = []
        self.c_index = 0

        ## validation loss
        self.validation_score_list= []
        self.y_status_val_list = []
        self.y_duration_val_list = []
        self.c_index_val = 0

    def forward(self, x, mask=None):
        latent_code = self.encoder(x,mask)
        # print (latent_code.shape)
        self.latent_code  = latent_code
        log_risk_score = self.downsteam_net(latent_code)

        if self.decoder is not None:
            recon_signal = self.decoder(latent_code)
            return [log_risk_score,recon_signal]
        return [log_risk_score]

    def run_batch(self, batch, batch_idx, stage="train"):
         ## sort the batch by the survival duration of the event
        x = batch[0] ## input signal
        y, duration, eid_list = batch[1][:,0],batch[1][:,1],batch[1][:,2]    
        y = y.float()
        duration = duration.float()
       
        if stage!="train":
            self.training =False
        else:
            self.training= True
            # print("status, duration",y[0],duration[0])
            # print (f"number of uncensoring status:{torch.sum(y).item()}")
            # print (f"number of censoring status:{torch.sum(y==0.).item()}")

            x, mask = batch_lead_mask(x,same_mask_per_batch=True)
        ## sort the batch by the survival duration of the event
        # duration, sort_idx = duration.sort(descending=True)
        # y = y[sort_idx]
        # x = x[sort_idx]
        if self.decoder is not None:
            y_hat,recon_signal = self(x,mask=None)
            ## survival loss
            # surv_loss = _negative_log_likelihood(y,y_hat)
            surv_loss = cox_loss(y_hat,y,duration)
            self.log(f"{stage}/survival_loss", surv_loss,prog_bar=True)

            ## mse loss 
            mse_loss = F.mse_loss(recon_signal, x)
            self.log(f"{stage}/mse_loss", mse_loss,prog_bar=True)
            loss = self.hparams.alpha*surv_loss + (1-self.hparams.alpha)*mse_loss

        else:
            y_hat = self(x,mask=None)[0]
            ## survival loss
            surv_loss = cox_loss(y_hat,y,duration)
            loss = surv_loss
            self.log(f"{stage}/survival_loss", surv_loss,prog_bar=True)
        ## in case VAE is presented
        if  self.add_VAE_loss:
            beta = 1e-4
            try:
                log_var = self.encoder.z_log_var
                mu = self.encoder.z_mu
                kld_loss = beta*torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                loss = loss + kld_loss
                self.log(f"{stage}/VAE_loss", kld_loss,prog_bar=True)
            except:
                Warning('VAE loss is not applied')
        
        if "test" in stage:
            # print(stage)
            # print("save risk score")
            self.test_score_list.append(y_hat)
            self.y_status_list.append(y)
            self.y_duration_list.append(duration)
        if "val" in stage:
            self.validation_score_list.append(y_hat)
            self.y_status_val_list.append(y)
            self.y_duration_val_list.append(duration)

            ## save risk score
        self.log(f"{stage}_loss",loss,prog_bar=True)
        return loss

    
    def training_step(self, batch, batch_idx):
        loss = self.run_batch(batch,batch_idx,stage="train")
        return loss
    def validation_step(self, batch, batch_idx):
        loss = self.run_batch(batch,batch_idx,stage="val")
        return loss
       
    def test_step(self, batch, batch_idx):
        loss = self.run_batch(batch,batch_idx,stage="test")
        return loss
    def on_test_epoch_end(self):
        if (not len(self.test_score_list)==0) and (not len(self.y_status_list)==0) and (not len(self.y_duration_list)==0):
            pred_score = torch.cat(self.test_score_list,dim=0)
            y_status = torch.cat(self.y_status_list,dim=0)
            y_duration = torch.cat(self.y_duration_list,dim=0)
            ## calculate the concordance index
            c_index = concordance_index(y_duration.cpu().numpy(), -pred_score.detach().cpu().numpy(), y_status.cpu().numpy())
            self.log("c_index",c_index)
            self.c_index = c_index
    def on_test_epoch_start(self):
        self.reset()
    def on_validation_epoch_end(self):
        if (not len(self.validation_score_list)==0) and (not len(self.y_status_val_list)==0) and (not len(self.y_duration_val_list)==0):
            pred_score = torch.cat(self.validation_score_list,dim=0)
            y_status = torch.cat(self.y_status_val_list,dim=0)
            y_duration = torch.cat(self.y_duration_val_list,dim=0)
            ## calculate the concordance index
            c_index = concordance_index(y_duration.cpu().numpy(), -pred_score.detach().cpu().numpy(), y_status.cpu().numpy())
            self.log("val_c_index",c_index)
            self.c_index_val = c_index
    def on_validation_epoch_start(self):
        self.reset_val()
            
    def reset(self):
        self.test_score_list = []
        self.y_status_list = []
        self.y_duration_list = []
        # self.c_index = 0
    def reset_val(self):
        ## validation loss
        self.validation_score_list= []
        self.y_status_val_list = []
        self.y_duration_val_list = []
    def configure_optimizers(self):
        optimizer= torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        if self.warm_up:
            self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps= 0.2*self.hparams.max_iters, num_training_steps= self.hparams.max_iters)
        else: self.lr_scheduler = None
        return optimizer
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.lr_scheduler is not None: self.lr_scheduler.step()  # Step per iteration

def DL_single_run(x_train,y_train, model_name,latent_code_dim,batch_size,add_VAE_loss=False,logger=None,train_from_scratch=False,freeze_encoder=True, lr=1e-3,alpha =0.5,wd=1e-5,dropout=0.5,max_epochs=50,enable_checkpointing=True,disable_logging=False,warm_up=False,random_seed=42, test_only=False, test_checkpoint_path=None):
    ## split it into train and validation
    x_inner_train, x_val, y_innner_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=random_seed,stratify=y_train[:,0])
    train_dataset = TensorDataset(torch.from_numpy(x_inner_train).float(),torch.from_numpy(np.array(y_innner_train)).float())
    val_dataset = TensorDataset(torch.from_numpy(x_val).float(),torch.from_numpy(np.array(y_val)).float())
    g = torch.Generator()
    g.manual_seed(random_seed)
    ##batch sampler
    batch_sampler = StratifiedBatchSampler(y_innner_train[:,0], batch_size=batch_size, shuffle=True,random_state=random_seed)
    train_dataloader = DataLoader(train_dataset,
                                batch_sampler = batch_sampler,
                                # batch_size = batch_size,
                                num_workers=4,
                                worker_init_fn=seed_worker,
                                # shuffle=True,
                                # drop_last=True,
                                generator=g,
                                )
    val_dataloader = DataLoader(val_dataset,
                                batch_size = batch_size,
                                num_workers=4,
                                worker_init_fn=seed_worker,
                                generator=g,
                                shuffle=False,
                                drop_last=False)

    encoder, decoder, max_epochs = get_model(model_name, train_from_scratch=train_from_scratch, freeze_encoder=freeze_encoder,time_steps=x_train.shape[-1],latent_code_dim=latent_code_dim)
    max_iters=max_epochs * len(train_dataloader)
    survival_model = LitSurvivalModel(encoder=encoder,
                                      input_dim=latent_code_dim,
                                      decoder = decoder,
                                      lr=lr,
                                      alpha = alpha,
                                      wd=wd,
                                      dropout = dropout,
                                      freeze_encoder=freeze_encoder,
                                      max_iters=max_iters,
                                      warm_up =warm_up,add_VAE_loss=add_VAE_loss)
                                     
    if not test_only:
        checkpoint_dir  = os.path.join(logger.log_dir,"checkpoints")

        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, 
                                                        filename='{epoch:02d}-{val_c_index:.2f}',
                                                        save_top_k=1, monitor="val_c_index", mode='max',save_last=True)


        callbacks=[
            FineTuneLearningRateFinder(milestones=[],min_lr=1e-7, max_lr=1e-2, 
                                        mode='exponential', early_stop_threshold=None),
            checkpoint_callback,
        ]
        trainer = pl.Trainer(accelerator="gpu",
                            gradient_clip_val=100,
                            devices=1, max_epochs=max_epochs,
                            default_root_dir=logger.log_dir if logger is not None else os.getcwd(),
                            enable_checkpointing=enable_checkpointing,
                            logger = logger,callbacks=callbacks,
                            )

        trainer.fit(survival_model,train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)
    else:
        survival_model.load_state_dict(torch.load(test_checkpoint_path))
        trainer = pl.Trainer(accelerator="gpu",
                            gradient_clip_val=100,
                            devices=1, max_epochs=max_epochs,
                            default_root_dir=logger.log_dir if logger is not None else os.getcwd(),
                            enable_checkpointing=False,
                            logger = logger
                            )
        survival_model.eval()
    return trainer,survival_model

# def hypersearch_DL(x_data, y_data, model_name,method, nfolds, nevals, lrexp_range, wdexp_range, dro, 
#                    alpha,batch_size=100,freeze_encoder=True,args=None,logger=None):
#     labels = y_data[:,0]
#     strata_labels = [list(zip(*g))[0]for _, g in it.groupby(enumerate(labels), op.itemgetter(1))]
#     @optunity.cross_validated(x=x_data, y=y_data, num_folds=nfolds, num_iter=2, strata=strata_labels)
#     def modelrun(x_train, y_train, x_test, y_test, wd):
#                 ## tenrin to torch tensor dataset
#         train_dataset = TensorDataset(torch.from_numpy(x_train).float(),torch.from_numpy(y_train).float())
#         test_dataset = TensorDataset(torch.from_numpy(x_test).float(),torch.from_numpy(y_test).float())
#         train_dataloader = DataLoader(train_dataset,
#                                       batch_size = args.batch_size,
#                                       num_workers=4,
#                                       shuffle=True,
#                                       drop_last=True)
#         test_data_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False, drop_last=False) # create your dataloader
#         trainer,survival_model = DL_single_run(
#                                 train_dataloader,
#                                 model_name,
#                                freeze_encoder=freeze_encoder,logger=logger,
#                                alpha = alpha,wd=wd,dropout=dro,
#                                disable_logging=True,enable_checkpointing=False,args=args)
        
#         survival_model.eval()
#         trainer.test(survival_model,test_data_loader)
#         ## get the c-index
#         c_index = survival_model.c_index
#         print ("c-index",c_index)
#         del survival_model
#         del trainer
#         return c_index
#     # box_constraints = {'lrexp': [-6, -3],'wdexp': [-6,0]}
#     s = optunity.solvers.GridSearch(wd=[1e-2,1e-1,0,1,2])


#     optimal_pars, searchlog= s.maximize(modelrun)
    
#     print('Optimal hyperparameters : ' + str(optimal_pars))
#     # print('Cross-validated C after tuning: %1.3f' % searchlog.optimum)
#     return optimal_pars, searchlog 
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset(dataset_name, full_wave=False, unit="month"):
    if dataset_name=="MI_with_HF_event":
        if full_wave:
            x = np.load("/Data/engs2522/ECG_data/MI_to_HF_survival_data/ecg_data_full_wave.npy")
            time_steps = 1024
        else: 
            x = np.load("/Data/engs2522/ECG_data/MI_to_HF_survival_data/ecg_data.npy")
            time_steps = 608
        y = np.load("/Data/engs2522/ECG_data/MI_to_HF_survival_data/y_status_duration.npy")
        # ## read the data from the numpy array
        # stacked_ecg_data_list = np.load("/home/engs2522/project/multi-modal-heart/multi_modal_heart/tasks/survival_regression/data/batched_ecg_median_wave_MI_with_HF_event.npy")
        # eid_list = np.load("/home/engs2522/project/multi-modal-heart/multi_modal_heart/tasks/survival_regression/data/e_id_list_MI_with_HF_event.npy")
        # eid_status_duration_df = pd.read_csv("/home/engs2522/project/multi-modal-heart/multi_modal_heart/toolkits/ukb/non_imaging_information/MI/MI_HF_coxreg_df.csv")
        # ## sort the data by the eid
        # status_list = []
        # duration_list = []
        # for eid in eid_list:
        #     print(eid)
        #     status_list.append(int(eid_status_duration_df[eid_status_duration_df["eid"]==eid]["HF_status"].values[0]))
        #     duration_list.append(int(eid_status_duration_df[eid_status_duration_df["eid"]==eid]["time_to_HF"].values[0]))
        # ## ECG data preprocessing
        # assert len(stacked_ecg_data_list)==len(status_list)==len(duration_list)==len(eid_list)
        #    ## concatentate the status and duration and eid to the y
        # y = np.stack([status_list,duration_list,eid_list],axis=-1)

    elif dataset_name=="HYP_with_HF_event":
        if full_wave:
            x = np.load("/Data/engs2522/ECG_data/HYP_to_HF_survival_data/ecg_data_full_wave.npy")
            time_steps = 1024
        else: 
            x = np.load("/Data/engs2522/ECG_data/HYP_to_HF_survival_data/ecg_data.npy")
            time_steps = 608
        y = np.load("/Data/engs2522/ECG_data/HYP_to_HF_survival_data/y_status_duration.npy")
        ## for duration, change it months
       
    else:
        raise NotImplementedError
    assert x.shape[0]==y.shape[0], "the number of samples should be the same, but got {} and {}".format(x.shape[0],y.shape[0])
    ## normalize the data
    x = zscore(x,axis=-1)
    x = np.nan_to_num(x)
    pad_num = (time_steps-x.shape[-1])//2
    x = np.pad(x,((0,0),(0,0),(pad_num,pad_num)),"constant",constant_values=0)
    if unit == "month":
        y[:,1] = y[:,1]/30.0
    elif unit == "year":
        y[:,1] = y[:,1]/365.0
    elif unit == "day":
        pass
    else: 
        raise NotImplementedError
    y[:,1] = y[:,1]+1 ##avoid the duration to be 0
    print("input ecg shape",x.shape)
    print("status, duration, eid", y.shape)
    return x,y

def get_model(model_name, train_from_scratch=False, freeze_encoder=False,time_steps=608, latent_code_dim=512):
        if model_name=="resnet1d101_pretrained_recon" or model_name=="resnet1d101":
            ecg_net = ECGAE(encoder_type="resnet1d101",in_channels=12,ECG_length=time_steps,decoder_type="attention_decoder",
                embedding_dim=latent_code_dim//2,latent_code_dim=latent_code_dim,
                add_time=False,
                encoder_mha = False,
                apply_method="",
                decoder_outdim=12,apply_batchwise_dropout=True)
            checkpoint_path = f"/home/engs2522/project/multi-modal-heart/log_median/resnet1d101_{latent_code_dim}+benchmark_classifier_attention_decoder/checkpoints/checkpoint_best_loss.ckpt"

        elif model_name =="resnet1d101_pretrained_recon_classification":
            ecg_net = ECGAE(encoder_type="resnet1d101",in_channels=12,ECG_length=time_steps,decoder_type="attention_decoder",
                    embedding_dim=latent_code_dim//2,latent_code_dim=latent_code_dim,
                    add_time=False,
                    encoder_mha = False,
                    apply_method="",
                    decoder_outdim=12,apply_batchwise_dropout=True)
            checkpoint_path =f"/home/engs2522/project/multi-modal-heart/log_median/resnet1d101_{latent_code_dim}+benchmark_classifier_attention_decoder_recon_classification/checkpoints/checkpoint_best_loss.ckpt"
   

        elif model_name==f"resnet1d101_pretrained_recon+ECG2Text" or  model_name==f"resnet1d101_pretrained_recon+ECG2RawText":
            # ecg_net= ECGAE(encoder_type="resnet1d101",in_channels=12,ECG_length=time_steps,decoder_type="ms_resnet",
            #                 embedding_dim=256,latent_code_dim=512,
            #                 add_time=False,
            #                 encoder_mha = False,
            #                 apply_method="",
            #                 decoder_outdim=12)
            # checkpoint_path = "./log_median/resnet1d101_512+benchmark_classifier_ms_resnet_ECG2Text/checkpoints/checkpoint_best_loss-v2.ckpt"
            if model_name==f"resnet1d101_pretrained_recon+ECG2Text":
                ecg_net = ECGAE(encoder_type="resnet1d101",in_channels=12,ECG_length=time_steps,decoder_type="attention_decoder",
                    embedding_dim=latent_code_dim//2,latent_code_dim=latent_code_dim,
                    add_time=False,
                    encoder_mha = False,
                    apply_method="",
                    decoder_outdim=12,apply_batchwise_dropout=True)
                checkpoint_path = f"/home/engs2522/project/multi-modal-heart/log_median/resnet1d101_{latent_code_dim}+benchmark_classifier_attention_decoder_ECG2Text/checkpoints/checkpoint_best_loss.ckpt"
            else: 
                ecg_net= ECGAE(encoder_type="resnet1d101",in_channels=12,ECG_length=time_steps,decoder_type="attention_decoder",
                            embedding_dim=latent_code_dim//2,latent_code_dim=latent_code_dim,
                            add_time=False,
                            encoder_mha = False,
                            apply_method="",
                            decoder_outdim=12,apply_batchwise_dropout=True)
                checkpoint_path = f"/home/engs2522/project/multi-modal-heart/log_median/resnet1d101_{latent_code_dim}+benchmark_classifier_attention_decoder_ECG2RawText/checkpoints/checkpoint_best_loss.ckpt"

        elif model_name==f"resnet1d101_pretrained_classification":
            ecg_net= ECGAE(encoder_type="resnet1d101",in_channels=12,ECG_length=time_steps,decoder_type="ms_resnet",
                            embedding_dim=latent_code_dim//2,latent_code_dim=latent_code_dim,
                            add_time=False,
                            encoder_mha = False,
                            apply_method="",
                            decoder_outdim=12,apply_batchwise_dropout=True)
            # checkpoint_path = "/home/egs2522/project/multi-modal-heart/log_median_finetune/resnet1d101_512+benchmark_classifier_ms_resnet/checkpoints/checkpoint_best_loss.ckpt"
            checkpoint_path  =f"/home/engs2522/project/multi-modal-heart/log_median_finetune/resnet1d101_{latent_code_dim}+benchmark_classifier_raw_ms_resnet/checkpoints/checkpoint_best_val_macro_auc.ckpt"

        elif model_name==f"resnet1d101_pretrained_classification+ECG2Text":
            
            # checkpoint_path = "/home/engs2522/project/multi-modal-heart/log_median_finetune/resnet1d101_512+benchmark_classifier_ms_resnet_ECG2Text/checkpoints/checkpoint_best_loss.ckpt"
            checkpoint_path  =f"/home/engs2522/project/multi-modal-heart/log_median_finetune/resnet1d101_{latent_code_dim}+benchmark_classifier_raw_ms_resnet_ECG2Text/checkpoints/checkpoint_best_val_macro_auc-v1.ckpt"
            ecg_net= ECGAE(encoder_type="resnet1d101",in_channels=12,ECG_length=time_steps,decoder_type="ms_resnet",
                        embedding_dim=latent_code_dim//2,latent_code_dim=latent_code_dim,
                        add_time=False,
                        encoder_mha = False,
                        apply_method="",
                        decoder_outdim=12,apply_batchwise_dropout=True)
        
        elif model_name==f"ECG_attention_pretrained_on_classification":
            ecg_net  = ECGAttentionAE(num_leads=12, time_steps=time_steps, z_dims=latent_code_dim, linear_out=latent_code_dim, downsample_factor=5, base_feature_dim=4,if_VAE=False,use_attention_pool=False,
                                no_linear_in_E=True, apply_lead_mask=False,apply_batchwise_dropout=True)
            checkpoint_path = f"./log_median_finetune/ECG_attention_{latent_code_dim}_raw_no_attention_pool_no_linear_ms_resnet/checkpoints/checkpoint_best_val_macro_auc.ckpt"
        
        elif model_name==f"ECG_attention_pretrained_on_classification+ECG2Text":
            ecg_net  = ECGAttentionAE(num_leads=12, time_steps=time_steps, z_dims=latent_code_dim, linear_out=latent_code_dim, downsample_factor=5, base_feature_dim=4,if_VAE=False,use_attention_pool=False,
                                no_linear_in_E=True, apply_lead_mask=False,apply_batchwise_dropout=True)
            checkpoint_path = f"./log_median_finetune/ECG_attention_{latent_code_dim}_raw_no_attention_pool_no_linear_ms_resnet_ECG2Text/checkpoints/checkpoint_best_val_macro_auc-v1.ckpt"

        elif model_name==f"ECG_attention_pretrained_on_recon" or model_name ==f"ECG_attention":
            ecg_net  = ECGAttentionAE(num_leads=12, time_steps=time_steps, z_dims=latent_code_dim, linear_out=latent_code_dim, 
                                    downsample_factor=5, base_feature_dim=4,if_VAE=False,
                                    use_attention_pool=False,
                                no_linear_in_E=True, apply_lead_mask=False, no_lead_attention=False,no_time_attention=False,apply_batchwise_dropout=True)
            if latent_code_dim==64:
                checkpoint_path =f"./log_median/ECG_attention_{latent_code_dim}_raw_no_attention_pool_no_linear_ms_resnet/checkpoints/checkpoint_best_loss.ckpt"
            else:
                checkpoint_path =f"./log_median/ECG_attention_{latent_code_dim}_finetuned_no_attention_pool_no_linear_ms_resnet/checkpoints/checkpoint_best_loss-v2.ckpt"

        elif model_name ==f"ECG_attention_downsample_3":
            ecg_net  = ECGAttentionAE(num_leads=12, time_steps=time_steps, z_dims=latent_code_dim, linear_out=latent_code_dim, 
                                    downsample_factor=3, base_feature_dim=4,if_VAE=False,
                                    use_attention_pool=False,
                                    upsample_factor=3,
                                    no_linear_in_E=True, apply_lead_mask=False, no_lead_attention=False,no_time_attention=False,apply_batchwise_dropout=True)
            checkpoint_path=""
        elif model_name ==f"ECG_attention_no_lead_attention"  or  model_name ==f"ECG_attention_no_lead_attention_pretrained_on_recon_ECG2Text":
            ecg_net  = ECGAttentionAE(num_leads=12, time_steps=time_steps, z_dims=latent_code_dim, linear_out=latent_code_dim, 
                                    downsample_factor=5, base_feature_dim=4,if_VAE=False,
                                    use_attention_pool=False,
                                no_linear_in_E=True, apply_lead_mask=False, 
                                no_lead_attention=True,no_time_attention=False,apply_batchwise_dropout=True)
            checkpoint_path=""
            if model_name=="ECG_attention_no_lead_attention_pretrained_on_recon_ECG2Text":
               checkpoint_path="/home/engs2522/project/multi-modal-heart/log_median/ECG_attention_512_raw_no_attention_pool_no_linear_abl_no_lead_attention_ms_resnet_ECG2Text/checkpoints/checkpoint_best_loss.ckpt"
        elif model_name ==f"ECG_attention_no_time_attention" or  model_name ==f"ECG_attention_no_time_attention_pretrained_on_recon_ECG2Text":
            ecg_net  = ECGAttentionAE(num_leads=12, time_steps=time_steps, z_dims=latent_code_dim, linear_out=latent_code_dim, 
                                    downsample_factor=5, base_feature_dim=4,if_VAE=False,
                                    use_attention_pool=False,
                                no_linear_in_E=True, apply_lead_mask=False, 
                                no_lead_attention=False,no_time_attention=True,apply_batchwise_dropout=True)
            checkpoint_path=""
            if model_name=="ECG_attention_no_time_attention_pretrained_on_recon_ECG2Text":
               checkpoint_path="/home/engs2522/project/multi-modal-heart/log_median/ECG_attention_512_raw_no_attention_pool_no_linear_abl_no_time_attention_ms_resnet_ECG2Text/checkpoints/checkpoint_best_loss.ckpt"
        
        elif model_name ==f"ECG_attention_no_lead_time_attention" or  model_name ==f"ECG_attention_no_lead_time_attention_pretrained_on_recon_ECG2Text":
            ecg_net  = ECGAttentionAE(num_leads=12, time_steps=time_steps, z_dims=latent_code_dim, linear_out=latent_code_dim, 
                                    downsample_factor=5, base_feature_dim=4,if_VAE=False,
                                    use_attention_pool=False,
                                no_linear_in_E=True, apply_lead_mask=False, 
                                no_lead_attention=True,no_time_attention=True,apply_batchwise_dropout=True)
            checkpoint_path=""
            if model_name=="ECG_attention_no_lead_time_attention_pretrained_on_recon_ECG2Text":
               checkpoint_path="/home/engs2522/project/multi-modal-heart/log_median/ECG_attention_512_raw_no_attention_pool_no_linear_abl_no_lead_time_attention_ms_resnet_ECG2Text/checkpoints/checkpoint_best_loss.ckpt"
        elif model_name==f"ECG_attention_pretrained_on_recon_ECG2Text" or model_name=="ECG_attention_pretrained_on_recon_ECG2RawText":
            ecg_net  = ECGAttentionAE(num_leads=12, time_steps=time_steps, z_dims=latent_code_dim, linear_out=latent_code_dim, 
                                    downsample_factor=5, base_feature_dim=4,if_VAE=False,
                                    use_attention_pool=False,
                                no_linear_in_E=True, apply_lead_mask=False, no_lead_attention=False,no_time_attention=False,apply_batchwise_dropout=True)
            if  "ECG_attention_pretrained_on_recon_ECG2RawText"==model_name:
                if latent_code_dim==64:
                    checkpoint_path = f"./log_median/ECG_attention_{latent_code_dim}_raw_no_attention_pool_no_linear_ms_resnet_ECG2Text/checkpoints/checkpoint_best_loss.ckpt"
                else:
                    checkpoint_path = f"./log_median/ECG_attention_{latent_code_dim}_finetuned_no_attention_pool_no_linear_ms_resnet_ECG2Text/checkpoints/checkpoint_best_loss-v5.ckpt"
        
            else: 
                if latent_code_dim==64:
                    checkpoint_path =f"./log_median/ECG_attention_{latent_code_dim}_raw_no_attention_pool_no_linear_ms_resnet_ECG2Text/checkpoints/checkpoint_best_loss.ckpt"
                else:
                    checkpoint_path =f"./log_median/ECG_attention_{latent_code_dim}_finetuned_no_attention_pool_no_linear_ms_resnet_ECG2Text/checkpoints/checkpoint_best_loss-v2.ckpt"
        
        elif model_name==f"ECG_attention_pretrained_on_recon_ECG2Text_fixed_text_encoder" or model_name=="ECG_attention_pretrained_on_recon_ECG2RawText_fixed_text_encoder":
            
            ecg_net  = ECGAttentionAE(num_leads=12, time_steps=time_steps, z_dims=latent_code_dim, linear_out=latent_code_dim, 
                                    downsample_factor=5, base_feature_dim=4,if_VAE=False,
                                    use_attention_pool=False,
                                no_linear_in_E=True, apply_lead_mask=False, no_lead_attention=False,no_time_attention=False,apply_batchwise_dropout=True)
            if model_name =="ECG_attention_pretrained_on_recon_ECG2Text_fixed_text_encoder":
                checkpoint_path = f"/home/engs2522/project/multi-modal-heart/log_median/ECG_attention_{latent_code_dim}_raw_no_attention_pool_no_linear_ms_resnet_ECG2Text_fixed_text_encoder/checkpoints/checkpoint_best_loss.ckpt"
            elif model_name =="ECG_attention_pretrained_on_recon_ECG2RawText_fixed_text_encoder":
                checkpoint_path = f"/home/engs2522/project/multi-modal-heart/log_median/ECG_attention_{latent_code_dim}_raw_no_attention_pool_no_linear_ms_resnet_ECG2RawText_fixed_text_encoder/checkpoints/checkpoint_best_loss.ckpt"

        elif model_name==f"ECG_attention_pretrained_on_recon_classification":
            ecg_net = ECGAttentionAE(num_leads=12, time_steps=time_steps, z_dims=latent_code_dim, linear_out=latent_code_dim, 
                                 downsample_factor=5, base_feature_dim=4,if_VAE=False,
                                 use_attention_pool=False, no_lead_attention=False,
                                 no_linear_in_E=True,apply_batchwise_dropout=True,
                                 )
            if latent_code_dim==64:
                checkpoint_path = f"/home/engs2522/project/multi-modal-heart/log_median/ECG_attention_{latent_code_dim}_raw_no_attention_pool_no_linear_ms_resnet_recon_classification/checkpoints/checkpoint_best_loss.ckpt"
            else:
                checkpoint_path = f"/home/engs2522/project/multi-modal-heart/log_median/ECG_attention_{latent_code_dim}_raw_no_attention_pool_no_linear_attention_decoder_recon_classification/checkpoints/checkpoint_best_loss-v3.ckpt"
        elif model_name.startswith("Marcel"):
            ecg_net = ECGMarcelVAE(num_leads=12,time_steps=time_steps,z_dims=latent_code_dim) ## for ease of comparison
            if model_name =="Marcel_pretrained_recon":
                checkpoint_path = f"/home/engs2522/project/multi-modal-heart/log_median/marcelVAE_{latent_code_dim}_attention_decoder/checkpoints/checkpoint_best_loss.ckpt"
            elif model_name=="Marcel_pretrained_on_recon+ECG2RawText":
                checkpoint_path = f"/home/engs2522/project/multi-modal-heart/log_median/marcelVAE_{latent_code_dim}_attention_decoder_ECG2RawText/checkpoints/checkpoint_best_loss.ckpt"
            elif model_name =="Marcel_pretrained_on_recon+ECG2Text":
                checkpoint_path = f"/home/engs2522/project/multi-modal-heart/log_median/marcelVAE_{latent_code_dim}_attention_decoder_ECG2Text/checkpoints/checkpoint_best_loss.ckpt"
            elif model_name =="Marcel_pretrained_on_recon_classification":
                checkpoint_path = f"/home/engs2522/project/multi-modal-heart/log_median/marcelVAE_{latent_code_dim}_attention_decoder_recon_classification/checkpoints/checkpoint_best_loss.ckpt"
            else:
                checkpoint_path = ""
        elif model_name.startswith("AE"):
            ecg_net = ECGMarcelVAE(num_leads=12,time_steps=time_steps,z_dims=latent_code_dim) ## for ease of comparison
            if model_name =="AE_pretrained_recon_disable_VAE_loss":
                checkpoint_path = f"/home/engs2522/project/multi-modal-heart/log_median/marcelVAE_{latent_code_dim}_attention_decoder_disable_VAE_loss/checkpoints/checkpoint_best_loss.ckpt"
            elif model_name =="AE_pretrained_on_recon+ECG2Text_disable_VAE_loss":
                checkpoint_path = f"/home/engs2522/project/multi-modal-heart/log_median/marcelVAE_{latent_code_dim}_attention_decoder_ECG2Text_disable_VAE_loss/checkpoints/checkpoint_best_loss.ckpt"
            else:
                checkpoint_path = ""

   
        else:
            print(f"{model_name} failed to load")
            raise NotImplementedError
    
        num_classes = 2 ## for binary classification
        if train_from_scratch:
            assert freeze_encoder==False, "if train from scratch, the encoder should not be frozen"

        encoder = ecg_net.encoder
        decoder = ecg_net.decoder
        if not train_from_scratch:
            checkpoint = torch.load(checkpoint_path)["state_dict"]
            try:
                encoder_params = {(".").join(key.split(".")[2:]):value for key, value in checkpoint.items() if str(key).startswith("network.encoder")}
                encoder.load_state_dict(encoder_params)
                ## load the decoder
                decoder_params = {(".").join(key.split(".")[2:]):value for key, value in checkpoint.items() if str(key).startswith("network.decoder")}
                decoder.load_state_dict(decoder_params)
            except:
                encoder_params = {(".").join(key.split(".")[1:]):value for key, value in checkpoint.items() if str(key).startswith("encoder")}
                encoder.load_state_dict(encoder_params)
                ## load the decoder
                decoder_params = {(".").join(key.split(".")[1:]):value for key, value in checkpoint.items() if str(key).startswith("decoder")}
                decoder.load_state_dict(decoder_params)
            max_epochs = 100
        else:
            max_epochs = 400
        return encoder,decoder,max_epochs

if __name__ == "__main__":

    ## args set up
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",type=str,default="MI_with_HF_event")
    parser.add_argument("--full_wave", action="store_true",help="whether to use full wave data or not")
    parser.add_argument("--model_name",type=str,default="resnet1d101_512_pretrained_recon")
    parser.add_argument("--latent_code_dim",type=int,default=512)
    parser.add_argument("--train_from_scratch",action="store_true")
    parser.add_argument("--freeze_encoder",action="store_true")
    parser.add_argument("--warm_up",action="store_true",help="whether to use warm up scheduler")
    parser.add_argument("--add_VAE_loss",action="store_true",help="whether to use VAE loss")

    ## seed
    parser.add_argument("--search_hyperparam",action="store_true")
    ## hyperparameters for survival model training
    parser.add_argument("--lr",type=float,default=1e-5)
    parser.add_argument("--alpha",type=float,default=0.5)
    parser.add_argument("--wd",type=float,default=1e-2)
    parser.add_argument("--dropout",type=float,default=0.5)
    parser.add_argument("--batch_size",type=int,default=1024)
    parser.add_argument("--n_folds",type=int,default=5)
    parser.add_argument("--unit",type=str,choices=["month","year","day"],default="month")
    ## for test only
    parser.add_argument("--test_only",action="store_true")
    parser.add_argument("--test_checkpoint_path",type=str,default="")

    args = parser.parse_args()
    # print(args)
    ## get the data, x is the ecg data N*num_leads*time_steps y consists of the status and duration information, N*2
    x, y = get_dataset(args.dataset_name,full_wave=args.full_wave,unit=args.unit)
    # model_name = "resnet1d101_512_pretrained_recon+ECG2Text"
    #make dir
    model_name = args.model_name
    latent_code_dim  = args.latent_code_dim
    root_dir ="/Data/engs2522/ECGresult"
    if args.add_VAE_loss:
        model_dir = f"{root_dir}/train_survival_net_{args.dataset_name}_{str(args.alpha)}/{model_name}_{latent_code_dim}_w_VAE_loss"
    else:
        model_dir = f"{root_dir}/train_survival_net_{args.dataset_name}_{str(args.alpha)}/{model_name}_{latent_code_dim}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    def run_nested_cv(x,y, n_folds,random_seed,args):
        # pl.seed_everything(random_seed)
        ## shuffle the data (x,y) together
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        # Initialize lists to store predictions
        val_c_index_list = []
        y_status_list = y[:,0]
        i =0

        for cval, (train_indices, test_indices) in enumerate(kf.split(x, y_status_list)):
            x_train, y_train = x[train_indices], y[train_indices]
            x_test, y_test = x[test_indices], y[test_indices]
            print("Step 1a")
            cval_dir = f"{model_dir}/{random_seed}/cval_{str(cval)}"
            tb_logger = TensorBoardLogger(f"{root_dir}/train_survival_net_{args.dataset_name}_{str(args.alpha)}/{model_name}_{args.latent_code_dim}", name=f"{random_seed}/cval_{str(cval)}", version="") 

            if not os.path.exists(cval_dir):os.makedirs(cval_dir)
            # if args.search_hyperparam:
            #     opars, osummary = hypersearch_DL(x_data=x_train, y_data=y_train,model_name=model_name,
            #                                 method='random search', nfolds=n_folds, nevals=50,
            #                                 lrexp_range=[-6,-2], wdexp_range=[-7,-2], 
            #                                 alpha=args.alpha,
            #                                 dro =args.dropout,
            #                                 batch_size=args.batch_size,freeze_encoder=args.freeze_encoder,args=args,logger=tb_logger)
            
            
            #     alpha = args.alpha
            #     # lr = 10**opars["lrexp"]
            #     wd = opars["wd"]
            #     dro =args.dropout
            #     hyperparam_dict = {"alpha":alpha,"lr":lr,"wd":wd,"dro":dro}
            #     pd.DataFrame(hyperparam_dict,index=[0]).to_csv(f"{cval_dir}/hyperparameters.csv")
            # else:
            #     alpha = args.alpha
            #     lr = args.lr
            #     wd = args.wd
            #     dro = args.dropout
            # (1b) using optimal hyperparameters, train a model and test its performance on the holdout validation set.
            ## now train the whole dataset, retrain the model with the optimal parameters
            # train_dataset = TensorDataset(torch.from_numpy(x_train).float(),torch.from_numpy(y_train).long())
            # g = torch.Generator()
            # g.manual_seed(random_seed)
            # train_dataloader = DataLoader(train_dataset,
            #                               batch_size = args.batch_size,
            #                               num_workers=4,
            #                               worker_init_fn=seed_worker,
            #                               generator=g,
            #                               shuffle=True,
            #                               drop_last=True)
            trainer,survival_model = DL_single_run(x_train, y_train,model_name=model_name,lr=args.lr,
                                         alpha = args.alpha,
                                         wd=args.wd,
                                         batch_size=args.batch_size,
                                         dropout=args.dropout,
                                         train_from_scratch=args.train_from_scratch,
                                         logger = tb_logger,freeze_encoder=args.freeze_encoder,
                                         random_seed=random_seed,
                                         latent_code_dim=args.latent_code_dim,
                                         warm_up=args.warm_up,add_VAE_loss = args.add_VAE_loss)
            test_dataset = TensorDataset(torch.from_numpy(x_test).float(),torch.from_numpy(y_test).float())
            test_dataloader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False, drop_last=False) # create your dataloader
            ## test the model back using the train dataloader
            survival_model.eval()
            test_summary = trainer.test(survival_model,test_dataloader,ckpt_path="best")
            print ("c-index",survival_model.c_index)
            ## save the model
            torch.save(survival_model.state_dict(), f"{cval_dir}/best_model_c_index_{str(np.round(survival_model.c_index,4))}_lr_{survival_model.hparams.lr}.pth")
            ## save the test summary
            if args.search_hyperparam:
                pd.DataFrame(test_summary).to_csv(f"{cval_dir}/test_summary_with_hyperparam_search.csv")
            else: pd.DataFrame(test_summary).to_csv(f"{cval_dir}/test_summary.csv")

            # test_score = survival_model.test_score_list
            # test_y_ground_truth = survival_model.y_status_list
            # test_y_duration = survival_model.y_duration_list
            test_c_index = survival_model.c_index
            val_c_index_list.append(test_c_index)

      
        mean_cval = np.mean(val_c_index_list)
        std_cval = np.std(val_c_index_list)
  
        print ("mean c-index",mean_cval)
        print ("std c-index",std_cval)
        val_c_index_list.append(mean_cval)
        val_c_index_list.append(std_cval)

        ## save the c-index to the csv
        if args.search_hyperparam:
            pd.DataFrame(val_c_index_list).to_csv(f"{model_dir}/{random_seed}/c_index_mean_val_with_hyperparam_search.csv")
        else: pd.DataFrame(val_c_index_list).to_csv(f"{model_dir}/{random_seed}/c_index_mean_val.csv")
        return mean_cval, std_cval
    
 
    seed_list = [42, 2021, 2022, 2023, 2024]
    seed_list = [2023] if args.n_folds>2 else seed_list
    mean_cval_list = []
    std_cval_list = []
    for random_seed in seed_list:
        pl.seed_everything(random_seed)
        # encoder, decoder, max_epochs = get_model(args.model_name, args)
        mean_cval, std_cval = run_nested_cv(x,y, n_folds=args.n_folds,random_seed=random_seed,args=args)
        mean_cval_list.append(mean_cval)
        std_cval_list.append(std_cval)
    print(args.model_name)
    print ("estimated c-index over multiple runs using different seed",np.mean(mean_cval_list))
    print ("std c-index over multiple runs using different seed",np.mean(std_cval_list))
    mean_cval_list.append(np.mean(mean_cval_list))
    std_cval_list.append(np.mean(std_cval_list))
    df = pd.DataFrame(mean_cval_list,columns=["mean_c_index"])
    df["std_c_index"] = std_cval_list
    seed_list.append("AVG")
    df["seed"] = seed_list
    ## add the average c-index to the last row of the dataframe
    if args.search_hyperparam:
        df.to_csv(f"{model_dir}/mean_c_index_for_different_seeds_with_hyperparam_search.csv")
    else:df.to_csv(f"{model_dir}/mean_c_index_for_different_seeds.csv")
