import os
import sys
import io
from argparse import ArgumentParser
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from PIL import Image
from torchmetrics.classification import MultilabelAUROC
import numpy as np
from torchmetrics import MeanSquaredError
from torchmetrics import SignalNoiseRatio
import torch.nn.functional as F
import pandas as pd
import pickle
sys.path.append("./")
from multi_modal_heart.ECG.ecg_dataset import ECGDataset
from multi_modal_heart.model.ecg_net import ECGAE
from multi_modal_heart.ECG.ecg_utils import plot_multiframe_in_one_figure,arraytodataframe
from multi_modal_heart.model.marcel_ECG_network import ECGMarcelVAE
from multi_modal_heart.common.scheduler import get_cosine_schedule_with_warmup
from multi_modal_heart.model.ecg_net_attention import ECGAttentionAE
from multi_modal_heart.model.ecg_net import BenchmarkClassifier
from multi_modal_heart.common.metrics import MyDynamicTimeWarpingScore
## this file performs ECG pretraining
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, network,task_name,input_key_name="input_seq", target_key_name="cleaned_seq", 
        future_key_name="next_seq", grad_clip=False, warmup=50,
        max_iters=2000,batch_size=128, 
        latent_code_dim = 512,
        lr = 1e-3,args=None,**kwargs):
        '''
        this is a pytorch lightning module to support training/validation/testing of an autoencoder
        network: nn.Module, the network to be trained
        task_name: str, the name of the task
        input_key_name: str, the key name of the input in the batch
        target_key_name: str, the key name of the target in the batch
        future_key_name: str, the key name of the future in the batch (optional)
        grad_clip: bool, whether to apply gradient clipping
        warmup: int, the number of warmup steps
        max_iters: int, the maximum number of iterations
        batch_size: int, the batch size
        latent_code_dim: int, the dimension of the latent code
        lr: float, the learning rate
        '''
        super().__init__()
        self.network=network
        self.task_name = task_name
        self.save_hyperparameters(ignore=["network"])
        ## enable gradient clipping by set mannual optimization
        self.automatic_optimization = not grad_clip
        self.args = args
       
        ## add text encoder here if apply text-to-ECG alignment loss
        if args.ECG2Text or args.ECG2RawText:
            assert (args.ECG2Text and args.ECG2RawText)==False, "only one of ECG2Text and ECG2RawText can be enabled"
            ## preload saved patientwise text embedding from local disk
            if args.ECG2Text:
                pickle_path = "./pretrained_weights/text_embeddings/PTBXL_LLM_scp_structed_text_embedding.pkl"
            elif args.ECG2RawText:
                pickle_path = "./pretrained_weights/text_embeddings/PTBXL_LLM_raw_text.pkl"
            with open(pickle_path, 'rb') as f:
                self.patient_embedding_dict = pickle.load(f)
           
            if latent_code_dim==512:
                self.ecg_projector = nn.Sequential(
                                        nn.Linear(512, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 128),
                                        )
                self.text_projector =  nn.Sequential(
                                        nn.Linear(768, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 128),
                                        )
            else:
                self.ecg_projector = nn.Sequential(
                                        nn.Linear(latent_code_dim, latent_code_dim//2),
                                        nn.ReLU(),
                                        nn.Linear(latent_code_dim//2, latent_code_dim//2),
                                        )
                self.text_projector =  nn.Sequential(
                                        nn.Linear(768, latent_code_dim//2),
                                        nn.ReLU(),
                                        nn.Linear(latent_code_dim//2, latent_code_dim//2),
                                        )

        else: 
            self.ecg_projector = None
            self.text_projector = None

        ## add classifier if apply classification
        if args.recon_classification:
            if latent_code_dim==512:
                self.classifier = BenchmarkClassifier(input_size=512,hidden_size=128,output_size=5)
            elif latent_code_dim==256 or latent_code_dim==128 or latent_code_dim==64:
                self.classifier = BenchmarkClassifier(input_size=latent_code_dim,hidden_size=latent_code_dim//2,output_size=5)
        else:
            self.classifier = None

            
        """ number of parameters """
        num_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print('[Info] Number of encoder-decoder parameters: {}'.format(num_params))


        ## define metrics here
        self.rmse_metric = MeanSquaredError(squared=False)
        self.snr_metric = SignalNoiseRatio()
        self.dtw_metric = MyDynamicTimeWarpingScore()

        ## define classification metric here
        self.class_labels = ["NORM","MI","STTC","HYP","CD"]
        self.macro_auroc_metric = MultilabelAUROC(num_labels=len(self.class_labels), average="macro", thresholds=None)
        self.classwise_auroc_metric = MultilabelAUROC(num_labels=len(self.class_labels), average=None, thresholds=None)
        self.test_macro_auroc_metric = MultilabelAUROC(num_labels=len(self.class_labels), average="macro", thresholds=None)
        self.test_classwise_auroc_metric = MultilabelAUROC(num_labels=len(self.class_labels), average=None, thresholds=None)
        self.test_preds = []
        self.test_ground_truth = []
        self.max_iters = max_iters

       
    def run_task(self, batch,batch_idx, if_train=True, prefix=""):
         # training_step defines the train loop.
        # it is independent of forward
        ##<--get the input -->
        x= batch[self.hparams.input_key_name]
        ##<--get the supervision signal -->
        y = batch[self.hparams.target_key_name]

        ##<--get the output -->
        if "mask" in batch.keys(): 
            mask = batch["mask"]
            try:
                x_hat = self.network(x,mask=mask)
            except: 
                raise Exception
        else:
            mask = torch.ones_like(x)
            x_hat = self.network(x,mask=mask)
        
        if self.args.norm_output:
           ## 
           out_process_net = nn.Sequential(
               nn.InstanceNorm1d(x_hat.size(1)),
           )
           x_hat = out_process_net(x_hat)

        self.x_hat = x_hat
        self.y = y
        ##<--compute the loss -->
        ## only evaluate 
        if if_train:
            if not self.args.no_recon:
                loss = nn.functional.mse_loss(y, x_hat, reduction="none")
                loss = loss.mean()
            else:
                loss = torch.tensor(0.0,device=x_hat.device)
            self.log(f"{prefix}_mse", loss,batch_size=self.hparams.batch_size)
            if  "VAE" in self.task_name:
                assert self.network.if_VAE==True, "VAE loss is only applied to VAE model"
                beta = 1e-4
                log_var = self.network.z_log_var
                mu = self.network.z_mu
                if not self.args.disable_VAE_loss:
                    kld_loss = beta*torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                    loss = loss + kld_loss
                    self.log(f"train_VAE(KL)", kld_loss,batch_size=self.hparams.batch_size,prog_bar=True)
                else:
                    kld_loss = torch.tensor(0.0,device=x_hat.device)
                    
            if self.args.ECG2Text or self.args.ECG2RawText:
                ## ecg report alignment loss
                ## ecg report alignment loss
                ecg_feature = self.network.z 
                ecg_proj_features = self.ecg_projector(ecg_feature)

                ## text feature
                ecg_ids = batch["ecg_id"]
                text_features = []
                for ecg_id in ecg_ids:
                    if isinstance(self.patient_embedding_dict[ecg_id.item()],list):
                        text_features.append(torch.from_numpy(self.patient_embedding_dict[ecg_id.item()][0]))
                    else:
                        text_features.append(torch.from_numpy(self.patient_embedding_dict[ecg_id.item()]))
                batched_text_features = torch.stack(text_features,dim=0)  ## batch_size x 768
                batched_text_features = batched_text_features.to(ecg_proj_features.device)
                text_proj_features = self.text_projector(batched_text_features)

                ## compute the alignment loss
                z1 = F.normalize(ecg_proj_features, p=2, dim=1)
                z2 = F.normalize(text_proj_features, p=2, dim=1)
                alignment_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')(z1,z2,target=torch.ones(z1.shape[0],device=z1.device))
                self.log(f"ecg-report alignment loss", alignment_loss,batch_size=z1.shape[0],prog_bar=True)
                loss += alignment_loss ## for pretraining

            if self.args.recon_classification:
                target = batch["super_class_encoding"].float()
                outputs_before_sigmoid = self.classifier(self.network.z)
                cls_loss = torch.nn.BCEWithLogitsLoss()(outputs_before_sigmoid,target)
                self.log("BCE loss", cls_loss,batch_size=outputs_before_sigmoid.shape[0])
                loss += cls_loss ## for pretraining
            return x_hat, loss

        else:
            ## at test time, only compute the losses around no-masking region or no-padding region.
            loss = nn.functional.mse_loss(y*mask, x_hat*mask, reduction="none")
            every_lead_loss = loss.mean(dim=[0,2])
            loss = loss.sum()/mask.sum()
            if prefix =="test":
                pass
           
            ## Compute loss for each lead (channel-wise)
            return x_hat,loss,every_lead_loss
        

    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        if not self.automatic_optimization:
            opt = self.optimizers()
            x_hat, loss = self.run_task(batch,batch_idx,if_train=True)
            opt.zero_grad()
            self.manual_backward(loss)
            # clip gradients for stability 
            if self.hparams.grad_clip: 
                self.clip_gradients(opt, gradient_clip_val=1, gradient_clip_algorithm="norm")
            opt.step()
        else:
            x_hat, loss = self.run_task(batch,batch_idx,if_train=True)

        self.log("train_loss", loss,batch_size=self.hparams.batch_size)
        return loss

    def configure_optimizers(self):
        # Make sure to filter the parameters based on `requires_grad`
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr, betas=(0.9, 0.95))
        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        if self.args.warm_up:
            num_warmup_steps = int(self.hparams.max_iters*0.2)
            self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps= num_warmup_steps, num_training_steps= self.hparams.max_iters)

        else: self.lr_scheduler = None
        return optimizer

    def validation_step(self, batch, batch_idx):
        x_hat, loss,every_lead_loss = self.run_task(batch,batch_idx, if_train=False,prefix="val")
        if  len(every_lead_loss)==12:
            labels = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
        elif len(every_lead_loss)==8:
            labels = ["I","II","V1","V2","V3","V4","V5","V6"]
        else:
            raise NotImplementedError
        assert len(labels)==len(every_lead_loss), "lead number not match, found {} leads and {} labels".format(len(every_lead_loss),len(labels))
        for i, lead_name in enumerate(labels):
            self.log("val_loss/"+lead_name, every_lead_loss[i])
        self.log("val_loss", loss,batch_size=self.hparams.batch_size)
        ## record the classification performance
        if self.classifier is not None:
            target = batch["super_class_encoding"].float()
            outputs_before_sigmoid = self.classifier(self.network.z)
            cls_loss = torch.nn.BCEWithLogitsLoss()(outputs_before_sigmoid,target)
            self.log("val_BCE_loss", cls_loss,batch_size=outputs_before_sigmoid.shape[0])
            self.classwise_auroc_metric.update(outputs_before_sigmoid,target.long())
            self.macro_auroc_metric.update(outputs_before_sigmoid,target.long())



    
    def test_step(self, batch, batch_idx):
        x_hat, loss, every_lead_loss = self.run_task(batch,batch_idx,if_train=False,prefix="test")
        self.log("test_loss", loss)

        if len(every_lead_loss)==12:
            labels = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
        elif len(every_lead_loss)==8:
            labels = ["I","II","V1","V2","V3","V4","V5","V6"]
        else:
            raise NotImplementedError
        assert len(labels)==len(every_lead_loss), "lead number not match, found {} leads and {} labels".format(len(every_lead_loss),len(labels))
        for i, lead_name in enumerate(labels):
            self.log("test_loss/"+lead_name, every_lead_loss[i],batch_size=self.hparams.batch_size)
        ## record the original dataset
        self.rmse_metric(x_hat, self.y)
        self.snr_metric(x_hat, self.y)
        self.dtw_metric(x_hat, self.y)

        ## record the classification performance
        if self.classifier is not None:
            target = batch["super_class_encoding"].float()
            outputs_before_sigmoid = self.classifier(self.network.z)
            cls_loss = torch.nn.BCEWithLogitsLoss()(outputs_before_sigmoid,target)
            self.log("test_BCE_loss", cls_loss,batch_size=outputs_before_sigmoid.shape[0])
            self.test_classwise_auroc_metric.update(outputs_before_sigmoid,target.long())
            self.test_macro_auroc_metric.update(outputs_before_sigmoid,target.long())
    
   
    def on_train_epoch_end(self):
        print (self.hparams.task_name,)
        
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.args.warm_up:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()  # Step per iteration

    
    def on_validation_epoch_end(self):
        if self.current_epoch==0 or self.current_epoch%10==0:
            self.plot_recon_ECG(title="val/ECG_recon")
        if self.classifier is not None:
            macro_auc = self.macro_auroc_metric.compute()
            self.log("val_macro_auc", macro_auc)
            classwise_auc = self.classwise_auroc_metric.compute()
            print ("macro_auc",macro_auc.item())
            for i in range(classwise_auc.shape[0]):
                self.log(f"val_auc_{self.class_labels[i]}", classwise_auc[i])
            self.classwise_auroc_metric.reset()
            self.macro_auroc_metric.reset()

    def on_test_epoch_end(self):
        if self.current_epoch==0 or self.current_epoch%10==0:
            self.plot_recon_ECG(title="test/ECG_recon")
        rmse_score = self.rmse_metric.compute()
        snr_score = self.snr_metric.compute()
        self.log("test_rmse",rmse_score)
        self.log("test_snr",snr_score)
        self.log("test_dtw",self.dtw_metric.compute())
        self.rmse_metric.reset()
        self.snr_metric.reset()
        self.dtw_metric.reset()
        print (self.hparams.task_name)
        if self.classifier is not None:
            macro_auc = self.test_macro_auroc_metric.compute()
            self.log("test_macro_auc", macro_auc)
            classwise_auc = self.test_classwise_auroc_metric.compute()
            print ("macro_auc",macro_auc.item())
            for i in range(classwise_auc.shape[0]):
                self.log(f"test_auc_{self.class_labels[i]}", classwise_auc[i])
            self.test_classwise_auroc_metric.reset()
            self.test_macro_auroc_metric.reset()

    def plot_recon_ECG(self, title="train/ECG_recon"):
        sample_y_nd  =self.y.data.detach().cpu().numpy()[0]
        sample_x_hat_nd  =self.x_hat.data.detach().cpu().numpy()[0]
        '''
        sample_y_nd: 12*L
        sample_x_hat_nd: 12*L
        '''
        y_df = arraytodataframe(sample_y_nd)
        y_recon = arraytodataframe(sample_x_hat_nd)

        if sample_x_hat_nd.shape[0]==12:
            lead_names= ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
            figure_arrangement=(4,3)
        elif sample_x_hat_nd.shape[0]==8:
            lead_names= ['I','II','V1','V2','V3','V4','V5','V6']
            figure_arrangement=(4,2)
        else:
            raise NotImplementedError
        y_df.columns = ["GT "+k for k in lead_names]
        y_recon.columns = ["Pred. "+k for k in lead_names]
        figure = plot_multiframe_in_one_figure([y_df,y_recon],figsize=(15,4), figure_arrangement=figure_arrangement, logger=self.logger,epoch=self.current_epoch, title=title)
        return figure

def cli_main():
    parser = ArgumentParser()
    # Trainer arguments
    parser.add_argument("--device", type=int, default=1)
    # Hyperparameters for the model
    parser.add_argument("--ae_type", type=str)
    parser.add_argument("--decoder_type", type=str,default="ms_resnet")
    ## optimization parameters
    parser.add_argument("--grad_clip", action="store_true", default=False)
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--checkpoint_path",  type=str, default="")
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--no_pretrain", action="store_true", default=False)
    parser.add_argument("--no_test", action="store_true", default=False)

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--no_log", action="store_true", default=False)
    ## TEST parameters
    parser.add_argument("--norm_output", action="store_true", default=False)
    parser.add_argument("--mask_half", action="store_true", default=False)
    parser.add_argument("--warm_up", action="store_true", default=False)
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--joint_training", action="store_true", default=False)
    parser.add_argument("--add_snr_loss", action="store_true", default=False)
    parser.add_argument("--finetune_taskname", type=str,default="finetuning")
    parser.add_argument("--encoder_lr", type=float,default=1e-2)
    parser.add_argument("--gradual_unfreeze", action="store_true", default=False)
    parser.add_argument("--use_median_wave", action="store_true", default=False)
    parser.add_argument("--ECG2Text", action="store_true", default=False)
    parser.add_argument("--ECG2RawText", action="store_true", default=False)
    parser.add_argument("--apply_uniformity_loss", action="store_true", default=False)
    parser.add_argument("--no_recon", action="store_true", default=False)
    parser.add_argument("--recon_classification", action="store_true", default=False)
    parser.add_argument("--freeze_encoder", action="store_true", default=False)
    parser.add_argument("--disable_VAE_loss", action="store_true", default=False)

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()
    # ------------
    # reproducibility
    # ------------
    if args.seed!="":
        pl.seed_everything(args.seed)
        deterministic=True
    else: deterministic = False
    torch.backends.cudnn.deterministic = deterministic

    
    # ------------
    # data
    # ------------
    ## set up the data
    ## initialize a dataloader
    ## contains ecg raw wave data and the corresponding report information
    data_folder = "./data/ptbxl/"
    ## data with report information, we select only those reports verified by experts for pretraining
    train_data_statement_path =os.path.join(data_folder,"high_quality_split/Y_train.csv")
    validate_data_statement_path =os.path.join(data_folder,"high_quality_split/Y_validate.csv")
    ## original test loader (incl. artefacted data)
    all_test_data_statement_path = os.path.join(data_folder, "raw_split/Y_test.csv")
 

    data_loaders = []
    n_leads = 12
    batch_size  = 128
    if args.use_median_wave:
        max_seq_len = 608
        sampling_rate=500
    else:
        sampling_rate=100
        max_seq_len = 1024 ## using 1024 for the 12 lead ECG, padding with 0 
    max_epochs= 300
    data_proc_config={
                    "if_clean":False, ## if perform signal cleaning first
                    }
    data_aug_config={
                    ## you can change the config here to add noise augmentation
                    "noise_frequency_list":[0],
                    "noise_amplitude_range":[0.,0.],
                    "powerline_frequency_list":[0,0.],
                    "powerline_amplitude_range":[0.,0.],
                    "artifacts_amplitude_range":[0.,0.],
                    "artifacts_number_range":[0,0],
                    "linear_drift_range":[0.,0.],
                    "artifacts_frequency_list":[5],
                    "random_prob":0, ## data augmentation prob
                    ## we turn on the mask augmentation
                    "if_mask_signal":True, ## change it to True
                    "mask_whole_lead_prob":0.5, ## mask the whole lead
                    "lead_mask_prob":0.2, ## mask certain parts of the lead
                    "region_mask_prob":0.2,
                    "mask_length_range":[0.08, 0.18],
                    "mask_value":0.0,
                    "random_drop_half_prob":0.2,
                    }
    if args.mask_half:
        data_aug_config["random_drop_half_prob"]=0.5
    if args.use_median_wave: 
        ## turn off the region mask augmentation if the input is median wave
        data_aug_config["region_mask_prob"]=0
        data_aug_config["random_drop_half_prob"]=0

  
    dataset_path_list={
            "train": train_data_statement_path,
            "validate":validate_data_statement_path,
            "test_all":all_test_data_statement_path
        }
    if not args.no_pretrain and not args.no_test:
        dataset_path_list =dataset_path_list
    else:
        if args.no_pretrain and not args.no_test:
            dataset_path_list={
                "test_all":all_test_data_statement_path,
            }
        elif args.no_pretrain and args.no_test:
             dataset_path_list={}



    data_loaders={}
    for split_name,label_csv_path in dataset_path_list.items():
        if_test ="test" in split_name
        if_train ="train" in split_name
        dataset = ECGDataset(data_folder,label_csv_path=label_csv_path,
                            n_leads=n_leads,
                            sampling_rate=sampling_rate,
                            max_seq_len=max_seq_len,
                            use_median_wave=args.use_median_wave,
                            augmentation=if_train,
                            data_proc_config=data_proc_config,
                            data_aug_config=data_aug_config,)
        data_loader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                num_workers=4,
                                shuffle = if_train,
                                drop_last= if_train,
                                )
        print ('load {} data: {} samples'.format(label_csv_path.split("/")[-1],len(dataset)))
        data_loaders[split_name]=data_loader
        
    # ------------
    # model
    # ------------
    ## set up the model
    ae_type = args.ae_type  
    latent_code_dim = 512

    if ae_type=="ECG_attention_512":
        ecg_net = ECGAttentionAE(num_leads=n_leads, time_steps=max_seq_len, 
                                 z_dims=512, 
                                 linear_out=512, 
                                 downsample_factor=5, 
                                 base_feature_dim=4,
                                 if_VAE=False,
                                 use_attention_pool=False,
                                 no_linear_in_E=True)
    ## ablation study: 
    elif ae_type=="ECG_attention_512_raw_no_attention_pool_no_linear_abl_no_lead_attention":
        ecg_net = ECGAttentionAE(num_leads=n_leads, time_steps=max_seq_len, z_dims=512, linear_out=512, 
                                 downsample_factor=5, base_feature_dim=4,if_VAE=False,
                                 use_attention_pool=False, no_lead_attention=True,no_linear_in_E=True)
    elif ae_type=="ECG_attention_512_raw_no_attention_pool_no_linear_abl_no_time_attention":
        ecg_net = ECGAttentionAE(num_leads=n_leads, time_steps=max_seq_len, z_dims=512, linear_out=512, downsample_factor=5, base_feature_dim=4,if_VAE=False,use_attention_pool=False, 
                                no_time_attention=True,
        no_linear_in_E=True,apply_lead_mask=False)
    elif ae_type=="ECG_attention_512_raw_no_attention_pool_no_linear_abl_no_lead_time_attention":
        ecg_net = ECGAttentionAE(num_leads=n_leads, time_steps=max_seq_len, z_dims=512, linear_out=512, downsample_factor=5, base_feature_dim=4,if_VAE=False,use_attention_pool=False, no_time_attention=True, no_lead_attention=True,no_linear_in_E=True)
   
    ## baseline methods:
    elif ae_type=="resnet1d101_512":
        ecg_net= ECGAE(encoder_type="resnet1d101",in_channels=n_leads,ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=512,
                    add_time=False,
                    encoder_mha = False,
                    apply_method="",
                    decoder_outdim=n_leads)
    elif ae_type=="marcelVAE_512":
        ecg_net = ECGMarcelVAE(num_leads=n_leads,time_steps=max_seq_len,z_dims=512) ## for ease of comparison

    else:
        raise NotImplementedError
    
    log_fix = "log_median" if args.use_median_wave else "log"
    log_name= ae_type+"_"+args.decoder_type
    log_name = log_name+"_no_recon" if args.no_recon else log_name
    log_name = log_name+"_ECG2Text" if args.ECG2Text else log_name
    log_name = log_name+"_ECG2RawText" if args.ECG2RawText else log_name
    log_name = log_name+"_recon_classification" if args.recon_classification else log_name
    log_name = log_name+"_disable_VAE_loss" if args.disable_VAE_loss else log_name

    if args.mask_half:
        log_name = log_name+"_mask_half"
    task_name = ae_type
    if not args.no_pretrain: 
        max_iters= max_epochs * len(data_loaders["train"])
    else: max_iters = 0

    autoencoder = LitAutoEncoder(ecg_net,task_name=ae_type,grad_clip = args.grad_clip, 
                                 max_iters=max_iters,
                                 latent_code_dim = latent_code_dim,
                                 batch_size=batch_size,args=args)
    
    if args.checkpoint_path !="" and os.path.exists(args.checkpoint_path):
        autoencoder = LitAutoEncoder.load_from_checkpoint(args.checkpoint_path)
        print ("auto encoder is loaded from {}".format(args.checkpoint_path))
    if args.pretrained:
        if args.checkpoint_path=="":
            ## check default saving checkpoint path
            checkpoint_path = os.path.join(f"./{log_fix}",log_name,"checkpoints","epoch=299-step=10200.ckpt")
            if os.path.exists(checkpoint_path):
                autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint_path)
                print ("auto encoder is loaded from {}".format(checkpoint_path))
            else:
                print (f"{checkpoint_path} does not exist")
        elif args.checkpoint_path !="" and os.path.exists(args.checkpoint_path):
            autoencoder = LitAutoEncoder.load_from_checkpoint(args.checkpoint_path)
            print ("auto encoder is loaded from {}".format(args.checkpoint_path))
        else:
            Warning("auto encoder is randomly initialized")
    
    # ------------
    # training
    # ------------
    if args.no_log or args.no_pretrain:
            ## move to temp_log, which can be deleted latter
            tb_logger = TensorBoardLogger( f"./temp_{log_fix}", name=log_name, version="")  
        
    else: tb_logger = TensorBoardLogger( f"./{log_fix}", name=log_name, version="")  
    checkpoint_dir  = os.path.join(tb_logger.log_dir,"checkpoints")
    checkpoint_callback_best_loss_min_pretrain = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, 
                                                           filename='checkpoint_best_loss',
                                                           save_top_k=1, monitor="val_loss"
                                                           , mode='min',save_last=True)
    trainer = pl.Trainer(accelerator="gpu",
                         devices=args.device, max_epochs=max_epochs,
                         logger=tb_logger,
                         log_every_n_steps=5,
                         callbacks=[checkpoint_callback_best_loss_min_pretrain],
                         ) 
   

                        
    if not args.no_pretrain:
       trainer.fit(autoencoder, data_loaders["train"],data_loaders["validate"])

    # ------------
    # testing
    # ------------
    ## evaluation based on the signal reconstruction
    autoencoder.args = args
    if not args.no_test:
        result = trainer.test(autoencoder, data_loaders["test_all"])
        ## save result to csv
        df = pd.DataFrame(result)
        df.to_csv(os.path.join(tb_logger.log_dir,"recon_result.csv"))
        
if __name__=="__main__":
    cli_main()