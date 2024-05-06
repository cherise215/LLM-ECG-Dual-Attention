
    ##test
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
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from torchmetrics.classification import MultilabelAUROC
import numpy as np
from torchmetrics import MeanSquaredError
from torchmetrics import SignalNoiseRatio
import torch.nn.functional as F
import pandas as pd

## add ray 

sys.path.append("./")
from multi_modal_heart.ECG.ecg_dataset import ECGDataset
from multi_modal_heart.model.ecg_net import ECGAE, doubleECGNet
from multi_modal_heart.model.my_ecg_transformer import ECG_transformer
from multi_modal_heart.ECG.ecg_utils import plot_multiframe_in_one_figure,arraytodataframe
from multi_modal_heart.model.custom_loss import cal_dtw_loss, calc_SNR_loss, calc_scale_invariant_SNR_loss
from multi_modal_heart.model.marcel_ECG_network import ECGMarcelVAE
from multi_modal_heart.model.ecg_lstm_model import ECGLSTMnet
from multi_modal_heart.common.scheduler import CosineWarmupScheduler,get_cosine_schedule_with_warmup
from multi_modal_heart.model.ecg_net_attention import ECGAttentionAE
from multi_modal_heart.model.ecg_net import ClassifierMLP,BenchmarkClassifier
from multi_modal_heart.ECG.utils import evaluate_experiment
from multi_modal_heart.common.metrics import MyDynamicTimeWarpingScore
from multi_modal_heart.model.custom_loss import CLIPLoss
## this file performs ECG pretraining
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, network,task_name,input_key_name="input_seq", target_key_name="cleaned_seq", 
        future_key_name="next_seq", grad_clip=False, warmup=50,
        max_iters=2000,batch_size=128, 
        lr = 1e-3,args=None,**kwargs):
        super().__init__()
        self.network=network
        self.task_name = task_name
        self.save_hyperparameters()
        ## enable gradient clipping by set mannual optimization
        self.automatic_optimization = not grad_clip
        self.args = args

        ## add text encoder here if apply text-to-ECG alignment loss
        if args.ECG2Text:
            ### change model to the clinical bert
            # from transformers import AutoTokenizer, AutoModel
            # self.text_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            # self.text_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            import pickle
            pickle_path = "/home/engs2522/project/multi-modal-heart/multi_modal_heart/pretrained/patient_embedding_dict_summed.pkl"
            with open(pickle_path, 'rb') as f:
                self.patient_embedding_dict = pickle.load(f)
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
            self.ecg_projector = None
            self.text_projector = None

            
        """ number of parameters """
        num_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print('[Info] Number of encoder-decoder parameters: {}'.format(num_params))


        ## define metrics here
        self.rmse_metric = MeanSquaredError(squared=False)
        self.snr_metric = SignalNoiseRatio()
        self.dtw_metric = MyDynamicTimeWarpingScore()

    
       
    def run_task(self, batch,batch_idx, if_train=True, prefix=""):
         # training_step defines the train loop.
        # it is independent of forward
        ##<--get the input -->
        if "identity" in self.task_name:
            x = batch[self.hparams.target_key_name]
        else:
            x= batch[self.hparams.input_key_name]

        if "shuffle_lead" in self.task_name and if_train:
            print ("lead shuffling enabled")
            shuffle_prob = 0.5
            if torch.rand(1)>shuffle_prob:
                x = x[:,torch.randperm(x.size(1)),:]
        if "use_template" in self.task_name:
            template = batch["template"]
            x = torch.cat([x,template],dim=1)

        ##<--get the supervision signal -->
        y = batch[self.hparams.target_key_name]

        if "predict_feature" in self.hparams.task_name and self.hparams.future_key_name in batch.keys():
            y_future= batch[self.hparams.future_key_name]
            y = torch.cat([y, y_future],dim=1)

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
            loss = nn.functional.mse_loss(y, x_hat, reduction="none")
            loss = loss.mean()
            self.log(f"{prefix}_mse", loss,batch_size=self.hparams.batch_size)
            if  "VAE" in self.task_name:
                assert self.network.if_VAE==True, "VAE loss is only applied to VAE model"
                beta = 1e-4
                log_var = self.network.z_log_var
                mu = self.network.z_mu
                kld_loss = beta*torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                loss = loss + kld_loss
                self.log(f"train_VAE(KL)", kld_loss,batch_size=self.hparams.batch_size)
            
            if self.args.ECG2Text:
                ## ecg report alignment loss
                ecg_feature = self.network.z
                ecg_proj_features = self.ecg_projector(ecg_feature)

                ## text feature
                ecg_ids = batch["ecg_id"]
                text_features = []
                for ecg_id in ecg_ids:
                    text_features.append(torch.from_numpy(self.patient_embedding_dict[ecg_id.item()]))
                batched_text_features = torch.stack(text_features,dim=0)  ## batch_size x 768
                batched_text_features = batched_text_features.to(ecg_proj_features.device)
                text_proj_features = self.text_projector(batched_text_features)

                ## compute the alignment loss
                z1 = F.normalize(ecg_proj_features, p=2, dim=1)
                z2 = F.normalize(text_proj_features, p=2, dim=1)
                # alignment_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')(z1,z2,target=torch.ones(z1.shape[0],device=z1.device))
                alignment_loss = CLIPLoss()(z1,z2)
                loss += alignment_loss ## for pretraining
                self.log(f"ecg-report alignment loss", alignment_loss,batch_size=self.hparams.batch_size)
                # ## maximize the uniformity loss
                # ecg_uniformity_loss = UniformityLoss(batch_size=z1.size(0),z_dim=z1.size(1))
                # self.log(f"ecg_uniformity_loss", ecg_uniformity_loss,batch_size=self.hparams.batch_size)
                # loss +=ecg_uniformity_loss
            



           
            if "LeadCorrLoss" in self.task_name:
                assert x_hat.size(1)==12, "LeadCorrLoss is only applied to 12-lead ECG"
                ## add physics informed linear correlation loss
                lead_I = x_hat[:,0,:]
                lead_II = x_hat[:,1,:]
                lead_III = x_hat[:,2,:]
                neative_lead_aVR = -x_hat[:,3,:]
                lead_aVL = x_hat[:,4,:]
                lead_aVF = x_hat[:,5,:]
                length = lead_I.shape[0]
                target = torch.ones(length,device = lead_aVF.device)
                einthoven_loss = 0.5*torch.nn.CosineEmbeddingLoss()(lead_II-lead_I,lead_III,target=target).mean()
                goldberger_loss = (torch.nn.CosineEmbeddingLoss()(0.5*lead_I-0.5*lead_III,lead_aVL,target=target)+
                                  torch.nn.CosineEmbeddingLoss()(0.5*lead_I+0.5*lead_II,neative_lead_aVR,
                                                                 target =target)+
                                  torch.nn.CosineEmbeddingLoss()(0.5*(lead_II+lead_III),lead_aVF, target=target)).mean()
                weight = 1e-3
                loss = loss + weight*(einthoven_loss + goldberger_loss)
                self.log(f"LeadCorrLoss",  weight*(einthoven_loss + goldberger_loss),batch_size=self.hparams.batch_size)
            if self.args.add_snr_loss:
                weight = 1e-4
                snr_loss = weight*calc_scale_invariant_SNR_loss(preds=x_hat, target=y)
                loss+=snr_loss
                self.log(f"{prefix}_SNR_loss", snr_loss,batch_size=self.hparams.batch_size)
            return x_hat,loss
        else:
            ## only compute the losses around no-masking region or no-padding region.
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
    
    # def optimizer_step(self, *args, **kwargs):
    #     super().optimizer_step(*args, **kwargs)
    #     if self.lr_scheduler is not None:
    #         self.lr_scheduler.step()  # Step per iteration

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
    
   
    def on_train_epoch_end(self):
        print (self.hparams.task_name,)
        ##  log the figure of the reference signal and reconstructed signal
        ## 12*L
        # if self.current_epoch==0 or self.current_epoch%10==0:
        #     self.plot_recon_ECG(title="train/ECG_recon")
        
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.args.warm_up:self.lr_scheduler.step()  # Step per iteration

    def on_validation_epoch_end(self):
        if self.current_epoch==0 or self.current_epoch%10==0:
            self.plot_recon_ECG(title="val/ECG_recon")
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


        
    def plot_recon_ECG(self, title="train/ECG_recon"):
        sample_y_nd  =self.y.data.detach().cpu().numpy()[0]
        sample_x_hat_nd  =self.x_hat.data.detach().cpu().numpy()[0]
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


## define finetuning classification task here:
class LitClassifier(pl.LightningModule):
    def __init__(self,encoder, input_dim,num_classes=5,learning_rate=1e-3, encoder_lr=1e-3,freeze_encoder=False, gradual_unfreeze=True,
                 task_name = "ECG_Classifier", max_iters =20000,output_dir="./",ecg_projector=None,text_projector=None, decoder=None, args=None,**kwargs):
        super().__init__()
        
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.freeze_encoder = freeze_encoder
        self.learning_rate = learning_rate
        self.encoder_lr = encoder_lr
        self.task_name  = task_name
        self.args =args

        if args.recon:
            ## 
            print ("use reconstruction loss")
            self.decoder = decoder
            assert self.decoder is not None, "decoder is not defined"
        else:
            self.decoder = None

        if args.ECG2Text:
            ### change model to the clinical bert
            # from transformers import AutoTokenizer, AutoModel
            # self.text_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            # self.text_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            import pickle
            pickle_path = "/home/engs2522/project/multi-modal-heart/multi_modal_heart/pretrained/patient_embedding_dict_summed.pkl"
            with open(pickle_path, 'rb') as f:
                self.patient_embedding_dict = pickle.load(f)
            if ecg_projector is not None:
                self.ecg_projector = ecg_projector
            else:
                self.ecg_projector = nn.Sequential(
                                        nn.Linear(input_dim, input_dim),
                                        nn.ReLU(),
                                        nn.Linear(input_dim, 256),
                                        )
            if text_projector is not None:
                self.text_projector = text_projector
            else:
                self.text_projector =  nn.Sequential(
                                        nn.Linear(768, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 256),
                                        )
        else: 
            self.ecg_projector = None
            self.text_projector = None


              #### add classifier if use benchmark classifier
        if "benchmark_classifier" in self.task_name:
            print ("use benchmark classifier")
            # input_dim = 1024
            assert input_dim==512, "benchmark classifier is only applied to 512-dim latent code"
            self.downsteam_net = BenchmarkClassifier(input_size=input_dim,hidden_size=128,output_size=num_classes)
        else:
            ## hard code the classifier here
            self.downsteam_net = ClassifierMLP(input_size=input_dim,hidden_sizes=[256],output_size=num_classes)

        self.output_dir = output_dir
        self.latent_code_dim =input_dim
        self.test_preds = []
        self.test_ground_truth = []
        self.max_iters = max_iters
        self.gradual_unfreeze=gradual_unfreeze
        self.save_hyperparameters()

        ## define metrics here
        self.class_labels = ["NORM","MI","STTC","HYP","CD"] 
        assert len(self.class_labels)==num_classes, "class number not match"
        self.macro_auroc_metric = MultilabelAUROC(num_labels=num_classes, average="macro", thresholds=None)
        self.classwise_auroc_metric = MultilabelAUROC(num_labels=num_classes, average=None, thresholds=None)
        self.test_macro_auroc_metric = MultilabelAUROC(num_labels=num_classes, average="macro", thresholds=None)
        self.test_classwise_auroc_metric = MultilabelAUROC(num_labels=num_classes, average=None, thresholds=None)
    
    def forward(self, x, mask=None, eval=False):
        if self.freeze_encoder or eval:
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.encoder.parameters():
                param.requires_grad = True
        if "benchmark_classifier" in self.task_name:
            latent_code = self.encoder.get_features_after_pooling(x,mask)
        else:
            latent_code = self.encoder(x)
        self.latent_code = latent_code
        return self.downsteam_net(latent_code)
    
    def run_task(self, batch, batch_idx, prefix_name=""):
        input = batch["input_seq"]
        target = batch["super_class_encoding"].float()
        if "mask" in batch.keys(): 
            mask = batch["mask"]
        else:
            mask = torch.ones_like(input)
        if prefix_name=="train":
            ## freeze for the first 20 epochs before finetuning
            if self.gradual_unfreeze:
                if self.global_step >20:
                    eval=False 
                else:
                    eval=True
            else:
                eval=False
        else: eval=False
        outputs_before_sigmoid = self.forward(input,mask,eval = eval)
        loss = torch.nn.BCEWithLogitsLoss()(outputs_before_sigmoid,target)
        self.log(f"{self.task_name}/BCE loss", loss,batch_size=outputs_before_sigmoid.shape[0])

        if prefix_name=="train":
            if self.args.ECG2Text:
                ## ecg report alignment loss
                ecg_feature = self.latent_code 
                ecg_proj_features = self.ecg_projector(ecg_feature)

                ## text feature
                ecg_ids = batch["ecg_id"]
                text_features = []
                for ecg_id in ecg_ids:
                    text_features.append(torch.from_numpy(self.patient_embedding_dict[ecg_id.item()]))
                batched_text_features = torch.stack(text_features,dim=0)  ## batch_size x 768
                batched_text_features = batched_text_features.to(ecg_proj_features.device)
                text_proj_features = self.text_projector(batched_text_features)

                ## compute the alignment loss
                z1 = F.normalize(ecg_proj_features, p=2, dim=1)
                z2 = F.normalize(text_proj_features, p=2, dim=1)
                alignment_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')(z1,z2,target=torch.ones(z1.shape[0],device=z1.device))
                self.log(f"{self.task_name}/ecg-report alignment loss", alignment_loss,batch_size=z1.shape[0])
                loss += alignment_loss ## for pretraining

        self.log(f"{self.task_name}/{prefix_name}_loss", loss)
        return loss,torch.sigmoid(outputs_before_sigmoid),target

    def training_step(self, batch, batch_idx):
        loss, _,_ = self.run_task(batch, batch_idx, prefix_name="train")
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW([
            {'params':self.encoder.parameters() , 'lr': self.encoder_lr or self.lr,},
            {'params':self.downsteam_net.parameters(), 'lr': self.learning_rate or self.lr},
        ],betas=(0.9, 0.95))
        if self.args.warm_up:
            self.lr_scheduler =get_cosine_schedule_with_warmup(optimizer, num_warmup_steps= int(0.2*self.hparams.max_iters), num_training_steps= self.hparams.max_iters)
        else: self.lr_scheduler = None
        ## one cylce lr
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=20,
        #     num_training_steps=self.trainer.estimated_stepping_batches,
        # )
        # scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        # self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,total_steps = self.max_iters,anneal_strategy='cos')
        return [optimizer] #, [scheduler]
    def on_train_epoch_end(self):
        print (self.task_name)
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.args.warm_up: self.lr_scheduler.step()  # Step per iteration
    def validation_step(self, batch, batch_idx):
        loss,pred,target= self.run_task(batch, batch_idx, prefix_name="val")
        self.classwise_auroc_metric.update(pred,target.long())
        self.macro_auroc_metric.update(pred,target.long())
        return loss
    
    def on_validation_epoch_end(self):
        ## gather all the predictions and ground truth
        macro_auc = self.macro_auroc_metric.compute()
        self.log(f"{self.task_name}/val_macro_auc", macro_auc)
        classwise_auc = self.classwise_auroc_metric.compute()
        print ("macro_auc",macro_auc.item())
        for i in range(classwise_auc.shape[0]):
            self.log(f"{self.task_name}/val_auc_{self.class_labels[i]}", classwise_auc[i])
       
        self.classwise_auroc_metric.reset()
        self.macro_auroc_metric.reset()
        

    def test_step(self, batch, batch_idx):
        loss, pred,target = self.run_task(batch, batch_idx, prefix_name="test")
        self.test_preds.append(pred)
        self.test_ground_truth.append(target)
        self.test_classwise_auroc_metric.update(pred,target.long())
        self.test_macro_auroc_metric.update(pred,target.long())
    

  
    def on_test_epoch_end(self):
        ## gather all the predictions and ground truth
        test_classwise = self.test_classwise_auroc_metric.compute()
        test_summary = self.test_macro_auroc_metric.compute()
        self.log(f"{self.task_name}/test_macro_auc", test_summary)
        for i in range(test_classwise.shape[0]):
            self.log(f"{self.task_name}/test_auc_{self.class_labels[i]}", test_classwise[i])
        print (test_summary)
        print (test_classwise)
        self.test_classwise_auroc_metric.reset()
        self.test_macro_auroc_metric.reset()

        ## save the predictions and ground truth to the disk:
        test_pred_all = torch.cat(self.test_preds,dim=0).cpu().numpy()
        test_gt_all = torch.cat(self.test_ground_truth,dim=0).cpu().numpy()
        np.save(os.path.join(self.output_dir,"test_pred.npy"),test_pred_all)
        np.save(os.path.join(self.output_dir,"test_gt.npy"),test_gt_all)
        return test_classwise,test_summary
class FineTuneLearningRateFinder(pl.callbacks.LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)

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
    parser.add_argument("--mask_half", action="store_true", default=False)
    parser.add_argument("--norm_output", action="store_true", default=False)
    parser.add_argument("--warm_up", action="store_true", default=False)
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--joint_training", action="store_true", default=False)
    parser.add_argument("--add_snr_loss", action="store_true", default=False)
    parser.add_argument("--finetune_taskname", type=str,default="finetuning")
    parser.add_argument("--encoder_lr", type=float,default=1e-2)
    parser.add_argument("--gradual_unfreeze", action="store_true", default=False)
    parser.add_argument("--use_median_wave", action="store_true", default=False)
    parser.add_argument("--ECG2Text", action="store_true", default=False)

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()
    # ------------
    # reproducibility
    # ------------
    if args.seed!="":
        pl.seed_everything(args.seed)
        deterministic=True
    else: deterministic = False

    
    # ------------
    # data
    # ------------
    ## set up the data
    ## initialize a dataloader 
    data_folder = "/home/engs2522/project/multi-modal-heart/multi_modal_heart/data/ptbxl/"
    train_data_statement_path = os.path.join(data_folder,"/home/engs2522/project/multi-modal-heart/multi_modal_heart/data/ptbxl/high_quality_split/Y_train.csv")
    validate_data_statement_path = os.path.join(data_folder,"/home/engs2522/project/multi-modal-heart/multi_modal_heart/data/ptbxl/high_quality_split/Y_validate.csv")
    test_data_statement_path = os.path.join(data_folder,"/home/engs2522/project/multi-modal-heart/multi_modal_heart/data/ptbxl/high_quality_split/Y_test.csv")

    ## original test loader
    all_test_data_statement_path = os.path.join(data_folder,"/home/engs2522/project/multi-modal-heart/multi_modal_heart/data/ptbxl/raw_split/Y_test.csv")
 

    data_loaders = []
    n_leads = 8 if "leads=8" in args.ae_type else 12
    batch_size  = 128
    if args.use_median_wave:
        max_seq_len = 608
        sampling_rate=500
    else:
        sampling_rate=100
        max_seq_len = 1024 ## using 1024 for the 12 lead ECG, padding with 0 
    max_epochs= 300
    data_proc_config={
                    "if_clean":False,
                    }
    data_aug_config={
                    "noise_frequency_list":[0],
                    "noise_amplitude_range":[0.,0.],
                    "powerline_frequency_list":[0,0.],
                    "powerline_amplitude_range":[0.,0.],
                    "artifacts_amplitude_range":[0.,0.],
                    "artifacts_number_range":[0,0],
                    "linear_drift_range":[0.,0.],
                    "artifacts_frequency_list":[5],
                    "random_prob":0, ## data augmentation prob
                    "if_mask_signal":True, ## change it to True
                    "mask_whole_lead_prob":0.5,
                    "lead_mask_prob":0, ## mask certain parts of the lead
                    "region_mask_prob":0,
                    "mask_length_range":[0.08, 0.18],
                    "mask_value":0.0,
                    "random_drop_half_prob":0,
                    }
    if args.use_median_wave: 
        data_aug_config["region_mask_prob"]=0
    if args.mask_half:
        data_aug_config["random_drop_half_prob"]=0.5
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
    if ae_type =="dae":
        ecg_net= ECGAE(in_channels=n_leads,ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=512,
                    add_time=False,
                    encoder_mha = False,
                    apply_method="",
                    decoder_outdim=n_leads,
                    time_dim=4)
    elif ae_type=="dae_64_groupconv":
         ecg_net= ECGAE(in_channels=n_leads,ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=64,
                    add_time=False,
                    encoder_mha = False,
                    apply_method="",
                    decoder_outdim=n_leads,
                    time_dim=4, groups=n_leads)

    elif ae_type=="dae_64" or ae_type=="dae_64_grad_clip" or  ae_type=="dae_64_identity" or ae_type=="dae_64_shuffle_lead" or ae_type=="dae_64_LeadCorrLoss":
        if "use_template" in ae_type:
            in_channels = n_leads+36
        else: in_channels = n_leads
        ecg_net= ECGAE(in_channels=in_channels,ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=64,
                    add_time=False,
                    encoder_mha = False,
                    apply_method="",
                    decoder_outdim=n_leads,
                    time_dim=4)
    elif ae_type=="dae_64_ELU":
         ecg_net= ECGAE(in_channels=n_leads,ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=64,
                    add_time=False,
                    encoder_mha = False,
                    apply_method="",
                    decoder_outdim=n_leads,
                    act = nn.ELU(),
                    time_dim=4,if_VAE=False)
    elif ae_type=="dae_64+VAE":
         ecg_net= ECGAE(in_channels=n_leads,ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=64,
                    add_time=False,
                    encoder_mha = False,
                    apply_method="",
                    decoder_outdim=n_leads,
                    time_dim=4,if_VAE=True)

    elif ae_type=="resnet1d101_64":
        ecg_net= ECGAE(encoder_type="resnet1d101",in_channels=n_leads,ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=64,
                    add_time=False,
                    encoder_mha = False,
                    apply_method="",
                    decoder_outdim=n_leads,
                    time_dim=4)
        latent_code_dim = 64
    elif ae_type=="resnet1d101_512+benchmark_classifier":
        ecg_net= ECGAE(encoder_type="resnet1d101",in_channels=n_leads,ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=512,
                    add_time=False,
                    encoder_mha = False,
                    apply_method="",
                    decoder_outdim=n_leads)
        latent_code_dim = 512
    elif ae_type=="resnet1d101_512+group_conv+benchmark_classifier":
        ecg_net= ECGAE(encoder_type="resnet1d101",in_channels=n_leads,ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=512,
                    add_time=False,
                    groups=n_leads,
                    encoder_mha = False,
                    apply_method="",
                    decoder_outdim=n_leads)
        latent_code_dim = 512
    elif ae_type=="resnet1d101_512+group_conv+mha+benchmark_classifier":
        ecg_net= ECGAE(encoder_type="resnet1d101",in_channels=n_leads,ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=512,
                    add_time=False,
                    encoder_mha = True,
                    groups=n_leads,
                    apply_method="",
                    decoder_outdim=n_leads)
        latent_code_dim = 512
    elif ae_type=="resnet1d101_512+mha+benchmark_classifier":
        ecg_net= ECGAE(encoder_type="resnet1d101",in_channels=n_leads,ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=512,
                    add_time=False,
                    encoder_mha = True,
                    groups=1,
                    apply_method="",
                    decoder_outdim=n_leads)
        latent_code_dim = 512
    elif ae_type=="marcelVAE_512" or  ae_type=="marcelVAE_512_raw" :
        ecg_net = ECGMarcelVAE(num_leads=n_leads,time_steps=max_seq_len,z_dims=512) ## for ease of comparison
        latent_code_dim = 512
    elif ae_type=="marcelVAE_64" or ae_type=="marcelVAE_64_identity":
        ecg_net = ECGMarcelVAE(num_leads=n_leads,time_steps=max_seq_len,z_dims=64) ## for ease of comparison
        args.decoder_type=""
    elif ae_type=="marcelVAE_16_identity" or ae_type=="marcelVAE_16":
        ecg_net = ECGMarcelVAE(num_leads=n_leads,time_steps=max_seq_len,z_dims=16) ## marcel's 32
        args.decoder_type=""
    elif ae_type=="LeiLSTM_VAE_16":
        ecg_net = ECGLSTMnet(n_lead=n_leads,num_length=max_seq_len,z_dims=16,out_ch=n_leads) ## lei's work
        args.decoder_type=""
    elif ae_type=="LeiLSTM_VAE_64":
        ecg_net = ECGLSTMnet(n_lead=n_leads,num_length=max_seq_len,z_dims=64,out_ch=n_leads) ## lei's work
        args.decoder_type=""
    elif ae_type=="ECG_transformer":
        print ("ECG_transformer")
        ecg_net = ECG_transformer(n_lead = n_leads,ECG_length=max_seq_len)
    elif ae_type=="ECG_attention_64" or ae_type=="ECG_attention_64_raw":
        ecg_net = ECGAttentionAE(num_leads=n_leads, time_steps=max_seq_len, z_dims=64,downsample_factor=5, base_feature_dim=4,if_VAE=False,use_attention_pool=True)
    elif ae_type=="ECG_attention_512_raw" or ae_type=="ECG_attention_512_finetuned":
        ecg_net = ECGAttentionAE(num_leads=n_leads, time_steps=max_seq_len, z_dims=512, linear_out=64, downsample_factor=5, base_feature_dim=4,if_VAE=False,use_attention_pool=True)
        latent_code_dim = 512
    elif ae_type=="ECG_attention_512_raw_no_attention_pool" or ae_type=="ECG_attention_512_finetuned_no_attention_pool":
        ecg_net = ECGAttentionAE(num_leads=n_leads, time_steps=max_seq_len, z_dims=512, linear_out=64, downsample_factor=5, base_feature_dim=4,if_VAE=False,use_attention_pool=False)
        latent_code_dim = 512
    ## ablation study: 
    elif ae_type=="ECG_attention_512_raw_no_attention_pool_no_linear_abl_no_lead_attention":
        ecg_net = ECGAttentionAE(num_leads=n_leads, time_steps=max_seq_len, z_dims=512, linear_out=512, downsample_factor=5, base_feature_dim=4,if_VAE=False,use_attention_pool=False, no_lead_attention=True,no_linear_in_E=True)
        latent_code_dim = 512
    elif ae_type=="ECG_attention_512_raw_no_attention_pool_no_linear_abl_no_time_attention":
        ecg_net = ECGAttentionAE(num_leads=n_leads, time_steps=max_seq_len, z_dims=512, linear_out=512, downsample_factor=5, base_feature_dim=4,if_VAE=False,use_attention_pool=False, no_time_attention=True,
        no_linear_in_E=True,apply_lead_mask=True)
        latent_code_dim = 512
    elif ae_type=="ECG_attention_512_raw_no_attention_pool_no_linear_abl_no_lead_time_attention":
        ecg_net = ECGAttentionAE(num_leads=n_leads, time_steps=max_seq_len, z_dims=512, linear_out=512, downsample_factor=5, base_feature_dim=4,if_VAE=False,use_attention_pool=False, no_time_attention=True, no_lead_attention=True,no_linear_in_E=True)
        latent_code_dim = 512
    ## test feature: add mask mechanism in lead attention:
    elif ae_type =="ECG_attention_512_raw_no_attention_pool_no_linear_add_mask_lead_attention":
        ecg_net = ECGAttentionAE(num_leads=n_leads, time_steps=max_seq_len, z_dims=512, linear_out=512, downsample_factor=5, 
                                 base_feature_dim=4,if_VAE=False,use_attention_pool=False,no_linear_in_E=True,
                                 apply_lead_mask=True)

        latent_code_dim = 512
    
    ## test feature: add subsequent attention function, first time attention and then cross-lead attention
    elif ae_type=="ECG_attention_512_raw_no_attention_pool_no_linear_add_subsequent_attention":
        ecg_net = ECGAttentionAE(num_leads=n_leads, time_steps=max_seq_len, z_dims=512, linear_out=512, downsample_factor=5, base_feature_dim=4,if_VAE=False,
                                 use_attention_pool=False, apply_subsequent_attention=True,no_linear_in_E=True)
        latent_code_dim = 512
    elif ae_type=="ECG_attention_512_raw_no_attention_pool_no_linear" or ae_type=="ECG_attention_512_finetuned_no_attention_pool_no_linear":
        ## remove last linear layers: 512->512
        print ("!!!!!!!!!!!!!!!!!!!!!!this is the best")
        ecg_net = ECGAttentionAE(num_leads=n_leads, time_steps=max_seq_len, z_dims=512, linear_out=512, downsample_factor=5, base_feature_dim=4,if_VAE=False,use_attention_pool=False,no_linear_in_E=True)
        latent_code_dim = 512
    elif ae_type=="ECG_attention_512_raw_no_attention_pool_no_linear_downsample_4":
        print ("------downsample 4------max seq len: ",max_seq_len)
        ecg_net = ECGAttentionAE(num_leads=n_leads, time_steps=max_seq_len, z_dims=512, linear_out=512, downsample_factor=4,upsample_factor=4, base_feature_dim=4,if_VAE=False,use_attention_pool=False,no_linear_in_E=True)
        latent_code_dim = 512
    elif ae_type=="ECG_attention_512_raw_no_attention_pool_no_linear_downsample_3":
        print ("------downsample 3------max seq len: ",max_seq_len)
        ecg_net = ECGAttentionAE(num_leads=n_leads, time_steps=max_seq_len, z_dims=512, linear_out=512, downsample_factor=3,upsample_factor=3, base_feature_dim=4,if_VAE=False,use_attention_pool=False,no_linear_in_E=True)
        latent_code_dim = 512
    elif ae_type=="ECG_attention_512_raw_no_attention_pool_no_linear_downsample_2":
        print ("------downsample 2------max seq len: ",max_seq_len)
        ecg_net = ECGAttentionAE(num_leads=n_leads, time_steps=max_seq_len, z_dims=512, linear_out=512, downsample_factor=2,upsample_factor=2, base_feature_dim=4,if_VAE=False,use_attention_pool=False,no_linear_in_E=True)
        latent_code_dim = 512

    elif ae_type=="ECG_attention_512_raw_no_attention_pool_no_linear_add_time":
        ecg_net = ECGAttentionAE(num_leads=n_leads, time_steps=max_seq_len, z_dims=512, linear_out=512, downsample_factor=5, base_feature_dim=4,if_VAE=False,use_attention_pool=False,
                                 no_linear_in_E=True, add_time=True)
        latent_code_dim = 512
    elif ae_type =="ECG_attention_64_linear1":
        ecg_net = ECGAttentionAE(num_leads=n_leads, time_steps=max_seq_len, z_dims=64,downsample_factor=5, base_feature_dim=4,if_VAE=False, num_linear_in_D=1)
    elif ae_type=="ECG_attention_64+VAE":
        ecg_net = ECGAttentionAE(num_leads=n_leads, time_steps=max_seq_len, z_dims=64,downsample_factor=5, base_feature_dim=4,if_VAE=True)
    elif ae_type =="dae_64+mha":
        ecg_net= ECGAE(in_channels=n_leads,ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=64,
                    add_time=False,
                    encoder_mha = True,
                    apply_method="",
                    decoder_outdim=n_leads,
                    time_dim=4)
    elif (ae_type=='dae_512+mha+group_conv+benchmark_classifier'):
        latent_code_dim=512 ## average+mha feature together with classifier
        ecg_net= ECGAE(in_channels=n_leads,
                    ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=256,
                    groups=n_leads,
                    add_time=False,
                    encoder_mha = True,
                    apply_method="",
                    decoder_outdim=n_leads,
                    if_VAE=False,
                    time_dim=4)

    elif (ae_type =="dae_64+mha+group_conv") or (ae_type =="dae_64+mha+group_conv_raw") or (ae_type=="dae_64+mha+group_conv_identity") or (ae_type=="dae_64+mha+group_conv+snr") or (
          ae_type=="dae_64+mha+group_conv+LeadCorrLoss") or (ae_type=="dae_64+mha+group_conv+norm_output") or (
            ae_type=="dae_64+mha+group_conv+leads=8") or (ae_type=="dae_64+mha+group_conv+VAE_warm_up") or (
        ae_type=="dae_256+mha+group_conv") or ae_type==("dae_256+mha+group_conv_raw") or (ae_type=="dae_512+mha+group_conv") or (ae_type=="dae_512+mha+group_conv_raw"):
        if "leads=8" in ae_type:
            n_leads = 8
        if "VAE" in ae_type:
            if_VAE=True
        else: if_VAE=False

        if "64" in ae_type:
            latent_code_dim=64
        elif "256" in ae_type:
            latent_code_dim=256
        elif "512" in ae_type:
            latent_code_dim=512
        else:
            latent_code_dim=64
        print ("n leads: ",n_leads)
        print ("latent_code_dim: ",latent_code_dim)
        ecg_net= ECGAE(in_channels=n_leads,
                    ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=latent_code_dim,
                    groups=n_leads,
                    add_time=False,
                    encoder_mha = True,
                    apply_method="",
                    decoder_outdim=n_leads,
                    if_VAE=if_VAE,
                    time_dim=4)
   
    elif ae_type =="dae_64+mha+layer_norm_w_o_group_conv":
        ecg_net= ECGAE(in_channels=n_leads,encoder_type="ms_resnet_mlp_0",
                    ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=64,
                    add_time=False,
                    encoder_mha = True,
                    apply_method="",
                    decoder_outdim=n_leads,
                    time_dim=4)
    elif ae_type =="dae_64+mha+time_FPE" or  ae_type =="dae_64+mha+time_FPE_new" or ae_type == "dae_64+mha+time_FPE_groupconv":
        if "groupconv" in ae_type:
            groups = n_leads
        else: groups = 1
        ecg_net= ECGAE(in_channels=n_leads,ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=64,
                    add_time=True,
                    encoder_mha = True,
                    apply_method="fixed_positional_encoding",
                    decoder_outdim=n_leads,
                    time_dim=4,
                    groups=groups)
    elif ae_type =="dae_64+mha+time2vec":
         ## need to check the time encoding when whole lead masking is applied
         ecg_net= ECGAE(in_channels=n_leads,ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=64,
                    add_time=True,
                    encoder_mha = True,
                    apply_method="time2vec",
                    decoder_outdim=n_leads,
                    time_dim=5)
    elif ae_type =="dae_64+mha+FPE2vec":
         ecg_net= ECGAE(in_channels=n_leads,ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=64,
                    add_time=True,
                    encoder_mha = True,
                    apply_method="FPE2vec",
                    decoder_outdim=n_leads,
                    time_dim=5)

    elif ae_type =="dae_64+mha+time":
         ## need to check the time encoding when whole lead masking is applied
         ecg_net= ECGAE(in_channels=n_leads,ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=64,
                    add_time=True,
                    encoder_mha = True,
                    apply_method="",
                    decoder_outdim=n_leads,
                    time_dim=4)
    elif ae_type =="dae_64+mha+VAE":
         ecg_net= ECGAE(in_channels=n_leads,ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=64,
                    add_time=False,
                    encoder_mha = True,
                    apply_method="",
                    decoder_outdim=n_leads,
                    time_dim=1, if_VAE=True)
    elif ae_type =="dae_64+mha+LPE":
         ## need to check the time encoding when whole lead masking is applied
         ecg_net= ECGAE(in_channels=n_leads,ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=64,
                    add_time=True,
                    encoder_mha = True,
                    apply_method="LPE",
                    decoder_outdim=n_leads,
                    time_dim=4)
    elif ae_type =="dae+mha+time2vec+predict_feature":
         ecg_net= ECGAE(in_channels=n_leads,ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=64,
                    add_time=True,
                    encoder_mha = True,
                    apply_method="time2vec",
                    decoder_outdim=24,
                    time_dim=4)

    elif ae_type =="double_dae_64":
         ecg_net= doubleECGNet(in_channels=n_leads,ECG_length=max_seq_len,decoder_type=args.decoder_type,
                    embedding_dim=256,latent_code_dim=64,
                    add_time=False,
                    encoder_mha = False,
                    apply_method="",
                    decoder_outdim=n_leads,
                    time_dim=4)
    else:
        raise NotImplementedError
    log_fix = "log_median" if args.use_median_wave else "log"
    log_name= ae_type+"_"+args.decoder_type
    log_name = log_name+"_ECG2Text" if args.ECG2Text else log_name
    if args.mask_half:
        log_name = log_name+"_mask_half"
    task_name = ae_type
    if not args.no_pretrain: max_iters= max_epochs * len(data_loaders["train"])
    else: max_iters = 10

    latent_code_dim = 64 if "64" in task_name else latent_code_dim
    autoencoder = LitAutoEncoder(ecg_net,task_name=ae_type,grad_clip = args.grad_clip, 
                                 max_iters=max_iters,
                                 batch_size=batch_size,args=args)

    ## load all data
    data_loaders=[]
    data_folder = "/home/engs2522/project/multi-modal-heart/multi_modal_heart/data/ptbxl/"
    train_data_statement_path = os.path.join(data_folder,"/home/engs2522/project/multi-modal-heart/multi_modal_heart/data/ptbxl/raw_split/Y_train.csv")
    validate_data_statement_path = os.path.join(data_folder,"/home/engs2522/project/multi-modal-heart/multi_modal_heart/data/ptbxl/raw_split/Y_validate.csv")
    test_data_statement_path = os.path.join(data_folder,"/home/engs2522/project/multi-modal-heart/multi_modal_heart/data/ptbxl/raw_split/Y_test.csv")

    for label_csv_path in [train_data_statement_path,validate_data_statement_path,test_data_statement_path]:
        if_train ="train" in label_csv_path.split("/")[-1]
        dataset = ECGDataset(data_folder,label_csv_path=label_csv_path,
                            sampling_rate=sampling_rate,
                            use_median_wave=args.use_median_wave,
                            max_seq_len=max_seq_len,
                            augmentation= if_train,
                            data_proc_config=data_proc_config,
                            data_aug_config=data_aug_config,)
        data_loader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                num_workers=4,
                                shuffle = if_train,
                                drop_last= if_train,
                                )
        print ('load {} data: {} samples'.format(label_csv_path.split("/")[-1],len(dataset)))
        data_loaders.append(data_loader)
    finetune_max_epochs = 50
    finetune_task_name = args.finetune_taskname if "benchmark_classifier" not in ae_type else "benchmark_classifier"
    freeze_encoder = False
    gradual_unfreeze=args.gradual_unfreeze
    tb_logger = TensorBoardLogger( f"./{log_fix}_finetune", name=log_name, version="")
    output_dir  = os.path.join(tb_logger.log_dir,"classfication_result")
    os.makedirs(output_dir,exist_ok=True)
    train_loader, validate_loader, test_loader = data_loaders[0],data_loaders[1],data_loaders[2]
    encoder = autoencoder.network.encoder
    decoder = autoencoder.network.decoder
    classifier = LitClassifier(encoder,decoder, input_dim = latent_code_dim, num_classes=5,learning_rate=1e-2,encoder_lr=args.encoder_lr,
                                freeze_encoder=freeze_encoder, task_name = finetune_task_name,
                                max_iters=finetune_max_epochs * len(train_loader),
                                output_dir=output_dir,gradual_unfreeze=gradual_unfreeze,args=args,
                                ecg_projector=autoencoder.ecg_projector,text_projector=autoencoder.text_projector)
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor=f"{finetune_task_name}/val_macro_auc",
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode='max'
    )  
    checkpoint_dir  = os.path.join(tb_logger.log_dir,"checkpoints")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, 
                                                        filename='{epoch:02d}-val_auroc:{'+finetune_task_name+'/val_macro_auc:.2f}',
                                                        save_top_k=2, monitor=f"{finetune_task_name}/val_macro_auc"
                                                        , mode='max',save_last=True)
    checkpoint_callback_best_loss_min = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, 
                                                        filename='checkpoint_best_loss',
                                                        save_top_k=1, monitor=f"{finetune_task_name}/val_loss"
                                                        , mode='min',save_last=False)


    callbacks=[
        FineTuneLearningRateFinder(milestones=[0],min_lr=1e-5, max_lr=1e-2, 
                                    mode='exponential', early_stop_threshold=4.0),
        # LearningRateMonitor(logging_interval='step'),
        # OneCycleLR(max_lr=0.01, total_steps=finetune_max_epochs * len(train_loader), pct_start=0.3, anneal_strategy='cos'),
        # early_stop_callback, ## stop early stopping
        checkpoint_callback,
        checkpoint_callback_best_loss_min
        ]


    finetuner = pl.Trainer(accelerator="gpu",
                        devices=1, max_epochs=finetune_max_epochs,fast_dev_run=args.debug,
                        logger=tb_logger,log_every_n_steps=1,check_val_every_n_epoch = 1,
                        callbacks=callbacks,
                        ) 
                    
    finetuner.fit(model = classifier,train_dataloaders=train_loader,val_dataloaders=validate_loader)
    ## evaluate the model
    result = finetuner.test(classifier,test_loader,ckpt_path="best")
    ## save result to csv
    df = pd.DataFrame(result)
    ## df round to 4 decimal places
    df = df.round(4)
    df.to_csv(os.path.join(tb_logger.log_dir,"result.csv"))
if __name__=="__main__":
    cli_main()