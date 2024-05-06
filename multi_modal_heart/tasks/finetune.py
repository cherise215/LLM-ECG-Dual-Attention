
## finetuning on classification tasks
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
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from lightning.pytorch.callbacks import EarlyStopping

sys.path.append("./")
from multi_modal_heart.ECG.ecg_dataset import ECGDataset
from multi_modal_heart.model.ecg_net import ECGAE
from multi_modal_heart.ECG.ecg_utils import plot_multiframe_in_one_figure,arraytodataframe
from multi_modal_heart.model.ecg_net import ClassifierMLP
## this file performs ECG pretraining
class LitClassifier(pl.LightningModule):
    def __init__(self, feature_extractor,task_name, linear_evaluation=True,latent_code_dim=None):
        super().__init__()
        self.feature_extractor=feature_extractor
        if linear_evaluation:
            ## freeze the feature extractor:
            for p in self.feature_extractor.parameters():
                p.requires_grad=False

        self.task_name = self.task_name
        self.save_hyperparameters()
        self.optim_lr = 0.001
        self.wd = 1e-4
        if self.task_name =="super_class":
            self.y_key_name = "super_class"
            self.num_classes = 5
        elif self.task_name=="challenge_class":
            self.y_key_name="challenge_class"
            self.num_classes = 26
            self.weights = weights
            self.loss_fn = torch.nn.BCEWithLogitsLoss(weights)
            self.optim_lr = 0.001
            self.wd = 1e-4

        
        if latent_code_dim is None:
            try:
                latent_code_dim = self.feature_extractor.latent_code_dim
            except:
                raise "latent code dim must be specified, this is the feature dimension from the feature extractor"
        mlp_dim  = latent_code_dim//2 if latent_code_dim//2>self.num_classes else self.num_classes*2
        downsteam_net = ClassifierMLP(input_size=latent_code_dim,hidden_sizes=[mlp_dim,mlp_dim],output_size=self.num_classes)
        self.net = nn.Sequential(
            self.feature_extractor,
            downsteam_net
        )
        
        # ## specify loss
        # if(self.loss_type == "binary_cross_entropy"):
        #     self.loss_fn = F.binary_cross_entropy_with_logits
        # elif(self.loss_type == "cross_entropy"):
        #     self.loss_fn = F.cross_entropy
        # elif(self.loss_type == "mse"):
        #     self.loss = F.mse
        # elif(self.loss_type == "nll_regression"):
        #     self.loss = F.nll_regression    
        # else:
        #     print("loss not found")
        #     assert(True)   
        



    def run_task(self, batch,batch_idx, if_train=True):
         # training_step defines the train loop.
        # it is independent of forward
        x= batch["input_seq"]
        y = batch[self.y_key_name]
       
        y_pred = self.loss_fn(y,y_pred) 
        self.y_pred = y_pred
        self.y = y
        return x_hat,loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        y_pred, loss = self.run_task(batch,batch_idx,)
        self.log(f"{self.task_name}/train_loss", loss)
        return loss

    def configure_optimizers(self):
        # Make sure to filter the parameters based on `requires_grad`
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr= self.optim_lr,weight_decay=self.wd)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return optimizer,scheduler

    def validation_step(self, batch, batch_idx):
        y_pred, loss = self.run_task(batch,batch_idx,)
        self.log(f"{self.task_name}/val_loss", loss)

    def test_step(self, batch, batch_idx, dataloader_idx):
        y_pred, loss = self.run_task(batch,batch_idx)
        if dataloader_idx==0: self.log(f"{self.task_name}/test_loss", loss)  
        elif  dataloader_idx==1: self.log(f"{self.task_name}/all_test_loss", loss)
        else: raise NotImplementedError
    


    def on_train_epoch_end(self):
        ##  log the figure of the reference signal and reconstructed signal
        ## 12*L
        if self.current_epoch==0 or self.current_epoch%10==0:
            self.plot_recon_ECG(title="train/ECG_recon")
    def on_validation_epoch_end(self):
        if self.current_epoch==0 or self.current_epoch%10==0:
            self.plot_recon_ECG(title="val/ECG_recon")
    def on_test_epoch_end(self):
        if self.current_epoch==0 or self.current_epoch%10==0:
            self.plot_recon_ECG(title="test/ECG_recon")
        
    def plot_recon_ECG(self, title="train/ECG_recon"):
        sample_y_nd  =self.y.data.detach().cpu().numpy()[0]
        sample_x_hat_nd  =self.x_hat.data.detach().cpu().numpy()[0]
        y_df = arraytodataframe(sample_y_nd)
        y_recon = arraytodataframe(sample_x_hat_nd)

        y_df.columns = ["GT "+k for k in ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']]
        y_recon.columns = ["Pred. "+k for k in ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']]

        figure = plot_multiframe_in_one_figure([y_df,y_recon],figsize=(15,4), figure_arrangement=(4,3), logger=self.logger,epoch=self.current_epoch, title=title)
        return figure



def cli_main():
    parser = ArgumentParser()
    # Trainer arguments
    parser.add_argument("--device", type=int, default=1)
    # Hyperparameters for the model
    parser.add_argument("--ae_type", type=str, default="dae")
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--checkpoint_path",  type=str, default="/home/engs2522/project/multi-modal-heart/log/dae/checkpoints/epoch=199-step=13800.ckpt")
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)

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
    sampling_rate=100
    batch_size  = 128
    max_seq_len = 1024
    max_epochs= 300
    data_proc_config={
                    "if_clean":True,
                    }
    data_aug_config={
                  "noise_frequency_list":[5],
                    "noise_amplitude_range":[0.,0.2],
                    "powerline_frequency_list":[5,10],
                    "powerline_amplitude_range":[0.,0.05],
                    "artifacts_amplitude_range":[0.,0.1],
                    "artifacts_number_range":[0,3],
                    "linear_drift_range":[0.,0.3],
                    "random_prob":0.5,
                    "if_mask_signal":False,
                    "mask_whole_lead_prob":0.1,
                    "lead_mask_prob":0.2,
                    "region_mask_prob":0.15,
                    "mask_length_range":[0.08, 0.18],
                    "mask_value":0.0,
                    "artifacts_frequency_list":[5],
                    }
    for label_csv_path in [train_data_statement_path,validate_data_statement_path,test_data_statement_path,all_test_data_statement_path]:
        if_test ="test" in label_csv_path.split("/")[-1]
        if_train ="train" in label_csv_path.split("/")[-1]
        dataset = ECGDataset(data_folder,label_csv_path=label_csv_path,
                            sampling_rate=sampling_rate,
                            max_seq_len=max_seq_len,
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
        data_loaders.append(data_loader)
        
    train_loader, validate_loader, test_loader, all_test_loader= data_loaders[0],data_loaders[1],data_loaders[2],data_loaders[3]
    # ------------
    # model
    # ------------
    ## set up the model
    ae_type = args.ae_type
    if ae_type =="dae":
        ecg_net= ECGAE(in_channels=12,ECG_length=max_seq_len,
                    embedding_dim=256,latent_code_dim=512,
                    add_time=False,
                    encoder_mha = False,
                    apply_method="",
                    decoder_outdim=12,
                    time_dim=4)
    elif ae_type=="dae_64":
        ecg_net= ECGAE(in_channels=12,ECG_length=max_seq_len,
                    embedding_dim=256,latent_code_dim=64,
                    add_time=False,
                    encoder_mha = False,
                    apply_method="",
                    decoder_outdim=12,
                    time_dim=4)
    elif ae_type=="resnet1d101_64":
         ecg_net= ECGAE(encoder_type="resnet1d101",in_channels=12,ECG_length=max_seq_len,
                    embedding_dim=256,latent_code_dim=64,
                    add_time=False,
                    encoder_mha = False,
                    apply_method="",
                    decoder_outdim=12,
                    time_dim=4)
    elif ae_type =="dae+mha":
        ecg_net= ECGAE(in_channels=12,ECG_length=max_seq_len,
                    embedding_dim=256,latent_code_dim=512,
                    add_time=False,
                    encoder_mha = True,
                    apply_method="time2vec",
                    decoder_outdim=12,
                    time_dim=4)
    elif ae_type =="dae+mha+time2vec":
         ## need to check the time encoding when whole lead masking is applied
         ecg_net= ECGAE(in_channels=12,ECG_length=max_seq_len,
                    embedding_dim=256,latent_code_dim=512,
                    add_time=True,
                    encoder_mha = True,
                    apply_method="time2vec",
                    decoder_outdim=12,
                    time_dim=4)
    elif ae_type =="dae+mha+time":
         ## need to check the time encoding when whole lead masking is applied
         ecg_net= ECGAE(in_channels=12,ECG_length=max_seq_len,
                    embedding_dim=256,latent_code_dim=512,
                    add_time=True,
                    encoder_mha = True,
                    apply_method="",
                    decoder_outdim=12,
                    time_dim=4)
    elif ae_type =="dae+mha+time2vec+predict_feature":
         ecg_net= ECGAE(in_channels=12,ECG_length=max_seq_len,
                    embedding_dim=256,latent_code_dim=512,
                    add_time=True,
                    encoder_mha = True,
                    apply_method="time2vec",
                    decoder_outdim=24,
                    time_dim=4)

    elif ae_type =="dae+to":
        pass
    else:
        raise NotImplementedError

    log_name= ae_type
    task_name = ae_type
    autoencoder = LitAutoEncoder(ecg_net,task_name=ae_type)
    if args.pretrained and args.checkpoint_path !="" and os.path.exists(args.checkpoint_path):
        autoencoder = LitAutoEncoder.load_from_checkpoint(args.checkpoint_path)
        print ("auto encoder is loaded from {}".format(args.checkpoint_path))
    else:
        print ("auto encoder is randomly initialized")
    # ------------
    # training
    # ------------
    tb_logger = TensorBoardLogger( "./log", name=log_name, version="")
    checkpoint_callback = ModelCheckpoint(dirpath=f"./log/{log_name}/checkpoint", save_top_k=2, monitor="val_loss")
    early_stopping = EarlyStopping('val_loss')
    trainer = pl.Trainer(deterministic=deterministic,accelerator="gpu",benchmark=not deterministic, 
                         devices=args.device,limit_train_batches=100, max_epochs=max_epochs,
                         logger=tb_logger,log_every_n_steps=50,callbacks=[early_stopping]
                         ) 

                        
    if not args.test_only:
       trainer.fit(autoencoder, train_loader,validate_loader)
    
    # ------------
    # testing
    # ------------
    ## evaluation 
    result = trainer.test(autoencoder,[test_loader,all_test_loader])
    print (result)

    # ------------
    # start finetuning
    # ------------
 

if __name__=="__main__":
    cli_main()