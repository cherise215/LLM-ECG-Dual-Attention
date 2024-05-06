# 1. load model
# 2. load test data 
# 3. get prediction along with attention map
# 4. for predictions with correctly predicted ones, get an averaged attention map for each disease class
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.classification import MultilabelAUROC
import pytorch_lightning as pl
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('../..')
from multi_modal_heart.model.ecg_net_attention import ECGEncoder,ECGAttentionAE
from multi_modal_heart.model.ecg_net import ECGAE
from multi_modal_heart.model.ecg_net import BenchmarkClassifier
from multi_modal_heart.ECG.ecg_dataset import ECGDataset

time_steps= 608
use_median_wave = True
sampling_rate=500
checkpoint_path = "../../log_median_finetune/ECG_attention_512_finetuned_no_attention_pool_no_linear_ms_resnet/checkpoints/checkpoint_best_loss.ckpt"


class LitClassifier(pl.LightningModule):
    def __init__(self,encoder,input_dim,num_classes=5):
        super().__init__()
        
        self.encoder = encoder
        #### add classifier if use benchmark classifier
        self.downsteam_net = BenchmarkClassifier(input_size=input_dim,hidden_size=128,output_size=num_classes)
    def forward(self, x, mask):
        latent_code = self.encoder.get_features_after_pooling(x,mask)
        return self.downsteam_net(latent_code)
    
def print_result(probs,super_classes_labels, topk=1):
    probs, label_indices = torch.topk(probs, topk)
    probs = probs.tolist()
    label_indices = label_indices.tolist()
    for prob, idx in zip(probs, label_indices):
        label = super_classes_labels[idx]
        print(f'{label} ({idx}):', round(prob, 4))
def calc_hamming_score(y_true, y_pred):
    return (
        (y_true & y_pred).sum(axis=1) / (y_true | y_pred).sum(axis=1)
    ).mean()    
ecg_net = ECGAttentionAE(num_leads=12, time_steps=time_steps, z_dims=512, linear_out=512, downsample_factor=5, base_feature_dim=4,if_VAE=False,
                         use_attention_pool=False,no_linear_in_E=True, apply_lead_mask=False, no_time_attention=False)
classification_net = LitClassifier(encoder=ecg_net.encoder,input_dim=512,num_classes=5)
# checkpoint_path  ="../../log_finetune/ECG_attention_512_raw_no_attention_pool_no_linear_abl_no_time_attention_ms_resnet/checkpoints/last-v3.ckpt"
# checkpoint_path  ="../../log_finetune/ECG_attention_512_raw_no_attention_pool_no_linear_ms_resnet_ECG2Text/checkpoints/last-v5.ckpt"
# print (torch.load(checkpoint_path)["state_dict"].keys())
mm_checkpoint = torch.load(checkpoint_path)["state_dict"]
encoder_params = {(".").join(key.split(".")[1:]):value for key, value in mm_checkpoint.items() if str(key).startswith("encoder")}
classification_params = {(".").join(key.split(".")[1:]):value for key, value in mm_checkpoint.items() if str(key).startswith("downsteam_net")}
classification_net.encoder.load_state_dict(encoder_params)
classification_net.downsteam_net.load_state_dict(classification_params)


## initialize a dataloader (all data)
data_folder = "/home/engs2522/project/multi-modal-heart/multi_modal_heart/data/ptbxl/"
test_data_statement_path = os.path.join(data_folder,
                                        "/home/engs2522/project/multi-modal-heart/multi_modal_heart/data/ptbxl/raw_split/Y_test.csv")


data_loaders = []
sampling_rate=sampling_rate
batch_size  = 1
max_seq_len = time_steps
data_proc_config={
                "if_clean":False,
                 }
data_aug_config={
                "noise_frequency_list":[5,20,100,150,175],
                "noise_amplitude_range":[0.,0.2],
                "powerline_frequency_list":[50],
                "powerline_amplitude_range":[0.,0.05],
                "artifacts_amplitude_range":[0.,0.1],
                "artifacts_frequency_list":[5,10],
                "artifacts_number_range":[0,3],
                "linear_drift_range":[0.,0.1],
                "random_prob":0.,
                "if_mask_signal":True, 
                "mask_whole_lead_prob":0.5,
                "lead_mask_prob":0.,
                "region_mask_prob":0.,
                "mask_length_range":[0., 0.],
                "mask_value":0.0,
                
}
dataset = ECGDataset(data_folder,label_csv_path=test_data_statement_path,
                         use_median_wave=use_median_wave, ## set to median wave, then it has 600 samples for each lead, when sampling rate is 100
                          sampling_rate=sampling_rate,
                          max_seq_len=max_seq_len,
                          augmentation= False,
                          data_proc_config=data_proc_config,
                          data_aug_config=data_aug_config,)
data_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=0,
                            shuffle = True,
                            drop_last= False,
                            )
print ('load {} data: {} samples'.format(test_data_statement_path.split("/")[-1],len(dataset)))
    
dataset.super_class_label_converter.inverse_transform(np.array([[0,0,0,1,1]]))

 
calc_hamming_score(np.array([[0,0,1,1,0]]),np.array([[0,0,1,1,0]]))
dataset.super_classes_labels



metric = MultilabelAccuracy(num_labels=5)
auroc = MultilabelAUROC(num_labels=5)
classification_net.eval()
classification_net.freeze()

label_attention_dict = {}
for i, data in enumerate(tqdm(data_loader)):
    ## batch size ==1
    torch.cuda.empty_cache()
    # ecg_id= data['ecg_id']
    input_data=data["input_seq"]
    print (input_data.shape)
    mask =  data["mask"]
    # ## remove lead 3- data
    # input_data[:,3:] = torch.zeros_like(input_data[:,3:])
    # mask[:,3:] = torch.zeros_like(mask[:,3:])
    with torch.inference_mode():
        pred = classification_net(input_data,mask)
    report=data["report"]
    pred = torch.sigmoid(pred)
    ground_truth = data["super_class_encoding"]
    # print (ground_truth.shape)
    acc = metric.update(pred, ground_truth.long())
    # print('pred:',torch.ones_like(ground_truth)*(pred>0.5))
    # print('ground_truth:',ground_truth)
    groundtruth_keys = dataset.super_class_label_converter.inverse_transform(ground_truth[[0]].cpu().numpy()) 
    # print ('groundtruth',groundtruth_keys)
    prediction_keys = dataset.super_class_label_converter.inverse_transform(torch.ones_like(ground_truth)*(pred>0.5)[[0]].cpu().numpy())
    # print ("prediction_keys",prediction_keys)
    # hamming_score = calc_hamming_score(torch.ones_like(ground_truth)*(pred>0.5),ground_truth)
    # print('hamming_score',hamming_score)
    auroc.update(pred, ground_truth.long())
    try:
        lead_attention,_ = classification_net.encoder.get_attention()
        # print("lead attention shape",lead_attention.shape)
    except:
        print ("error in getting attention")
        break
    # print(lead_attention)
    # print_result(pred[0], dataset.super_classes_labels,topk=5)
    # normalized_lead_attention = (lead_attention[0]-lead_attention[0].min())/(lead_attention[0].max()-lead_attention[0].min())
    # normalized_lead_attention = (lead_attention[0]-lead_attention[0].min())/(lead_attention[0].max()-lead_attention[0].min())
    normalized_lead_attention = lead_attention[0]
        # print (key)
    # if len(prediction_keys[0])==1:
    for key in prediction_keys[0]:
        if key in groundtruth_keys[0]:
            ## accurately predicted:
            if key in label_attention_dict.keys():
                # current_map = label_attention_dict[key][0]
                # current_count = label_attention_dict[key][1]
                # if current_count>=50:continue
                # current_map.append(normalized_lead_attention)
                # current_count += 1
                label_attention_dict[key].append(normalized_lead_attention)
            else:
                label_attention_dict[key] = [normalized_lead_attention]

        # print (report[5:10])
    # print (pred[5:10])
    # print (ground_truth[5:10])
    # print (acc)
    # print (auroc_score)
    # if i==50:
    #     break
    

print('auroc score',auroc.compute())

## visualize the attention map for the population level model
fig, axes = plt.subplots(2,5,figsize=(25,10))
for i, (disease, attention_map_list) in enumerate(label_attention_dict.items()):
    average_attention = sum(attention_map_list)/len(attention_map_list)
    ## compute variance of the list
    variance = sum([(x-average_attention)**2 for x in attention_map_list])/(len(attention_map_list))
    print(variance.shape)
    print (f'{disease}:  {len(attention_map_list)}')
    x_ticks = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    colormap = sns.color_palette("Greens")

    g = sns.heatmap(average_attention.cpu().detach().numpy(),cmap=colormap,fmt=".1f",annot=False,xticklabels=x_ticks,yticklabels=x_ticks,ax =axes[0,i])

    # g = sns.heatmap(average_attention.cpu().detach().numpy(),cmap="YlGnBu", ax =axes[0,i])
    axes[0,i].set_title(disease+" average attention map")
    axes[0,i].set_ylabel("leads")
    axes[0,i].set_xlabel("leads")
    ## change xticks in each axis
    axes[0,i].xaxis.tick_top()
    g.set_xticklabels(x_ticks)
    g.set_yticklabels(x_ticks)

    ## variance map
    g_2 = sns.heatmap(variance.cpu().detach().numpy(),cmap=colormap,fmt=".1f",annot=False,xticklabels=x_ticks,yticklabels=x_ticks,ax =axes[1,i])
    axes[1,i].set_title(disease+' variance attention map')
    axes[1,i].set_ylabel("leads")
    axes[1,i].set_xlabel("leads")
    ## change xticks in each axis
    axes[1,i].xaxis.tick_top()
    g_2.set_xticklabels(x_ticks)
    g_2.set_yticklabels(x_ticks)

# ## change xticks
# plt.xticks(np.arange(0,12),x_ticks,rotation=45)
# plt.yticks(np.arange(0,12),x_ticks,rotation=45)
## save figure
fig.savefig("./population_level_attention_map_median_wave.png",dpi=500,bbox_inches='tight')
