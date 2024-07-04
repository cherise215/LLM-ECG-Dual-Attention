# LLM-ECG-Dual-Attention

[IEEE trans. on Big Data, 2024] Large Language Model-informed ECG Dual Attention Network for Heart Failure Risk Prediction.

If you find the code useful, please cite
```
@ARTICLE{Chen2024-ea,
  title         = "Large Language Model-informed {ECG} Dual Attention Network
                   for Heart Failure Risk Prediction",
  author        = "Chen, Chen and Li, Lei and Beetz, Marcel and Banerjee,
                   Abhirup and Gupta, Ramneek and Grau, Vicente",
  year          =  2024,
  journal = "IEEE trans. on Big Data"
}
```
or 
```
Chen C, Li L, Beetz M, Banerjee A, Gupta R, Grau V. Large Language Model-informed ECG Dual Attention Network for Heart Failure Risk Prediction. IEEE. trans on Big Data. 2024. Available: http://arxiv.org/abs/2403.10581
```

## Key Features
- **ECG Dual Attention Network**: A novel, lightweight ECG dual attention framework that visualizes cross-lead and temporal attention simultaneously. Initialize an instance with:
    ```python
    from multi_modal_heart.model.ecg_net_attention import ECGAttentionAE
    ecg_net = ECGAttentionAE(num_leads=n_leads, time_steps=max_seq_len, 
                             z_dims=512, 
                             linear_out=512, 
                             downsample_factor=5, 
                             base_feature_dim=4,
                             if_VAE=False,
                             use_attention_pool=False,
                             no_linear_in_E=True)
    ```
- **Robust Risk Prediction**: Implemented `StratifiedBatchSampler` (see `/path/to/LLM-ECG-Dual-Attention/multi_modal_heart/tasks/train_risk_regression_model_with_recon_task.py`) and replaced standard dropout with `BatchwiseDropout` for risk prediction tasks (see `multi_modal_heart/model/custom_layers/fixable_dropout/, function: BatchwiseDropout`) to stabilize training. `BatchwiseDropout` applies the same feature masking to all subjects within a batch. In this way, it ensure fair comparison between censored and uncensored subjects when calculating the risk prediction loss. 
- **Large-langguage model guided structured text embedding from ECG report to deal with uncertainty and bilingual**

## Project structure
├── data                    # Raw and processed data
│   ├── ptxbl
│   ├── ukb
├── multi_modal_heart       # core code base
├── pretrained_weights      # Pretrained model weights
├── result                  # Results of the experiments with risk prediction models.
├── scripts                 # Shell and other scripts
│   └── train.sh            # Training script
├── toolkits                # Toolkit scripts and utilities
│   └── ukb/assets
│       └── generate_ptb_scp_with_confidence_embeddings.ipynb  # Example of a Jupyter notebook for LLM-based text embedding processing.
├── README.md               # Readme file
└── requirements.txt        # Python dependencies

## Set Up
- Git clone this project: 
`git clone https://github.com/cherise215/LLM-ECG-Dual-Attention.git`
- (optional) Create a fresh Python 3.9.x virtual environment. [guide](https://www.arch.jhu.edu/python-virtual-environments/)
- Install PyTorch with CUDA support for GPU-enabled computation (in our settings, we use Pytorch 1.13.0, cuda version 12.2, Driver Version: 535.183.01, hardware: GTX TITAN X) [check official guide here](https://pytorch.org/)
`pip3 install torch torchvision torchaudio`
- Install other required python libraries with: 
 `pip install -r requirements.txt`

## Preparation:
Here are some basic steps to run the project:
- Data Preparation: Ensure that your data is put in the the appropriate directories under data/. For example, PTXBL data should be in `[project folder]/data/ptxbl` [Dowloand post-processed one via Google Drive](https://drive.google.com/file/d/1FkCoGAfMeg2dmOSBYDljW8mVp9Rli9W4/view?usp=sharing) or official websites [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) and [PTB-XL+](https://physionet.org/content/ptb-xl-plus/1.0.1/) for median wave information. For UK biobank data, unfortunately, we cannot directly share the data due to the UK biobank's strict data sharing policy. 

- Download pretrained models and saved structured text embeddings using ClinicalBert for different patients in PTB-XL [link](https://drive.google.com/drive/folders/1j6qbuQYjJJ4yn_zz4aHZdQRrvuFUJ7pS?usp=sharing). Please put it as `[project folder]/pretrained_weights`. For interested researchers, step-by-step tutorial on how we generate these embeddings can be found at: `toolkits/generate_ptb_scp_with_confidence_embeddings.ipynb`. 

## Task 1: Pretraining 
-  Pretraining of ECG dual attention network using PTB-XL data, see `multi_modal_heart/tasks/pretrain.py`
    - for LLM-informed pretraining,text embeddings for each patient's record is needed.
    '''
    CUDA_VISIBLE_DEVICES=0 python multi_modal_heart/tasks/pretrain.py --ae_type "ECG_attention_512" --ECG2Text  --use_median_wave  --warm_up 
    '''
## Task 2: Finetuning 
- You can also directly run code for finetuning the risk prediction task using our pretrained models.
    - Pretrained model: the model checkpoint can be found at: [G-drive](https://drive.google.com/drive/folders/1j6qbuQYjJJ4yn_zz4aHZdQRrvuFUJ7pS?usp=sharing). Please put it under `Path/to/LLM-ECG-Dual-Attention/pretrained_weights/model/ECG2Text/checkpoint_best_loss-v2.ckpt`
    - Risk prediction data. Unfortunately, we are not allowed to share the UK biobank data. Here, we use a dummy data for users to test their environment. Please change the datasetname in those below commands  to `dummy`, e.g "--dataset_name 'dummy'" to turn it on. 
    - Use a dummy dataset for testing:
      `CUDA_VISIBLE_DEVICES=0 python multi_modal_heart/tasks/train_risk_regression_model_with_recon_task.py --model_name "ECG_attention" --checkpoint_path "./pretrained_weights/model/ECG2Text/checkpoint_best_loss-v2.ckpt" --dataset_name "dummy"  --batch_size 128  --lr 1e-4 --n_folds 2 --latent_code_dim 512`

    - Train from scratch (just without specifying the pretrained model path):
      `CUDA_VISIBLE_DEVICES=1 python multi_modal_heart/tasks/train_risk_regression_model_with_recon_task.py --model_name "ECG_attention" --train_from_scratch --dataset_name "dummy"  --batch_size 128  --lr 1e-4 --n_folds 2 --latent_code_dim 512`
    0 
    - In case you have access to our real-wolrd datasets (e.g., you are a member in our UK biobank application). Then you could have access to our processed data incl. patients w/ myocardial infarction (MI) and patients with hypertension (HYP).
        - For MI  (batch size=128, as the total number of dataset is only 800): 
        ``` 
        CUDA_VISIBLE_DEVICES=1 python multi_modal_heart/tasks/train_risk_regression_model_with_recon_task.py --model_name "ECG_attention" --train_from_scratch --dataset_name "MI_with_HF_event" --batch_size 128 --lr 1e-4 --n_folds 2 --latent_code_dim 512 --checkpoint_path "./pretrained_weights/model/ECG2Text/checkpoint_best_loss-v2.ckpt" 
        ```
        - For HYP (batch size=1024, as this dataset is much bigger): 
        ``` 
        CUDA_VISIBLE_DEVICES=1 python multi_modal_heart/tasks/train_risk_regression_model_with_recon_task.py --model_name "ECG_attention" --train_from_scratch --dataset_name "HYP_with_HF_event" --batch_size 1024 --lr 1e-4 --n_folds 2 --latent_code_dim 512 --checkpoint_path "./pretrained_weights/model/ECG2Text/checkpoint_best_loss-v2.ckpt" 
        ```

## Contact
For any questions or issues, please contact [Chen Chen] at [work.cherise@gmail.com]. Thank you for checking out our project!