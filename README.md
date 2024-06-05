# LLM-ECG-Dual-Attention


-Installation: 
- 

- Code for the implementation of ECG dual attention network:`LLM-ECG-Dual-Attention/multi_modal_heart/model/ecg_net_attention.py`
    ```
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
- Code for finetuning the risk prediction task using the pretrained model.
    - Pretrained model: the model checkpoint can be found at: [G-drive]()
    - Data. We are not allowed to share the UKB data. Here, we use a dummy data for users to test their environment. 
    - todo: add synthetic data here. 
    - To run the finetuning, run command: 
        `CUDA_VISIBLE_DEVICES=1 python multi_modal_heart/tasks/train_risk_regression_model_with_recon_task.py --model_name "ECG_attention" --checkpoint_path "./pretrained_weights/model/ECG2Text/checkpoint_best_loss-v2.ckpt" --dataset_name "MI_with_HF_event"  --batch_size 128  --lr 1e-4 --n_folds 2 --latent_code_dim 512`
    - To train from scratch, run command:
      `CUDA_VISIBLE_DEVICES=1 python multi_modal_heart/tasks/train_risk_regression_model_with_recon_task.py --model_name "ECG_attention" --train_from_scratch --dataset_name "MI_with_HF_event"  --batch_size 128  --lr 1e-4 --n_folds 2 --latent_code_dim 512`
    - Note, to stabilize training, we implemented stratified batch sampling when sampling batchs for optimization as well as replacing standard dropout with batch dropout in this code repo. The batch dropout allows each time, the same feature masking is applied to all the subjects within the same batch. 
    
- Code for pretraining network: `multi_modal_heart/tasks/pretrain.py`
    -  Preparation: Text embeddings 
        To extract text embeddings from report for text-ecg alignment, we use pre-computed confidence-reweighted embeddings for EACH patient. Those embeddings can be directly downloaded from [XXX](). Please download it, and put it under the directory: `[Project folder]/pretrained_weights/text_embeddings/PTBXL_LLM_scp_structed_text_embedding.pkl` 
        For interested researchers, step-by-step tutorial on how we generate these embeddings can be found at: `toolkits/generate_ptb_scp_with_confidence_embeddings.ipynb`. 

- Others:
    - UK Biobank:
        - [todo] How to find ECG/CMR date?
        - [todo] How can I find heart failure data?


