import torch
import torch.nn as nn

 
class ECGMarcelVAE(nn.Module):
    def __init__(self, num_leads=12, time_steps=1024, z_dims=64):
        super(ECGMarcelVAE, self).__init__()
        self.encoder = ECGEncoder(num_leads, time_steps, z_dims)
        self.decoder = ECGDecoder(num_leads, time_steps, z_dims)
        self.if_VAE = True ## declare this for pytorch lightning solver
    
 
    
    def forward(self, ecg, mask=None):
        ## mask is not used in this model, but for the sake of compatibility with other models
        z = self.encoder(ecg)
        self.z =z
        ecg_preds = self.decoder(z)
        self.z_log_var = self.encoder.z_log_var    ## declare this for pytorch lightning solver
        self.z_mu = self.encoder.z_mu  ## declare this for pytorch lightning solver
        return ecg_preds
    

class ECGEncoder(nn.Module):
    def __init__(self, num_leads=12, time_steps=1024, z_dims=64,act = nn.ELU(inplace=True)):
        assert time_steps % 2 == 0, "Time steps must be even"
        super(ECGEncoder, self).__init__()
        self.num_leads = num_leads
        self.time_steps = time_steps
        
        self.conv1 = nn.Sequential(
                 nn.Conv2d(1, 5, kernel_size=(1, 40), padding='same',bias=False),
                 nn.BatchNorm2d(5),
                 act,

                )
        self.conv2 = nn.Sequential(
                 nn.Conv2d(5, 5, kernel_size=(num_leads, 1), padding='valid',bias=False),
                 nn.BatchNorm2d(5),
                 act,

                )
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.flatten = nn.Flatten()
        self.dense_mean = nn.Linear(int((time_steps // 2) * 5), z_dims)
        self.dense_std = nn.Linear(int((time_steps // 2) * 5), z_dims)
    
    ## function to extract func name
    def get_func_name(self):
        return "VAEEncoder"
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample
    
    def forward(self, ecg, mask=None):
        ecg_features = ecg.view(-1, 1, self.num_leads, self.time_steps)
        ecg_features = self.conv1(ecg_features)
        ecg_features = self.conv2(ecg_features)
        ecg_features = self.avg_pool(ecg_features)
        ecg_features = self.flatten(ecg_features)        # ecg_mean = ecg_features
        # ecg_std = ecg_features
        ecg_mean = self.dense_mean(ecg_features)
        ecg_std = self.dense_std(ecg_features)
        sample  = self.reparameterize(ecg_mean, ecg_std)
        self.z  = sample
        self.z_mu = ecg_mean  ## declare this for pytorch lightning solver
        self.z_log_var = ecg_std  ## declare this for pytorch lightning solver
        return  sample

    def get_features_after_pooling(self,ecg, mask=None):
        ## warning: this function is not used in the model, but for the sake of compatibility with other models
        
        return self.forward(ecg)


# remove ths shared encoder for now, since we don't have the pcd data yet
# class SharedEncoder(nn.Module):
#     def __init__(self, z_dims):
#         super(SharedEncoder, self).__init__()
#         self.dense_mean = nn.Linear(z_dims * 4, z_dims)
#         self.dense_std = nn.Linear(z_dims * 4, z_dims)
    
#     def forward(self, pcd_mean_ED, pcd_mean_ES, ecg_mean, pcd_std_ED, pcd_std_ES, ecg_std):
#         mean = torch.cat([pcd_mean_ED, pcd_mean_ES, ecg_mean], dim=1)
#         std = torch.cat([pcd_std_ED, pcd_std_ES, ecg_std], dim=1)
        
#         mean = self.dense_mean(mean)
#         std = self.dense_std(std)
        
#         return mean, std

 


class ECGDecoder(nn.Module):
    def __init__(self, num_leads, time_steps, z_dims, act=nn.ELU(inplace=True)):
        super(ECGDecoder, self).__init__()
        self.num_leads = num_leads
        self.time_steps = time_steps
        
        self.dense1 = nn.Linear(z_dims, int((time_steps // 2) * 5))
        self.dense2 = nn.Linear(int((time_steps // 2) * 5), int((time_steps // 2) * 5 * 2))
        self.conv_transpose1 =nn.Sequential(
                                nn.ConvTranspose2d(5, 5, kernel_size=(num_leads, 1), padding=(0, 0),bias=False),
                                nn.BatchNorm2d(5),
                                act
                                )
                                
        self.conv_transpose2 = nn.Sequential(
                                nn.ConvTranspose2d(5, 1, kernel_size=(1, 41), padding=(0, 20),bias=False),
                                nn.BatchNorm2d(1),
                                act
                                )

    
    def forward(self, features):
        ecg_features = self.dense1(features)
        ecg_features = self.dense2(ecg_features)
        ecg_features = ecg_features.reshape(-1, 5, 1, int(self.time_steps))
        ecg_features = self.conv_transpose1(ecg_features)
        # print ("before conv1",ecg_features.shape)
        ecg_features = self.conv_transpose2(ecg_features)
        ecg_preds = ecg_features.reshape(-1, self.num_leads,self.time_steps)
        # print (ecg_features.shape)
        return ecg_preds

 


if __name__=="__main__":
    import torch
    import torch.nn.functional as F

    # Assuming N = 1 for simplicity
    N = 1
    num_leads = 12
    time_steps = 1024
    z_dims = 64
    # Create a random input tensor of shape N * 1 * 12 * 1024
    input_ecg = torch.randn(N, 1, num_leads, time_steps)

    # Create an instance of the ECGAutoencoder
    autoencoder = ECGMarcelVAE(num_leads, time_steps, z_dims)

    # Pass the input through the autoencoder
    ecg_preds= autoencoder(input_ecg)

    # Calculate the reconstruction loss
    reconstruction_loss = F.mse_loss(ecg_preds, input_ecg)

    # Print the shapes of the output tensors
    print("Input shape:", input_ecg.shape)
    print("Reconstructed ECG shape:", ecg_preds.shape)
    # print("Mean shape:", mean.shape)
    # print("Standard Deviation shape:", std.shape)
    print("Reconstruction Loss:", reconstruction_loss.item())