import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGLSTMnet(nn.Module):
    def __init__(self, n_lead=12, num_length=1024, z_dims=64, out_ch=12, act=nn.ELU(inplace=True)):
        super(ECGLSTMnet, self).__init__()
        feature_scale = num_length//512
        self.encoder_signal = CRNN(n_lead=n_lead, z_dims=z_dims)
        self.fc1 = nn.Linear(z_dims, 256*feature_scale)
        self.fc2 = nn.Linear(256*feature_scale, 512*feature_scale)
        self.act = act

        self.up = nn.Upsample(size=(out_ch, num_length), mode='bilinear')
        self.deconv = DoubleDeConv(1, 1)
        self.if_VAE = True

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var) # standard deviation
        eps = torch.randn_like(std).to(std.device) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def decode_signal(self, latent_z):
        inputs = latent_z
        f = self.act(self.fc1(inputs))
        f = self.act(self.fc2(f))
        u = self.up(f.reshape(f.shape[0], 1, 8, -1))
        dc = self.deconv(u)
        dc = dc.squeeze(1)
        return dc

    def forward(self, signal_input,mask=None):
        mu, log_var = self.encoder_signal(signal_input)
        latent_z_signal = self.reparameterize(mu, log_var)
        y_ECG = self.decode_signal(latent_z_signal)
        self.z_log_var = log_var    ## declare this for pytorch lightning solver
        self.z_mu = mu  ## declare this for pytorch lightning solver
        return y_ECG


class CRNN(nn.Module):
    def __init__(self, n_lead=8, z_dims=16):
        super(CRNN, self).__init__()
        n_out = 128
        self.z_dims = z_dims

        self.cnn = nn.Sequential(
            nn.Conv1d(n_lead, n_out, kernel_size=16, stride=2, padding=2),
            nn.BatchNorm1d(n_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(n_out, n_out*2, kernel_size=16, stride=2, padding=2),
            nn.BatchNorm1d(n_out*2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.rnn = BidirectionalLSTM(256, z_dims*4, z_dims*2)

    def forward(self, input):
        x = self.cnn(input)
        b, c, w = x.size()
        x = x.permute(2, 0, 1)

        output = self.rnn(x).permute(1, 0, 2)
        features = torch.max(output, 1)[0]
        mean = features[:, :self.z_dims]
        std = features[:, self.z_dims:] + 1e-6
        return mean, std

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output

class DoubleDeConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleDeConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(out_ch, out_ch, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


if __name__=="__main__":
    import torch
    import torch.nn.functional as F

    # Assuming N = 1 for simplicity
    N = 1
    num_leads = 12
    time_steps = 1024
    z_dims = 64
    # Create a random input tensor of shape N * 1 * 12 * 1024
    input_ecg = torch.randn(N, num_leads, time_steps)

    # Create an instance of the ECGAutoencoder
    autoencoder = ECGLSTMnet(num_leads, time_steps, z_dims)

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