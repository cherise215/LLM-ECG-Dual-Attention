import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch_ecg.utils.rpeaks import xqrs_detect,gqrs_detect,hamilton_detect
from biosppy.signals import ecg as biosppy_ecg
import neurokit2 as nk ## package for simulation of ECG signals
from neurokit2.misc import check_random_state
from neurokit2.signal.signal_distort import _signal_distort_noise_multifrequency, _signal_distort_powerline, _signal_distort_artifacts, _signal_linear_drift
from scipy.stats import zscore
# process to get the template ECG signal for each lead
import numpy as np
import os
import torch
import pandas as pd
import xmltodict
import numpy as np
import seaborn as sns
## resample the ECG signal to 100 hz
import scipy.signal as signal
sns.set()


def plot_overlapped_multi_lead_signals(sample_y_nd,sample_x_hat_nd, labels=["GT ","Pred "],title="",color_list=["tab:blue","tab:red"]):
    '''
    input: sample_y_nd: numpy array, shape: (12, time_steps)
         sample_x_hat_nd: numpy array, shape: (12, time_steps)
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
    y_df.columns = [labels[0]+k for k in lead_names]
    y_recon.columns = [labels[1]+k for k in lead_names]

    figure = plot_multiframe_in_one_figure([y_df,y_recon],figsize=(15,4), figure_arrangement=figure_arrangement, logger=None, color_list = color_list, title=title)
    return figure
def get_simulated_ecg(duration=2, heart_rate=75, multi=False):
    '''
    Simulate the ECG signal
    input: 
        duration: the duration of the signal (seconds)
        heart_rate: the heart rate of the signal
        multi: whether to simulate the 12 leads signal
    output: 
        simulated: the simulated signal
    '''
    if multi:
        simulated = nk.ecg_simulate(duration=duration, heart_rate=heart_rate,method="multileads")
    else:
        simulated = nk.ecg_simulate(duration=duration, heart_rate=heart_rate)
    return simulated


def arraytodataframe(leads_signal ,plot=False, title= "", figsize=(8,5),lead_names=None):
    '''
    Convert the 12 leads signal to a dataframe with 12 columns
    input: leads_signal: 12 leads signal (12,5000)
    output: ecg12: dataframe with 12 columns
    '''

    ecg12 = pd.DataFrame()
    if lead_names is None:
        if leads_signal.shape[0]==12:
            lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
        elif leads_signal.shape[0]==8:
            lead_names= ['I','II','V1','V2','V3','V4','V5','V6']
        else:
            lead_names = np.arange(leads_signal.shape[0])
    for i, lead_name in enumerate(lead_names):
        ecg12[lead_name] = leads_signal[i]
    if plot:plot_ecg_frame(ecg12, title=title,figsize=figsize)
    return ecg12

def standardize_multi_lead_ecg(recording,sampling_rate=500,if_clean=True):
    # for each lead, 
    for i in range(recording.shape[0]):
        ## clean
        recording [i] = standardize_single_lead_ecg(recording[i],sampling_rate=sampling_rate,if_clean=if_clean)
    return recording

def standardize_single_lead_ecg(signal_1d, sampling_rate=500,if_clean=True):
    if if_clean: signal_1d = nk.ecg_clean(signal_1d, sampling_rate=sampling_rate,method = "neurokit") ## lowcut=0.5, highcut=50, method="neurokit"
    recording = zscore(signal_1d, axis=-1)
    recording = np.nan_to_num(recording)
    return recording



def plot_ecg_frame(ecg12, title="",figsize=(8,5),figure_arrangement=(4,3), axes=None):
    
    if axes is None:
        fig, axes = plt.subplots(nrows=figure_arrangement[0], ncols=figure_arrangement[1], figsize=figsize,squeeze=False)
    ecg12.plot(subplots=True, ax=axes)
    for ax in axes:
        for axi in ax:
            axi.legend(loc='upper left') # upper left corner
            # axi.set_ylim(-8,8)
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()
    return plt,axes


def extract_median_wave_for_single_lead(array, sampling_rate=500, output_length=300):
    signals,info = nk.ecg_process(array, sampling_rate=sampling_rate)
    heartbeats = nk.ecg_segment(signals["ECG_Clean"], info['ECG_R_Peaks'], sampling_rate=sampling_rate, show=False)
    list_of_waves = [v["Signal"].values for k, v in heartbeats.items()]
    
    average_wave = list_of_waves[len(list_of_waves)//2]
    print (average_wave.shape)
    # template= ecg_toolkit.ecg(signal=lead_array[i], sampling_rate=sampling_rate, show=False,interactive=False)['templates'].mean(axis=0)
    start = (output_length-len(average_wave))//2
    end = output_length-len(average_wave)-start
    average_wave = np.pad(average_wave, (start, end), 'edge')
    return average_wave

def get_median_wave_value(lead_array,sampling_rate=500, lead_axis=0, output_length=300):
    '''
    extract the median wave value from the ECG signal
    lead_array: 12*5000
    sampling_rate: sampling rate of the ECG signal
    lead_axis: 0 or 1, 0 means the first axis is the lead axis, 1 means the second axis is the lead axis
    output_length: the length of the output ECG signal
    return: the median wave value of the ECG signal (12,output_length) or (output_length,12) depending on the lead_axis
    '''
    num_lead = lead_array.shape[lead_axis]
    list_of_lead = []
    for i in range(num_lead):
        # print(i)
        if i==3:
            sign = -1.0
        else: sign= 1.0
        if lead_axis==0:
            template = extract_median_wave_for_single_lead(sign*lead_array[i], sampling_rate=sampling_rate, output_length=output_length)
            list_of_lead.append(sign*template)
        else:
            template = extract_median_wave_for_single_lead(sign*lead_array[:,i], sampling_rate=sampling_rate, output_length=output_length)
            list_of_lead.append(sign*template)
    new_lead_array = np.stack(list_of_lead)
    if lead_axis==1:
        new_lead_array = np.transpose(new_lead_array)
    elif lead_axis >1: raise ValueError('lead_axis should be 0 or 1')
    return new_lead_array


def plot_multiframe_in_one_figure(df_list, figsize=(12,12),figure_arrangement=(4,3), 
                                  style_list=["-","--","-.",":"],
                                  color_list=["tab:blue","tab:red","tab:orange","tab:green","tab:gray","tab:purple","tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"], logger=None, epoch=0, title="train/recon ECG"):
    """
    plot multiple dataframe in one figure

    Args:
        df_list (_type_): multi-dataframe
        figsize (tuple, optional): _description_. Defaults to (12,12).
        figure_arrangement (tuple, optional): _description_. Defaults to (4,3).
        style_list (list, optional): _description_. Defaults to ["-","--","-.",":"].
        color_list (list, optional): _description_. Defaults to ["tab:blue","tab:gray","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"].
    """
    assert len(color_list) >= len(df_list), "color list should be longer than the number of dataframes"
    assert len(style_list) >= len(df_list), "style list should be longer than the number of dataframes"
    fig, axes = plt.subplots(nrows=figure_arrangement[0], ncols=figure_arrangement[1], figsize=figsize,squeeze=False)
    
    first_plot = df_list[0].plot(subplots=True, ax=axes, style=style_list[0],color=color_list[0],legend=False, alpha=0.7)
    for i in range(1,len(df_list)): 
        df_list[i].plot(subplots=True,style=style_list[i], color=color_list[i],legend=False, alpha=0.7, ax=first_plot)
    
    for ax in axes:
        for axi in ax:
            axi.legend(loc='upper left') # upper left corner
    plt.tight_layout()
    # Adding plot to tensorboard
    if logger is not None:
        logger.experiment.add_figure(title, plt.gcf(),epoch)
    else:
        plt.title(title)
    return fig

def plot_two_data_frame_difference(df1,df2, title="Squared Difference",logger=None, epoch=0):
    '''
    Plot the difference between two dataframe
    input: df1,df2: dataframe
    output: plot
    '''
    assert df1.shape == df2.shape, "The shape of the two dataframe should be the same"
    assert df1.columns.all() == df2.columns.all(), "The column of the two dataframe should be the same"
    diff = (df1-df2)**2
    diff.plot.box(title="Difference (||x-x_hat'||2) between original and distorted signal")
    plt.tight_layout()
    if logger is not None:
        logger.experiment.add_figure(title, plt.gcf(),epoch)
    else:
        plt.title(title)
                                   
                                   
def detect_r_peaks_from_multiple_lead(simulated_signals,sampling_rate=500, method = "promac",if_clean=True, use_lead_index=None):
    '''
    Detect the R peaks from a 12 leads signal using different methods
    input: simulated_signals: 12 leads signal, 2d array (12,5000)
    output: r_peaks: the index of the R peaks
    '''
    r_peaks = []
    if use_lead_index is not None:
        ## use one representative lead to detect r peaks
        assert use_lead_index < simulated_signals.shape[0], "The index of the lead should be smaller than 12"
        r_peak = detect_r_peaks_from_single_lead(simulated_signals[use_lead_index],sampling_rate, method, if_clean=True)
        for i in range(simulated_signals.shape[0]):
            r_peaks.append(r_peak)
    else:
        for i in range(simulated_signals.shape[0]):
            r_peak = detect_r_peaks_from_single_lead(simulated_signals[i],sampling_rate, method, if_clean=if_clean)
            r_peaks.append(r_peak)
    return r_peaks

def detect_r_peaks_from_single_lead(signal_1d,sampling_rate=500, method = "promac", if_clean=True):
    '''
    Detect the R peaks from a single lead signal using different methods
    input: signal_1d: 1d array of the signal
    output: r_peak: the index of the R peaks
    '''
    if if_clean: 
        signal_1d = nk.ecg_clean(signal_1d, sampling_rate=sampling_rate)

    if method =="xqrs":
        r_peak =xqrs_detect(signal_1d,sampling_rate)
    elif method == "gqrs":
        r_peak =gqrs_detect(signal_1d,sampling_rate)
    elif method == "hamilton":
        r_peak =hamilton_detect(signal_1d,sampling_rate)
    elif method =="promac":
            # clean signal first
        info = nk.ecg_findpeaks(signal_1d, sampling_rate=sampling_rate, method=method, show=False)
        r_peak = info["ECG_R_Peaks"]
    elif method in ["neurokit","rodrigues2021","pantompkins1985",
                        "nabian2018",
                        "hamilton2002",
                        "martinez2004",
                        "christov2004",
                        "gamboa2008",
                        "elgendi2010",
                        "engzeemod2012",
                        "kalidas2017",
                        "rodrigues2021",
                        "koka2022"]:
        _, info = nk.ecg_peaks(signal_1d, method=method) 
        r_peak = info["ECG_R_Peaks"]
    
    elif method == "biosppy":
        # process it and plot
        out = biosppy_ecg.ecg(signal=signal_1d, sampling_rate=sampling_rate, show=False,interactive=False)
        r_peak = out["rpeaks"]
    else:
        raise ImportError("The method is not supported")
    return r_peak

## select a random lead masking from following:
    ## reference: ISIBrnoAIMT: cinc2021 winner algorithm
def batch_lead_mask(x, input_type = 'torch', same_mask_per_batch = False):
    '''
    perform random lead masking on the input batch
    input x: (batch_size, 12, 1024), dtype, torch or numpy
    return: (batch_size, 12, 1024), dtype, torch or numpy
    mask: (batch_size, 12, 1), dtype, torch or numpy
    '''
    masked_signals = x
    L2 = np.array([1,1,0,0,0,0,0,0,0,0,0,0])
    L3 = np.array([1,1,0,0,0,0,0,1,0,0,0,0])
    L4 = np.array([1,1,1,0,0,0,0,1,0,0,0,0])
    L6 = np.array([1,1,1,1,1,1,0,0,0,0,0,0])
    L8 = np.array([1,1,0,0,0,0,1,1,1,1,1,1])
    L12 = np.array([1,1,1,1,1,1,1,1,1,1,1,1])
    lead_mask_group = np.array([L2,L3,L4,L6,L8,L12])
    lead_mask_indexes = lead_mask_group.shape[0]
    if same_mask_per_batch:
        lead_mask_index = np.random.choice(a = lead_mask_indexes, size = 1, replace = False,p=[0.1,0.1,0.1,0.1,0.1,0.5])
        lead_mask = lead_mask_group[lead_mask_index]
        ## repeat the lead mask to the batch size
        lead_mask = np.repeat(lead_mask,x.shape[0],axis=0)
    else:
        lead_mask_index = np.random.choice(a = lead_mask_indexes, size = x.shape[0], replace = True,p=[0.1,0.1,0.1,0.1,0.1,0.5])
        lead_mask = lead_mask_group[lead_mask_index]

    if input_type == 'torch':
        lead_mask_new_dim = torch.from_numpy(lead_mask).long().to(x.device)
        lead_mask_new_dim = lead_mask_new_dim.unsqueeze(2).repeat(1,1,x.shape[2])
        masked_signals = lead_mask_new_dim*masked_signals 

    else:
        lead_mask_new_dim = lead_mask[:,:,np.newaxis]
        masked_signals = lead_mask_new_dim*masked_signals 

    return masked_signals, lead_mask_new_dim

def mask_signal(simulated_signals, critical_points, mask_whole_lead_prob = 0,lead_mask_prob=0.3, region_mask_prob=0.15, mask_length_range=[0.08, 0.18], mask_value=0.0, 
                random_drop_half_prob=0,sampling_rate=500):
    '''
    Mask the signal with whole lead  and/or random position at early stage of signals, and/or mask the critical points (r-peaks) at a probaiblity of region_mask_prob.
    input: simulated_signals: 12 leads signal , e.g.,(12,5000)
           critical_points: a list of critical points of the signal (12, N)
           mask_whole_lead_prob: probability of masking a whole lead
           lead_mask_prob: probability of masking a lead
           region_mask_prob: probability of masking a certain region of a lead
           mask_length_range: range of the length of the mask
           mask_value: value of the mask
    output: masked_signals: masked 12 leads signal (12,5000)
    mask: mask matrix (12,5000) indicating masked or not (0/1)
    '''
    assert simulated_signals.shape[0] == 12, "The shape of the input signal should be (12,N)"
    if critical_points is not None:
        assert len(critical_points) == 12, "The shape of the critical points should be (12,N)"
    masked_signals = simulated_signals.copy()
    mask = np.ones(simulated_signals.shape)
    signal_length = simulated_signals.shape[1]
    ## mask whole lead
    lead_mask = np.array([1,1,1,1,1,1,1,1,1,1,1,1])
    # median_values = np.median(simulated_signals, axis=1)
    minimum_mask_length = 80

    if np.random.rand() < mask_whole_lead_prob:
        ## select a random lead masking from following:
        ## reference: ISIBrnoAIMT: cinc2021 winner algorithm
        L2 = np.array([1,1,0,0,0,0,0,0,0,0,0,0])
        L3 = np.array([1,1,0,0,0,0,0,1,0,0,0,0])
        L4 = np.array([1,1,1,0,0,0,0,1,0,0,0,0])
        L6 = np.array([1,1,1,1,1,1,0,0,0,0,0,0])
        L8 = np.array([1,1,0,0,0,0,1,1,1,1,1,1])
        L12 = np.array([1,1,1,1,1,1,1,1,1,1,1,1])
        lead_mask_indexes = np.arange(6)
        lead_mask_group = np.array([L2,L3,L4,L6,L8,L12])
        lead_mask_index = np.random.choice(a = lead_mask_indexes, size = 1, replace = False,p=[0.1,0.1,0.1,0.1,0.1,0.5])
        lead_mask = lead_mask_group[lead_mask_index]
        masked_signals = lead_mask.T*masked_signals 
        mask = lead_mask.T*mask

    for i in range(simulated_signals.shape[0]):
        if lead_mask.flatten()[i]: ## if the lead has not been whole masked
            if np.random.rand() < lead_mask_prob:
                ## random drop r peaks
                if critical_points is not None:
                    if len(critical_points[i])>0:
                        for  critical_point_position in critical_points[i]:
                            if np.random.rand() < region_mask_prob:
                                mask_length = int(np.random.uniform(mask_length_range[0], mask_length_range[1])*sampling_rate)
                                if mask_length <minimum_mask_length:
                                    mask_length = minimum_mask_length
                                mask_start,mask_end = mask_pos_cal(signal_length, critical_point_position, mask_length)
                                ## find nearest critical point value
                                if isinstance(mask_value,float):
                                    masked_signals[i][mask_start:mask_end] = mask_value
                                elif isinstance(mask_value,str):
                                    if mask_value=='random':
                                        masked_signals[i][mask_start:mask_end] = np.random.choice(mask_value,size=mask_length)
                                    if mask_value=='median':
                                        masked_signals[i][mask_start:mask_end] = np.median(simulated_signals[i])
                                else:
                                    Warning("The mask value is not specified, using default value 0.0")
                                    masked_signals[i][mask_start:mask_end] = 0
                ## random drop half of the signal
            if np.random.rand() < random_drop_half_prob:
                signal_length = simulated_signals.shape[1]
                ## drop the first half of the signal or the second half of the signal
                if np.random.rand() < 0.5:
                    masked_signals[i][signal_length//2:] = 0.
                else:
                    masked_signals[i][:signal_length//2] = 0.
                        # mask[i][mask_start:mask_end]=0
    return masked_signals, mask

def mask_pos_cal(signal_length, mask_anchor, mask_length):
    mask_start = mask_anchor-int(mask_length//2)
    if mask_start < 0: mask_start = 0
    mask_end = mask_start+mask_length
    if mask_end > signal_length: mask_end=signal_length
    return mask_start,mask_end

def distort_multi_lead_signals(ecg_12_lead_signal,sampling_rate=500,noise_frequency=[5,20,100,150,175],noise_amplitude=0.2,
                               powerline_amplitude=0.05,
                               powerline_frequency = 10,
                               artifacts_amplitude=0.1,artifacts_number=5,linear_drift=0.1, random_prob = 0.5,
                               artifacts_frequency=100):
    '''
    Distort the 12 leads signal with noise, linear trend, artifacts etc.
    Input: 
    ecg_12_lead_signal: np array (12,N)

    '''
    num_leads = ecg_12_lead_signal.shape[0]
    new_signal = np.zeros(ecg_12_lead_signal.shape)
    for i in range(num_leads):    
        new_signal[i] = random_signal_distort(ecg_12_lead_signal[i],noise_amplitude=noise_amplitude,
                                noise_frequency=noise_frequency,
                                sampling_rate = sampling_rate,
                                powerline_amplitude=powerline_amplitude,
                                artifacts_amplitude=artifacts_amplitude,
                                powerline_frequency=powerline_frequency,
                                artifacts_number=artifacts_number,
                                artifacts_frequency = artifacts_frequency,
                                linear_drift=linear_drift,
                                random_prob=random_prob)
    return new_signal


def random_signal_distort(
    signal,
    sampling_rate,
    noise_shape="laplace",
    noise_amplitude=0,
    noise_frequency=100,
    powerline_amplitude=0,
    powerline_frequency=50,
    artifacts_amplitude=0,
    artifacts_frequency=100,
    artifacts_number=5,
    linear_drift=False,
    random_state=None,
    silent=False, random_prob=0.5):
    """**Signal distortion**

    Add noise of a given frequency, amplitude and shape to a signal.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    noise_shape : str
        The shape of the noise. Can be one of ``"laplace"`` (default) or
        ``"gaussian"``.
    noise_amplitude : float
        The amplitude of the noise (the scale of the random function, relative
        to the standard deviation of the signal).
    noise_frequency : float
        The frequency of the noise (in Hz, i.e., samples/second).
    powerline_amplitude : float
        The amplitude of the powerline noise (relative to the standard
        deviation of the signal).
    powerline_frequency : float
        The frequency of the powerline noise (in Hz, i.e., samples/second).
    artifacts_amplitude : float
        The amplitude of the artifacts (relative to the standard deviation of
        the signal).
    artifacts_frequency : int
        The frequency of the artifacts (in Hz, i.e., samples/second).
    artifacts_number : int
        The number of artifact bursts. The bursts have a random duration
        between 1 and 10% of the signal duration.
    linear_drift : bool
        Whether or not to add linear drift to the signal.
    random_state : None, int, numpy.random.RandomState or numpy.random.Generator
        Seed for the random number generator. See for ``misc.check_random_state`` for further information.
    silent : bool
        Whether or not to display warning messages.

    Returns
    -------
    array
        Vector containing the distorted signal.

    Examples
    --------
    .. ipython:: python

      import numpy as np
      import pandas as pd
      import neurokit2 as nk

      signal = nk.signal_simulate(duration=10, frequency=0.5)

      # Noise
      @savefig p_signal_distort1.png scale=100%
      noise = pd.DataFrame({"Freq100": nk.signal_distort(signal, noise_frequency=200),
                           "Freq50": nk.signal_distort(signal, noise_frequency=50),
                           "Freq10": nk.signal_distort(signal, noise_frequency=10),
                           "Freq5": nk.signal_distort(signal, noise_frequency=5),
                           "Raw": signal}).plot()
      @suppress
      plt.close()

    .. ipython:: python

      # Artifacts
      @savefig p_signal_distort2.png scale=100%
      artifacts = pd.DataFrame({"1Hz": nk.signal_distort(signal, noise_amplitude=0,
                                                        artifacts_frequency=1,
                                                        artifacts_amplitude=0.5),
                               "5Hz": nk.signal_distort(signal, noise_amplitude=0,
                                                        artifacts_frequency=5,
                                                        artifacts_amplitude=0.2),
                               "Raw": signal}).plot()
      @suppress
      plt.close()

    """
    # Seed the random generator for reproducible results.
    rng = check_random_state(random_state)

    # Make sure that noise_amplitude is a list.
    if isinstance(noise_amplitude, (int, float)):
        noise_amplitude = [noise_amplitude]

    signal_sd = np.std(signal, ddof=1)
    if signal_sd == 0:
        signal_sd = None

    noise = 0

    # Basic noise.
    if min(noise_amplitude) > 0 and np.random.rand() < random_prob:
        noise += _signal_distort_noise_multifrequency(
            signal,
            signal_sd=signal_sd,
            sampling_rate=sampling_rate,
            noise_amplitude=noise_amplitude,
            noise_frequency=noise_frequency,
            noise_shape=noise_shape,
            silent=silent,
            rng=rng,
        )

    # Powerline noise.
    if powerline_amplitude > 0 and  np.random.rand() < random_prob:
        noise += _signal_distort_powerline(
            signal,
            signal_sd=signal_sd,
            sampling_rate=sampling_rate,
            powerline_frequency=powerline_frequency,
            powerline_amplitude=powerline_amplitude,
            silent=silent,
        )

    # Artifacts.
    if artifacts_amplitude > 0 and np.random.rand() < random_prob:
        noise += _signal_distort_artifacts(
            signal,
            signal_sd=signal_sd,
            sampling_rate=sampling_rate,
            artifacts_frequency=artifacts_frequency,
            artifacts_amplitude=artifacts_amplitude,
            artifacts_number=artifacts_number,
            silent=silent,
            rng=rng,
        )

    if linear_drift and np.random.rand() < random_prob:
        noise += _signal_linear_drift(signal)
    distorted = signal + noise
    return distorted

def contains_medianwave_in_xml(xml_path):
    # Parse the XML file
    with open(xml_path, 'rb') as xml:
        ECGsignal = xmltodict.parse(xml.read().decode('utf8'))
        if "MedianSamples" not in ECGsignal['CardiologyXML']['RestingECGMeasurements'].keys():
            return False
        else:
            return True
def contains_full_lengthwave_in_xml(xml_path):
    # Parse the XML file
    with open(xml_path, 'rb') as xml:
        ECGsignal = xmltodict.parse(xml.read().decode('utf8'))
        if "StripData" not in ECGsignal['CardiologyXML'].keys():
            return False
        else:
            return True
### UKB related functions
def extract_ECG_data_from_xml_file(ECG_xmlfile_path, if_median=False, debug=False):
    '''
    given the path of the xml file, extract the ECG data and patient information from the xml file
    return a dictionary of patient information and 
    a dictionary of ECG data: 
        {'I':np.array(5000),
        'II':np.array(5000),
        'III':np.array(5000),
        'aVR':np.array(5000),
        'aVL':np.array(5000),
        'aVF':np.array(5000),
        'V1':np.array(5000),
        'V2':np.array(5000),
        'V3':np.array(5000),
        'V4':np.array(5000),
        'V5':np.array(5000),
        'V6':np.array(5000)}
    if if_median is True, return the median waveform data instead of the raw data
    
    '''
    with open(ECG_xmlfile_path, 'rb') as xml:
        ECGsignal = xmltodict.parse(xml.read().decode('utf8'))
        PatientInfo = ECGsignal['CardiologyXML']['PatientInfo']
        date= ECGsignal["CardiologyXML"]["ObservationDateTime"]
        year = date["Year"]
        month = date["Month"]
        day = date["Day"]
        hour = date["Hour"]
        minute = date["Minute"]
        second = date["Second"]
        date = str(year) + '-' + str(month) + '-' + str(day) + ' ' + str(hour) + ':' + str(minute) + ':' + str(second)
        # print ('date',date)
        pid = PatientInfo.get('PID')
        gender = PatientInfo['Gender']
        age = PatientInfo['Age']['#text']
        height = PatientInfo['Height']['#text']
        weight = PatientInfo['Weight']['#text']
        pacemaker = PatientInfo['PaceMaker']
  
        if debug: print(f'pid: {pid}, age: {age}, gender:{gender}, height:{height}, weight:{weight}, pacemaker:{pacemaker}')
        StripData = ECGsignal['CardiologyXML']['StripData']
        # print (StripData.keys())
        # print ('ArrhythmiaResults',StripData['ArrhythmiaResults'])
        # print ('strip data',StripData)
        # print (StripData)
        ## get the report of the ECG
        
        
        patient_info_dict= {
            'pid': pid,
            'age': age,
            'gender':gender,
            'height': height,
            'weight': weight,
            'date':date,
            'pacemaker': pacemaker,
            'sample_rate': StripData['SampleRate']['#text'],
            # "arrhythmia": StripData['ArrhythmiaResults'],
        }
        # print (StripData['NumberOfStrips'])
        leads = {}

        if if_median:
            data = ECGsignal['CardiologyXML']['RestingECGMeasurements']['MedianSamples']  # Yuling is using Median data now
           
        else:
            data = ECGsignal['CardiologyXML']['StripData']  # get the raw data
            
        for LeadID in range(0, int(data['NumberOfLeads'])):
                lead_vals = data['WaveformData'][LeadID]['#text']
                lead_id_name = data['WaveformData'][LeadID]['@lead']
                mm = np.fromstring(lead_vals, dtype=int, sep=',')
                leads[str(lead_id_name)] = mm      
    return patient_info_dict,leads

def convert_lead_dict_to_array(leads):
    lead_array=[]
    for leadID in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]:
        if leadID not in leads:
            leads[leadID] = np.zeros(len(leads["I"]))
            print(f'lead ID {leadID} not in leads',leadID)
            lead_array.append(np.zeros(len(leads["I"])))
        else:
            lead_array.append(leads[leadID])
    multi_lead_array = np.array(lead_array)  ## 12*5000
    return multi_lead_array      

def get_ecg_data_from_xml_file(f_path, sampling_rate=100.0, if_median=True, return_type="numpy"):
    '''
    get the ECG signal from the xml file
    :param f_path: the path of the xml file
    :param sampling_rate: the sampling rate of the output ECG signal
    :param if_median: if use the median value of the 12 leads
    :param return_type: the type of the output ECG signal, numpy array or dictionary
    :return: the ECG signal
    '''
    meta_dict, wave_dict = extract_ECG_data_from_xml_file(f_path, if_median=if_median)
    original_sample_rate = float(meta_dict["sample_rate"])
    new_wave = resample_ECG_signal(wave_dict, original_sample_rate, new_sample_rate=sampling_rate)
    if return_type == "numpy":
        new_wave = np.array([new_wave[k] for k in new_wave.keys()])
    else:
        pass
    return new_wave

def resample_ECG_signal(leads, sample_rate, new_sample_rate=100):
    '''
    leads: dict of 12-lead ECG signals, key: lead name, value: lead voltage
    sample_rate: original sample rate
    new_sample_rate: new sample rate
    '''

    resampled_leads = {}
    if isinstance(leads, dict):
        for lead_name, lead_voltage in leads.items():
            num_samples = len(lead_voltage)
            num_samples_new = int(num_samples * new_sample_rate / sample_rate)
            resampled_leads[lead_name] = signal.resample(lead_voltage, num_samples_new)
    elif isinstance(leads, np.ndarray):
        print ('before resampling',leads.shape)

        num_leads = leads.shape[0]
        num_samples = leads.shape[1]
        num_samples_new = int(num_samples * new_sample_rate / sample_rate)
        resampled_leads = np.zeros((num_leads, num_samples_new))
        for i in range(num_leads):
            resampled_leads[i] = signal.resample(leads[i], num_samples_new)
        print ('resampled_leads',resampled_leads.shape)
    return resampled_leads

# Plot 12-lead ECG signals in separate subplots
def plot_12_lead(leads, separate_lead=False):
    '''
    leads: dict of 12-lead ECG signals, key: lead name, value: lead voltage
    Plot 12-lead ECG signals in separate subplots
    '''
    if not separate_lead:
        plt.figure(figsize=(8, 4))
        for lead_name, lead_voltage in leads.items():
            plt.plot(lead_voltage, label=lead_name)

        plt.xlabel("Sample Number")
        plt.ylabel("Voltage (mV)")
        plt.title("12-Lead ECG Signals (All Leads)")
        plt.legend()
        plt.grid(True)
    else:
        # Plot each lead's ECG signal in separate subplots below
        num_leads = len(leads)
        fig, axs = plt.subplots(num_leads, 1, figsize=(8,12), sharex=True)

        for i, (lead_name, lead_voltage) in enumerate(leads.items()):
            axs[i].plot(lead_voltage)
            axs[i].set_ylabel("Voltage (mV)")
            axs[i].set_title(lead_name)
            axs[i].grid(True)

        axs[-1].set_xlabel("Sample Number")
    plt.suptitle("12-Lead ECG Signals (Individual Leads)")
    plt.tight_layout()
    plt.show()