import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import neurokit2 as nk

sns.set()

def vis_colored_lines(input_arr, color_arr, orig_sampling_rate, desired_sampling_rate=5000, 
                      scatter_size = 10,
                      rescale_color=False, x_axis_max=None, title=None, 
                      color_map='copper',ax=None):
    """
    Visualize a 1D signal with a color gradient.
    :param input_arr: array
        The input signal.
    :param color_arr: array
        The color signal.
    :param orig_sampling_rate: int
        The original sampling rate of the input signal.
    :param desired_sampling_rate: int
        The desired sampling rate of the input signal.
    :param scatter_size: int
        The size of the scatter plot.
    :param rescale_color: bool
        If True, rescale the color array.
    :param x_axis_max: int
        The maximum value of the x-axis.
    :param title: str
        The title of the plot.
    :param color_map: str
        The color map to use.
    :param ax: matplotlib axis
        The axis to plot on.
    :return: matplotlib axis
        The axis on which the plot was drawn.
    """

    if orig_sampling_rate < desired_sampling_rate:    
    # Resample the signal
        input_array = nk.signal_resample(input_arr, sampling_rate=orig_sampling_rate, desired_sampling_rate=desired_sampling_rate, method='interpolation')
        color_array = nk.signal_resample(color_arr, sampling_rate=orig_sampling_rate, desired_sampling_rate=desired_sampling_rate, method='interpolation')
    
    else:
        input_array = input_arr
        color_array = color_arr
    assert len(input_array) == len(color_array), "The length of the input array and the color array should be the same."
         # Preprocess ECG signal
    x_axis_max  =1 if x_axis_max is None else x_axis_max
    x = np.linspace(0,x_axis_max, input_array.shape[0])
    ## resclae
    if rescale_color:
        color_array = (color_array - np.min(color_array)) / (np.max(color_array) - np.min(color_array))
    else:
        color_array = color_array
    print(len(x), len(input_array), len(input_arr))
    if ax is None:
        fig, ax = plt.subplots()
        points = plt.scatter(x, input_array, c=cm.get_cmap(color_map)(color_array), edgecolor='none', s=scatter_size)
    else:
        points = ax.scatter(x, input_array, c=cm.get_cmap(color_map)(color_array), edgecolor='none', s=scatter_size)
    if ax is None:
        plt.title(title)
    else:
        ax.set_title(title)

    plt.colorbar(cm.ScalarMappable(cmap=cm.get_cmap(color_map)))
    return ax
