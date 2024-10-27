# imports
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import json
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

def generate_save_truth_data(trial_seed, dataset_size, chart_type, save_dir):
    """Generates and saves truth table data and in JSON format to the given save directory for the 
    given chart type, trial seed, and dataset size. 

    Parameters:
    trial_seed (int): Seed for random number generation.
    dataset_size (int): Frequency of data points.
    chart_type (str): Type of chart to generate truth data for.
    save_dir (str): Directory to save the generated truth data.

    Returns:
    x (list): List of x values.
    y_actual (list): List of y values.
    """
    np.random.seed(trial_seed)
    y_actual = np.random.randint(100, size=dataset_size)
    x = np.arange(1, dataset_size + 1)

    data = {
    "x": x.tolist(),
    "y": y_actual.tolist()
    }

    title = f'{trial_seed}_{chart_type}_{dataset_size}_actual.json'
    file_loc = os.path.join(save_dir, title)
    with open(file_loc, 'w') as f:  
        json.dump(data, f, indent=4)
    
    return x, y_actual


def generate_save_plot_figure(x, y, chart_type, trial_seed, save_dir):
    """Generates and saves a plot figure for the given x and y values and chart type to the given save directory. 

    Parameters:
    x (list): List of x values.
    y (list): List of y values.
    chart_type (str): Type of chart to generate plot figure for.
    save_dir (str): Directory to save the generated plot figure.
    trial_seed (int): Seed for random number generation reproducibility. 


    Returns:
    None

    """
    dataset_size = len(x)
    chart_title = f'{trial_seed}_{chart_type}_{dataset_size}'
    
    plt.figure(dpi=100, figsize=(8,6))
    plt.scatter(x, y)

    major_ticks = np.arange(0, 101, 10)
    minor_ticks = np.arange(0, 101, 5)
    plt.xticks(x)
    plt.yticks(major_ticks)
    plt.yticks(minor_ticks, minor=True)
    plt.grid(which='minor', alpha=0.4)
    plt.grid(which='major', alpha=0.7)

    image_path = os.path.join(save_dir, chart_title + '.png')
    plt.savefig(image_path)
    plt.close()

    img_data = Image.open(image_path).convert("RGB")

    return img_data


