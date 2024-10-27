# imports
import os
import numpy as np
import json
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
load_dotenv()
from tqdm.notebook import tqdm

def calculate_MAPE(y_true, y_pred):
    """Calculate the mean absolute percentage error between true and predicted values.

    Parameters:
    y_true (array-like): Array of true values.
    y_pred (array-like): Array of predicted values.

    Returns:
    float: The maximum error between the true and predicted values.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    non_zero_mask = y_true != 0
    y_true_non_zero = y_true[non_zero_mask]
    y_pred_non_zero = y_pred[non_zero_mask]

    mape = np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100
    return mape


def calculate_median_error(y_true, y_pred):
  """Calculate the median absolute error between true and predicted values.

  Parameters:
  y_true (array-like): Array of true values.
  y_pred (array-like): Array of predicted values.

  Returns:
  float: The maximum error between the true and predicted values.
  """
  median_error = np.median(np.abs(y_true - y_pred))
  return median_error


def calculate_max_error(y_true, y_pred):
  """Calculate the maximum absolute error between true and predicted values.

  Parameters:
  y_true (array-like): Array of true values.
  y_pred (array-like): Array of predicted values.

  Returns:
  float: The maximum error between the true and predicted values.
  """
  max_error = np.max(np.abs(y_true - y_pred))
  return max_error


def process_json_to_arrays(json_file_path):
  """Given a JSON file path, reads the file and returns the x and y values in np arrays.

  Parameters:
  json_file_path (str): Absolute path to the JSON file.

  Returns:
  x_values (array-like): Array of x values.
  y_values (array-like): Array of y values.
  """
  with open(json_file_path, 'r') as f:
    data = json.load(f)

  x_values = np.array(data['x'])
  y_values = np.array(data['y'])
  return x_values, y_values


def calc_metrics(truth_json_file, predicted_json_file):
  """Given the truth data and the predicted JSON file paths for one trial, read the files 
        and returns the MAPE, median error and max error metrics. 

  Parameters:
  truth_json_file (str): Absolute path to the JSON file.
  predicted_json_file (str): Absolute path to the JSON file.

  Returns:
    mape (float): Mean absolute percentage error.
    median_error (float): Median absolute error.
    max_error (float): Maximum absolute error.
  """
   
  true_y = process_json_to_arrays(truth_json_file)[1]
  predicted_y = process_json_to_arrays(predicted_json_file)[1]

  mape = calculate_MAPE(true_y, predicted_y)
  median_error = calculate_median_error(true_y, predicted_y)
  max_error = calculate_max_error(true_y, predicted_y)

  return mape, median_error, max_error


def calc_mape(truth_json_file, predicted_json_file):
    """Given the truth data and the predicted JSON file paths for one trial, read the files
        and returns the MAPE metric.

    Parameters:
    truth_json_file (str): Absolute path to the JSON file.
    predicted_json_file (str): Absolute path to the JSON file.

    Returns:
        mape (float): Mean absolute percentage error.
        """ 
    
    true_y = process_json_to_arrays(truth_json_file)[1]
    predicted_y = process_json_to_arrays(predicted_json_file)[1]

    return calculate_MAPE(true_y, predicted_y)


def calc_median_error(truth_json_file, predicted_json_file):
    """Given the truth data and the predicted JSON file paths for one trial, read the files
        and returns the median error metric.

    Parameters:
    truth_json_file (str): Absolute path to the JSON file.
    predicted_json_file (str): Absolute path to the JSON file.

    Returns:
        median_error (float): Median absolute error.
        """ 
    
    true_y = process_json_to_arrays(truth_json_file)[1]
    predicted_y = process_json_to_arrays(predicted_json_file)[1]

    return calculate_median_error(true_y, predicted_y)


def calc_max_error(truth_json_file, predicted_json_file):
    """Given the truth data and the predicted JSON file paths for one trial, read the files
        and returns the max error metric.

    Parameters:
    truth_json_file (str): Absolute path to the JSON file.
    predicted_json_file (str): Absolute path to the JSON file.

    Returns:
        max_error (float): Maximum absolute error.
        """ 
    
    true_y = process_json_to_arrays(truth_json_file)[1]
    predicted_y = process_json_to_arrays(predicted_json_file)[1]

    return calculate_max_error(true_y, predicted_y)


def calculate_batch_MAPE_by_data_frequency(truth_json_dir, predicted_json_dir):
    """Given the truth data and predicted JSON directory paths for a batch of trials, read the files 
        and returns the average MAPE and the MAPE for each trial.

    Parameters:
    truth_json_dir (str): Absolute path to the directory containing the truth JSON files.
    predicted_json_dir (str): Absolute path to the directory containing the predicted JSON files.

    Returns:
        average_mapes (dict): Dictionary of average MAPEs by data frequency.
        all_mapes (dict): Dictionary with lists of MAPEs for each trial by data frequency.
        """
    
    truth_files = os.listdir(truth_json_dir)
    
    all_mapes = {}
    for data_freq in np.arange(5, 21, 1):
        all_mapes[data_freq] = []
    
    for truth_file in truth_files:
        trial_num = int(truth_file.split('_')[0])
        chart_type = truth_file.split('_')[1]
        data_freq = int(truth_file.split('_')[2])
        
        truth_json_file = os.path.join(truth_json_dir, truth_file)
        predicted_json_file = os.path.join(predicted_json_dir, f'{trial_num}_{chart_type}_{data_freq}.json')
        
        mape = calc_mape(truth_json_file, predicted_json_file)
        all_mapes[data_freq].append(mape)
    
    average_mapes = {}
    for data_freq in np.arange(5, 21, 1):
        average_mapes[data_freq] = np.mean(all_mapes[data_freq])
    
    return average_mapes, all_mapes
    

def calculate_batch_median_error(truth_json_dir, predicted_json_dir):
    """Given the truth data and predicted JSON directory paths for a batch of trials, read the files 
        and returns the average median error and the median error for each trial.

    Parameters:
    truth_json_dir (str): Absolute path to the directory containing the truth JSON files.
    predicted_json_dir (str): Absolute path to the directory containing the predicted JSON files.

    Returns:
        average_median_errors (dict): Dictionary of average median errors by data frequency.
        all_median_errors (dict): Dictionary with lists of median errors for each trial by data frequency.
        """
    
    truth_files = os.listdir(truth_json_dir)
    
    all_median_errors = {}
    for data_freq in np.arange(5, 21, 1):
        all_median_errors[data_freq] = []
    
    for truth_file in truth_files:
        trial_num = int(truth_file.split('_')[0])
        chart_type = truth_file.split('_')[1]
        data_freq = int(truth_file.split('_')[2])
        
        truth_json_file = os.path.join(truth_json_dir, truth_file)
        predicted_json_file = os.path.join(predicted_json_dir, f'{trial_num}_{chart_type}_{data_freq}.json')
        
        median_error = calc_median_error(truth_json_file, predicted_json_file)
        all_median_errors[data_freq].append(median_error)
    
    average_median_errors = {}
    for data_freq in np.arange(5, 21, 1):
        average_median_errors[data_freq] = np.mean(all_median_errors[data_freq])
    
    return average_median_errors, all_median_errors
    

def calculate_batch_max_error(truth_json_dir, predicted_json_dir):
    """Given the truth data and predicted JSON directory paths for a batch of trials, read the files 
        and returns the average max error and the max error for each trial.

    Parameters:
    truth_json_dir (str): Absolute path to the directory containing the truth JSON files.
    predicted_json_dir (str): Absolute path to the directory containing the predicted JSON files.

    Returns:
        average_max_errors (dict): Dictionary of average max errors by data frequency.
        all_max_errors (dict): Dictionary with lists of max errors for each trial by data frequency.
        """
    
    truth_files = os.listdir(truth_json_dir)
    
    all_max_errors = {}
    for data_freq in np.arange(5, 21, 1):
        all_max_errors[data_freq] = []
    
    for truth_file in truth_files:
        trial_num = int(truth_file.split('_')[0])
        chart_type = truth_file.split('_')[1]
        data_freq = int(truth_file.split('_')[2])
        
        truth_json_file = os.path.join(truth_json_dir, truth_file)
        predicted_json_file = os.path.join(predicted_json_dir, f'{trial_num}_{chart_type}_{data_freq}.json')
        
        max_error = calc_max_error(truth_json_file, predicted_json_file)
        all_max_errors[data_freq].append(max_error)
    
    average_max_errors = {}
    for data_freq in np.arange(5, 21, 1):
        average_max_errors[data_freq] = np.mean(all_max_errors[data_freq])
    
    return average_max_errors, all_max_errors
   
  
def metric_post_processing(chart_type, truth_json_dir, predicted_json_dir):
  """Given the chart type and the truth data and predicted JSON file paths for a batch of trials, read the files 
        and returns the MAPE, median error and max error metrics. 

  Parameters:
  truth_json_file (array-like): Absolute path to the JSON file.
  predicted_json_file (array-like): Absolute path to the JSON file.

  Returns:
    mape (float): Mean absolute percentage error.
    median_error (float): Median absolute error.
    max_error (float): Maximum absolute error.
  """
  truth_files = os.listdir(truth_json_dir)
  predicted_files = os.listdir(predicted_json_dir)

  MAPEs_by_data_frequency = {}
  Median_Errors_by_data_frequency = {}
  max_errors_by_data_frequency = {}

  for data_freq in np.arange(5, 21, 1):
    MAPEs_by_data_frequency[data_freq] = []
    Median_Errors_by_data_frequency[data_freq] = []
    max_errors_by_data_frequency[data_freq] = []

  for truth_file in truth_files:
    data_freq = int(truth_file.split('_')[2])
    trial_num = int(truth_file.split('_')[0])

    truth_json_file = os.path.join(truth_json_dir, truth_file)
    predicted_json_file = os.path.join(predicted_json_dir, f'{trial_num}_{chart_type}_{data_freq}.json')

    mape, median_error, max_error = calc_metrics(truth_json_file, predicted_json_file)


    MAPEs_by_data_frequency[data_freq].append(mape)
    Median_Errors_by_data_frequency[data_freq].append(median_error)
    max_errors_by_data_frequency[data_freq].append(max_error)


  average_MAPEs = {}
  average_med_errors = {}
  mean_max_error = {}
  max_max_error = {}
  for data_freq in np.arange(5, 21, 1):
    average_MAPEs[data_freq] = np.mean(MAPEs_by_data_frequency[data_freq])

  for data_freq in np.arange(5, 21, 1):
    average_med_errors[data_freq] = np.mean(Median_Errors_by_data_frequency[data_freq])

  for data_freq in np.arange(5, 21, 1):
    max_max_error[data_freq] = np.max(max_errors_by_data_frequency[data_freq])
    mean_max_error[data_freq] = np.mean(max_errors_by_data_frequency[data_freq])

  return MAPEs_by_data_frequency, Median_Errors_by_data_frequency, max_errors_by_data_frequency, average_MAPEs, average_med_errors, mean_max_error, max_max_error

