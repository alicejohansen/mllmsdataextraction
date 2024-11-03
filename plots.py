import numpy as np
import plotly.express as px

def show_plotly_for_mean_sem(dict_obj_MAPE, title='Mean and SEM', yaxis_title='MAPE (%)', yaxis_range=[0, 250]):
    v2_mean_pt_MAPE = {k:np.mean(v) for k,v in dict_obj_MAPE.items()}
    v2_sem_pt_MAPE = {k:calculate_errorbar_sem(v) for k,v in dict_obj_MAPE.items()}
    # we won't use the below for now as it is more difficult to show in the plot
    # v2_iqr_pt_MAPE = {k:calculate_errorbar_iqr(v) for k,v in dict_obj_MAPE.items()}

    fig = px.scatter(x=v2_mean_pt_MAPE.keys(), 
                 y=v2_mean_pt_MAPE.values(), 
                 error_y={k:np.round(v,1) for k,v in v2_sem_pt_MAPE.items()}, 
                 error_y_minus={k:np.round(v,1) for k,v in v2_sem_pt_MAPE.items()})
    fig.update_layout(title=title, 
                      xaxis_title='Number of Data Points', 
                      yaxis_title=yaxis_title)
    fig.update_yaxes(range=yaxis_range) #range=[0, max(v2_mean_pt_MAPE.values())])
    fig.show()
    return fig

def calculate_error_bars(mean_values):
    std_devs = np.std(mean_values)
    sem = std_devs / np.sqrt(len(mean_values))
    lower_bound = np.maximum(0, mean_values - sem)
    upper_bound = mean_values + sem
    return lower_bound, upper_bound

def calculate_errorbar_sem(mean_values):
    std_devs = np.std(mean_values)
    sem = std_devs / np.sqrt(len(mean_values))
    return sem

def calculate_errorbar_iqr(mean_values):
    p25,p75 = np.percentile(mean_values,25), np.percentile(mean_values,75)
    return np.array([[p25 ,p75]]).T

