"""Compute anglez and enmo
Future use for calculating other metrics from  (x,y,z) and temp? 
std(anglez), abs(diff(z))
median/mean"""

from actigrapy.common.data_model import InputData, OutputData
import numpy as np

def _calc_metrics(raw_data):
    #calculate ENMO and angle z from raw data
    #assumes a df 'raw_data' with columns 'X', 'Y', 'Z' with accelerometer data
    ENMO_calc = np.asarray(np.sqrt(np.square(raw_data).sum(axis =1))-1)
    angle_z_raw = np.asarray(np.arctan(raw_data.Z / (np.sqrt( np.square(raw_data.X) + np.square(raw_data.Y)))) / (np.pi/180))
    raw_data['ENMO'] = ENMO_calc
    raw_data['Anglez'] = angle_z_raw
    return raw_data
