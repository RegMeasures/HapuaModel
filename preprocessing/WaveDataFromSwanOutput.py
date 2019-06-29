import pandas as pd
import numpy as np

WaveModelOutputFile = 'C:\\Users\measuresrj\OneDrive - NIWA\Hapua\Hurunui\SWAN\output\TS_hurunui.txt'
ShoreNormalDir = 128 # [degrees] used for back refracting to deep water

#%% Read model wave data
ColNames = ['ModelTime', 'Year', 'Month', 'Day', 'Hour', 'Min', 
            'Hsig', 'Tm01', 'WavePeriod', 'RTpeak', 'Dir', 'Dspr', 
            'PkDir', 'XWindv', 'YWindv', 'Ubot', 'Urms', 'Wlen_h', 'Depth', 
            'XTransp', 'YTransp', 'XWForce', 'YWForce', 'Radstr', 'Setup']

WaveData = pd.read_table(WaveModelOutputFile, sep=' ', skipinitialspace=True,
                         skiprows=27, names=ColNames)
WaveData['DateTime'] = pd.to_datetime(WaveData.iloc[:,1:5])
WaveData = WaveData.set_index('DateTime')

# Extract the columns we want
OutputData = WaveData.loc[:,['WavePeriod','Wlen_h']]

#%% Calculate net wave power magnitude and direction
# Note All directional wave information is reported in direction it arrives FROM
OutputData['WavePower'] = 1025 * 9.81 * np.sqrt(np.power(WaveData['XTransp'],2) + 
                                                np.power(WaveData['YTransp'],2))
OutputData['EDir_h'] = np.rad2deg(np.arctan2(WaveData['XTransp'], WaveData['YTransp']))-180
OutputData['EDir_h'] = OutputData['EDir_h'] % 360

#%% Calculate offshore significant wave height

# wave angle relative to beach [radians] (at model output point)
WaveData['Angle_h'] = np.mod((np.deg2rad(ShoreNormalDir) - 
                              np.deg2rad(WaveData.Dir)) + np.pi/2, 
                             2*np.pi) - np.pi/2

# Flag onshore directed waves and set offshore directed waves to have Hs=0
WaveData['OnshoreWaves'] = np.abs(WaveData['Angle_h']) < np.pi/2
WaveData.loc[WaveData['OnshoreWaves']==False,'Hsig'] = 0
WaveData.loc[WaveData['OnshoreWaves']==False,'Angle_h'] = 0

# wave number (k) and ratio of group speed (celerity) to wave speed (n) (both calculated at 10m)
WaveData['k_h'] = 2.0 * np.pi / WaveData['Wlen_h']
WaveData['n_h'] = 0.5 * (1.0 + 2.0 * WaveData['k_h'] * WaveData['Depth'] / 
                         np.sinh(2.0 * WaveData['k_h'] * WaveData['Depth']))

# Reverse shoal model data to deep water (needed for runup equations)
WaveData['LoffRatio'] = 1.0 / np.tanh(2.0 * np.pi * WaveData['Depth'] / 
                                      WaveData['Wlen_h'])
WaveData['Wlen_Offshore'] = WaveData['Wlen_h'] * WaveData['LoffRatio']
WaveData['Angle_Offshore'] = np.arcsin(np.maximum(np.minimum(WaveData['LoffRatio'] * 
                                                             np.sin(WaveData['Angle_h']), 
                                                             1.0), -1.0)) # min and max is to prevent tiny irregularities generating complex numbers due to calculation of asin(>1) or asin(<-1)
OutputData['Hsig_Offshore'] = (WaveData['Hsig'] / 
                               np.sqrt(0.5 * WaveData['n_h'] * WaveData['LoffRatio'] * 
                                       (np.cos(WaveData['Angle_Offshore']) / 
                                        np.cos(WaveData['Angle_h']))))

# Save the data ready for the model to use
OutputData.to_csv('..\inputs\WaveData.csv', float_format='%1.3f', 
                  date_format='%d/%m/%Y %H:%M')