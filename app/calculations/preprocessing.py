"""Copyright (c) 2022 VIKTOR B.V.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

VIKTOR B.V. PROVIDES THIS SOFTWARE ON AN "AS IS" BASIS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import pandas as pd


def compute_shaft_resistance(fs: pd.DataFrame, elevation: pd.DataFrame, diameter: float):
    """
    Method that computes the shaft resistance according to a finite element method
    Shaft resistance across the entire elevation spectrum according to equation
    F_shaft_resist ~ pi * D * delta_z * sum(fs)
    """
    
    circum = np.pi * diameter  # Compute circumference of the pile
    np_elevation = np.array(elevation)  # numpy array of the elevation
    np_fs = np.array(fs)  # numpy array of the shaft pressure
    
    elevation_diff = np.abs(np_elevation[0:-1] - np_elevation[1:])  # Elevation difference array
    fs_means = np.mean(np.vstack((np_fs[0:-1], np_fs[1:])), axis=0)  # Average shear pressure on elevation intervals
        
    shaft_force_arr = circum * elevation_diff * fs_means  # Shaft pressure force addition at elevation interval

    shaft_force_cont = np.cumsum(shaft_force_arr)  # Shaft pressure force as a function of pile penetration depth
    elevation_arr = np_elevation[0] - np.cumsum(elevation_diff)

    # Set the first array element to a shaft resistance force of 0 (as no part of the pile is submerged,
    # therefore no shaft resistance force)
    elevation_arr = np.insert(elevation_arr, 0, np_elevation[0])
    shaft_force_cont = np.insert(shaft_force_cont, 0, 0)

    return shaft_force_cont, elevation_arr # MegaNewton, meters


def compute_qc_MA(qc: pd.DataFrame, elevation: pd.DataFrame, window_size=20):
    """
    Method that computes the moving average of the cone tip pressure.
    Filters out high-frequency components of the qc signal.
    Returns array of the same size as the input qc.
    """
    qc_MA = qc.rolling(window=window_size).mean()
    elevation_MA = elevation.rolling(window=window_size).min()

    qc_MA = np.array(qc_MA)
    elevation_MA = np.array(elevation_MA)

    for i in range(1, window_size):
        qc_i = qc.rolling(window=i).mean()
        elevation_i = elevation.rolling(window=i).min()
        
        i = i-1
        qc_MA[i] = np.array(qc_i)[i]
        elevation_MA[i] = np.array(elevation_i)[i]   

    return qc_MA, elevation_MA


def add_composition(df: pd.DataFrame, epsilon=1e-2):
    """
    Add soil composition according to Robertson method (1990) to dataframe
    """
    # Robertson method 2010
    df['I_c'] = np.sqrt((3.47 - np.log10((df['qc [MPa]'] + epsilon) / 0.1))**2 +
                        (np.log10(df['fs [MPa]']/(df['qc [MPa]']+epsilon) * 100 + epsilon) + 1.22)**2)
    df['dense sand'] = np.zeros(len(df['I_c']))
    df['clean sand'] = np.zeros(len(df['I_c']))
    df['sand mixtures'] = np.zeros(len(df['I_c']))
    df['silt mixtures'] = np.zeros(len(df['I_c']))
    df['clays'] = np.zeros(len(df['I_c']))
    df['peats'] = np.zeros(len(df['I_c']))
    
    df.loc[df['I_c'] <= 1.31, 'dense sand'] = 1
    df.loc[(df['I_c'] > 1.31) & (df['I_c'] <= 2.05), 'clean sand'] = 1
    df.loc[(df['I_c'] > 2.05) & (df['I_c'] <= 2.6), 'sand mixtures'] = 1
    df.loc[(df['I_c'] > 2.6) & (df['I_c'] <= 2.95), 'silt mixtures'] = 1
    df.loc[(df['I_c'] > 2.95) & (df['I_c'] <= 3.6), 'clays'] = 1
    df.loc[df['I_c'] > 3.6, 'peats'] = 1
    
    df.pop('I_c')
    return df


def measurement_data_to_processed_DF(measurement_data: dict, diameter=None, epsilon=1e-2) -> pd.DataFrame:
    """
    Method that transforms measurement_data (GEF) dictionary to a Pandas DataFrame of the GEF data
    """
    if diameter is not None:    
        # Dictionary to DataFrame
        df = pd.DataFrame.from_dict(measurement_data)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna()
        df['elevation'] = df['elevation']*1e-3
        df = df.rename(columns={
            'Rf': 'Rf [-]',
            'fs': 'fs [MPa]',
            'qc': 'qc [MPa]',
            'u2': 'u2 [MPa]',
            'elevation': 'elevation [m]',
            'inclination': 'inclination [degrees]'
        })
        df['Rf [-]'] = df['fs [MPa]'] / (df['qc [MPa]'] + epsilon)  # Just to be sure that it is not in percentages

        # Compute shaft and cone forces
        shaft_resist, elevation_arr = compute_shaft_resistance(df['fs [MPa]'], df['elevation [m]'], diameter)

        # Moving average of the cone resistance pressure
        qc_MA, elevation_arr = compute_qc_MA(df['qc [MPa]'], df['elevation [m]'])

        # Computation of cone resistance force
        cone_resist = np.array(df['qc [MPa]']) * np.pi / 4 * diameter**2
        cone_resist_MA = qc_MA * np.pi / 4 * diameter**2

        total_force = shaft_resist + cone_resist
        total_force_MA = shaft_resist + cone_resist_MA

        # Insert new computed values into DataFrame
        df['Pile diameter [m]'] = diameter * np.ones(len(elevation_arr))
        df['Cone resist [MN]'] = cone_resist
        df['Cone resist MA [MN]'] = cone_resist_MA
        df['Shaft resist [MN]'] = shaft_resist
        df['Total resist [MN]'] = total_force
        df['Total resist MA [MN]'] = total_force_MA
        
        return df
    else:
        # Dictionary to DataFrame
        df = pd.DataFrame.from_dict(measurement_data)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna()
        df['elevation'] = df['elevation']*1e-3
        df = df.rename(columns={
            'Rf': 'Rf [-]',
            'fs': 'fs [MPa]',
            'qc': 'qc [MPa]',
            'u2': 'u2 [MPa]',
            'elevation': 'elevation [m]',
            'inclination': 'inclination [degrees]'
        })
        
        return df
