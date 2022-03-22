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
import tensorflow as tf

from viktor import UserException
from ..calculations import preprocessing


def look_back(data: 'pandas.DataFrame (single column) or numpy.array', steps_back: 'int >= 0') -> np.array:
    """
    Method that translates the array a few elements backwards, the first 'step_back' number of elments
    are set to value 0.
    """
    data = np.array(data)
    b = np.zeros(len(data))
    for i in range(steps_back, len(b)):
        b[i] = data[i-steps_back]
    return b


def look_forward(data, steps_forward: int) -> np.array:
    data = np.array(data)
    f = np.zeros(len(data))
    for i in range(len(f) - steps_forward):
        f[i] = data[i+steps_forward]
    return f


def add_upper_and_lower_params(df: pd.DataFrame) -> pd.DataFrame:
    """
    Method that adds the cone pressure and shear pressure a few distance elements back and forward
    (required for ML algorithm). First element values are set to zero.
    Input requires GEF DataFrame or ML train DataFrame (i.e. DataFrame must include 'qc [MPa]' and 'fs [MPa]').

    """
    step_list = [1, 2, 3, 4, 5] # Take up to 5 steps back and forward

    # Initialize dataframe columns
    for steps_back in step_list:
        df['qc_-'+str(steps_back)+' [MPa]'] = np.zeros(len(df))
        df['fs_-'+str(steps_back)+' [MPa]'] = np.zeros(len(df))

        # Same amount of steps forward
        df['qc_+'+str(steps_back)+' [MPa]'] = np.zeros(len(df))
        df['fs_+'+str(steps_back)+' [MPa]'] = np.zeros(len(df))

    # Add the steps back
    for steps_back in step_list:        
        df['qc_-'+str(steps_back)+' [MPa]'] = look_back(df['qc [MPa]'], steps_back)
        df['fs_-'+str(steps_back)+' [MPa]'] = look_back(df['fs [MPa]'], steps_back)

        df['qc_+'+str(steps_back)+' [MPa]'] = look_forward(df['qc [MPa]'], steps_back)
        df['fs_+'+str(steps_back)+' [MPa]'] = look_forward(df['fs [MPa]'], steps_back)

    return df


def delete_gef_columns(df: pd.DataFrame):
    """
    Method that deletes GEF data in GEF DataFrame that is irrelevant to ML algorithm
    """
    for column in [
        'Pile diameter [m]',
        'Total resist MA [MN]',
        'Cone resist MA [MN]',
        'inclination [degrees]',
        'u2 [MPa]'
    ]:
        try:
            df.pop(column)
        except:
            print('Info:', column, 'not available in GEF data.')    
    return df


def create_gef_df(gef_data, diameter) -> pd.DataFrame:
    """
    Method that creates Pandas GEF DataFrame that is compatible with ML algorithm.
    """
    data_dict = gef_data

    df = preprocessing.measurement_data_to_processed_DF(data_dict['measurement_data'], diameter)
    df = delete_gef_columns(df)

    # Set first data point to elevation = 0 m
    df['elevation [m]'] = np.round(df['elevation [m]'] - np.array(df['elevation [m]'])[0], decimals=2)

    # The ML algorithm operates on intervals of 10 cm's, while CPT is given every 2 cm ->
    # take every element of the CPT dataframe that has value n * -0.1 m
    try:
        index_gef = np.where(np.array(df['elevation [m]']) == -0.1)[0][0]
        index_gef_2 = np.where(np.array(df['elevation [m]']) == -0.2)[0][0]
        index_gef = index_gef_2 - index_gef
    except:
        print('Warning: index where elevation equals -0.1 m not found. Setting index to default (5).')
        index_gef = int(5)

    df = df[::index_gef]  # As the ML algorithm is trained from samples every 0.1 m
    return df


def choose_model(option: str, **kwargs):
    if option is None:
        raise UserException('No ML model selected!')
        
    if option == 'Default Berkel model':
        model = tf.keras.models.load_model('./app/calculations/models/dense')
        print('Option is Default Berkel model, Default model loaded!')
        return model

    if option == 'Default Zaandam model':
        model = tf.keras.models.load_model('./app/calculations/models/Dense_Zaandam')
        print('Option is Default Zaandam model, Default model loaded!')
        return model
        
    else:
        raise UserException('Trained model class is not (yet) available in model database')


def ML_prediction(gef_data: dict, diameter: float, model_option: str, **kwargs) \
        -> 'array, pandas.DataFrame (containing elevation)':
    """
    Method that uses TensorFlow model in combination with supplied GEF file to compute pile
    penetration speed prediction.
    """
    df = create_gef_df(gef_data, diameter)

    # The ML algorithm does not use knowledge on the elevation (otherwise it would assume the soil properties
    # were the same as at another location at a certain depth (which is obviously not the case))
    elevation_df = df.pop('elevation [m]')

    df = preprocessing.add_composition(df)
    # Add the qc and fs values a few meters upwards (as this is found to be relevant for the ML algorithm)
    df = add_upper_and_lower_params(df)

    ML_model = choose_model(option=model_option)  # Model chosen for prediction according to front-end input

    predictions = ML_model.predict(df)
    predictions = predictions.flatten()
    predictions = np.nan_to_num(predictions)

    return predictions, elevation_df
