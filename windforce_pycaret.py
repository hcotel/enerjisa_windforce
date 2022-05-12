import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.plotting.register_matplotlib_converters()
from scipy.stats import probplot

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
from pycaret.regression import *

pd.options.display.max_rows = 20
pd.options.display.max_columns = 12
features = pd.read_csv('data/features.csv')
power = pd.read_csv('data/power.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')

all = pd.merge(features, power, on="Timestamp", how='left')
all["is_train"] = all['Power(kW)'].notnull()
all['Timestamp'] = pd.to_datetime(all['Timestamp'])
all['Date'] = pd.to_datetime(all['Timestamp'].dt.date)

target_feature = 'power'
submission_feature = 'Power(kW)'

power_offset = all['Power(kW)'].min()
#power_offset = 50
power_max = all['Power(kW)'].max()
all['Power(kW)'] = all['Power(kW)'] - power_offset

# all = all.loc[(all['Gearbox_T1_High_Speed_Shaft_Temperature'] > 13) | (all.is_train == 0)]
# all = all.loc[(all['Gearbox_T3_High_Speed_Shaft_Temperature'] >= 30) | (all.is_train == 0)]
# all = all.loc[(all['Torque'] != 99999) | (all.is_train == 0)]

all['Timestamp'] = pd.to_datetime(all['Timestamp'])
sample_submission['Timestamp'] = pd.to_datetime(sample_submission['Timestamp'])

all['month'] = all['Timestamp'].dt.month
all['hour'] = all['Timestamp'].dt.hour
all['week'] = all['Timestamp'].dt.weekofyear
all['dayofweek'] = all['Timestamp'].dt.dayofweek
all['dayofyear'] = all['Timestamp'].dt.dayofyear
all['minute'] = all['Timestamp'].dt.hour * 60 + all['Timestamp'].dt.minute

all.replace(to_replace=99999, value=np.nan, inplace=True)

all['Nacelle Position_Degree_sin'] = np.sin(all['Nacelle Position_Degree'] * np.pi/180).abs()
all['Nacelle Position_Degree_cos'] = np.cos(all['Nacelle Position_Degree'] * np.pi/180).abs()
all['Nacelle Revolution_sin'] = np.sin(all['Nacelle Revolution'] * np.pi).abs()
all['Nacelle Revolution_cos'] = np.cos(all['Nacelle Revolution'] * np.pi).abs()

all["operation_12"] = (all['Operating State'] == 12.0).astype('int')
all["operation_11"] = (all['Operating State'] == 11.0).astype('int')
all["operation_15"] = (all['Operating State'] == 15.0).astype('int')
all["operation_19"] = (all['Operating State'] == 19.0).astype('int')
all["operation_16"] = (all['Operating State'] == 16.0).astype('int')
all["n-set_1_0"] = (all['N-set 1'] == 0.0).astype('int')
all["n-set_1_limit"] = all['N-set 1'] * 2780 / 1735.0
all['limit_difference'] = all['Internal Power Limit'] - all["Power(kW)"]
all["limit_capped"] = (all['limit_difference'] < 5).astype('int')
all["pf*reactive"] = all['Power Factor'] * all['Reactive Power']
all["pf_abs"] = all['Power Factor'].abs()
all = all.rename(columns={'Power(kW)': 'power'})

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
all = reduce_mem_usage(all)

features = [
       'Gearbox_T1_High_Speed_Shaft_Temperature', 'Nacelle Position_Degree_cos', 'Nacelle Revolution_sin', 'Nacelle Revolution_cos', 'Nacelle Revolution', 'Nacelle Position_Degree',
       'Gearbox_T3_High_Speed_Shaft_Temperature', 'Operating State', "operation_12", "operation_11","operation_15","operation_16","operation_19",
       'Gearbox_T1_Intermediate_Speed_Shaft_Temperature', "n-set_1_0",
       'Temperature Gearbox Bearing Hollow Shaft', 'Tower Acceleration Normal',
       'Gearbox_Oil-2_Temperature', 'Tower Acceleration Lateral',
       'Temperature Bearing_A', 'Temperature Trafo-3',
       'Gearbox_T3_Intermediate_Speed_Shaft_Temperature',
       'Gearbox_Oil-1_Temperature', 'Gearbox_Oil_Temperature', 'Torque',
       'Converter Control Unit Reactive Power', 'Temperature Trafo-2',
       'Reactive Power', 'Temperature Shaft Bearing-1',
       'Gearbox_Distributor_Temperature', 'Moment D Filtered',
       'Moment D Direction', 'N-set 1', 'Power Factor',
        #'Temperature Shaft Bearing-2', 'Temperature_Nacelle',
        'Voltage A-N',
       #'Temperature Axis Box-3',
       'Voltage C-N', 'Temperature Axis Box-2',
       'Temperature Axis Box-1', 'Voltage B-N',
       'Converter Control Unit Voltage',
       #'Temperature Battery Box-3',
       #'Temperature Battery Box-2', 'Temperature Battery Box-1',
        #'Hydraulic Prepressure',
        'Angle Rotor Position',
       #'Temperature Tower Base',
      'Pitch Offset-2 Asymmetric Load Controller',
       'Pitch Offset Tower Feedback', 'Line Frequency', 'Internal Power Limit',
       #'Circuit Breaker cut-ins',
       #'Particle Counter',
       #'Tower Accelaration Normal Raw',
       'Torque Offset Tower Feedback',
       #'Blade-2 Actual Value_Angle-B', #'External Power Limit',
       #'Blade-1 Actual Value_Angle-B', 'Blade-3 Actual Value_Angle-B',
       'Temperature Heat Exchanger Converter Control Unit',
       #'Tower Accelaration Lateral Raw', 'Temperature Ambient',
        'Pitch Offset-1 Asymmetric Load Controller',
       'Tower Deflection', 'Pitch Offset-3 Asymmetric Load Controller',
       'Wind Deviation 1 seconds', 'Wind Deviation 10 seconds',
       'Proxy Sensor_Degree-135', 'State and Fault', 'Proxy Sensor_Degree-225',
       'Blade-3 Actual Value_Angle-A', 'Scope CH 4',
       'Blade-2 Actual Value_Angle-A', 'Blade-1 Actual Value_Angle-A',
       'Blade-2 Set Value_Degree', 'Pitch Demand Baseline_Degree',
       'Blade-1 Set Value_Degree', 'Blade-3 Set Value_Degree',
       'Moment Q Direction', 'Moment Q Filltered', 'Proxy Sensor_Degree-45',
       'Turbine State', 'Proxy Sensor_Degree-315',
       'month', 'hour', 'week', 'dayofweek', 'dayofyear', target_feature
]

def prep(df, features):
    new_features = features.copy()
    for feature in features:
        if feature in ['month', 'hour', 'week', 'dayofweek', 'dayofyear']:
            continue
        df[feature + '_lag1'] = df[feature].shift(1)
        df[feature + '_lead1'] = df[feature].shift(-1)
        df[feature + '_diff1'] = df[feature] - df[feature + '_lag1']
        df[feature + '_rolling2'] = df[feature].rolling(2).mean()
        df[feature + '_log'] = np.log1p(df[feature])
        new_features.append(feature + '_lag1')
        new_features.append(feature + '_lead1')
        new_features.append(feature + '_diff1')
        new_features.append(feature + '_rolling2')
        new_features.append(feature + '_rolling3')

    return df

all = prep(all, features)

features = ['pf_abs',
            'Torque',
            'Torque_lag1',
            'Torque_lead1',
            'Torque_rolling2',
            'Torque_rolling3',
            'Torque_log',
            'Blade-3 Actual Value_Angle-A',
            'Scope CH 4',
            'Pitch Offset-2 Asymmetric Load Controller',
            'Wind Deviation 1 seconds',
            'Proxy Sensor_Degree-315',
            'Gearbox_Oil-2_Temperature',
            'Gearbox_Oil-2_Temperature_lead1',
            #'Temperature Trafo-2',
            'Moment D Filtered',
            'Proxy Sensor_Degree-45',
            'Tower Acceleration Normal',
            'Proxy Sensor_Degree-135',
            'Gearbox_T1_Intermediate_Speed_Shaft_Temperature',
            'Gearbox_T1_Intermediate_Speed_Shaft_Temperature_lead1',
            'Gearbox_T1_High_Speed_Shaft_Temperature',
            'Converter Control Unit Voltage',
            'Pitch Offset Tower Feedback',
            'Pitch Offset Tower Feedback_diff1',
            'Pitch Demand Baseline_Degree',
            'Tower Acceleration Lateral',
            #'Temperature Trafo-3',
            'Operating State',
            'Operating State_lead1',
            'N-set 1',
            'N-set 1_rolling2',
            'Blade-1 Set Value_Degree',
            'n-set_1_0',
            'pf*reactive',
            'Turbine State',
            'hour',
            'Temperature Bearing_A',
            #'Blade-2 Set Value_Degree',
            'Gearbox_T3_High_Speed_Shaft_Temperature',
            'Gearbox_Oil-1_Temperature',
            #'Nacelle Revolution_cos',
            target_feature
            ]

object_cols = ['n-set_1_0', 'hour']
numeric_cols = list(set(all[features].columns) - set(object_cols))
numeric_cols.remove(target_feature)
print('numeric_cols: ', numeric_cols)
print('object_cols: ', object_cols)

all[object_cols] = all[object_cols].astype('category')

train_x = all[all.is_train == 1][features]
train_y = all[all.is_train == 1][target_feature]
test_x = all[all.is_train == 0][features]



reg = setup(data=train_x,
            target=target_feature,
            session_id=42,
            data_split_shuffle=True,
            create_clusters=False,
            normalize=True,
            normalize_method='robust',
            numeric_features=numeric_cols,
            categorical_features=object_cols,
            fold_strategy='kfold',
            feature_selection=True,
            imputation_type='iterative',
            silent=True,
            use_gpu=True,
            fold=5,
            n_jobs=-1)

remove_metric('MSE')
remove_metric('RMSLE')
remove_metric('MAPE')

top_models = compare_models(sort='RMSE', n_select=3, include=['lightgbm', 'xgboost', 'catboost', 'et', 'rf'])

regression_results = pull()
print(regression_results)

blender = blend_models(top_models)
final = finalize_model(blender)

regression_results = pull()
print(regression_results)

predictions = predict_model(blender, data=test_x.reset_index(drop=True))
sample_submission['Timestamp'] = pd.to_datetime(sample_submission['Timestamp'])
all['Timestamp'] = pd.to_datetime(all['Timestamp'])
sample_submission[submission_feature] = predictions['Label'] + power_offset
sample_submission = pd.merge(sample_submission, all, on="Timestamp", how='left')
sample_submission[submission_feature] = np.clip(sample_submission[submission_feature], -48.5966682434082, 2780)
print(f"Submission mean: {sample_submission[submission_feature].mean()}")
sample_submission[["Timestamp", submission_feature]].to_csv(f"submission_pycaret_3.csv", sep=",", index=False)