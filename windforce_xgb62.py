import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scipy import stats
from scipy.special import inv_boxcox
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.tree import ExtraTreeRegressor
import optuna
from constants import (
    run_optuna,
    run_adv,
    use_nfolds,
    trial,
    validation_strategy,
    validation_files_index,
    check_val_results,
    use_scaler,
    optuna_trials,
    plot_importance,
    model_type,
    isolation_forest,
)
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings(action="ignore")
from utils import downcast_df_int_columns, downcast_df_float_columns

features = pd.read_csv("data/features.csv")
power = pd.read_csv("data/power.csv")
sample_submission = pd.read_csv("data/sample_submission.csv")

all = pd.merge(features, power, on="Timestamp", how="left")
all["is_train"] = all["Power(kW)"].notnull()

power_offset = all["Power(kW)"].min()
power_max = all["Power(kW)"].max()
all["Power(kW)"] = all["Power(kW)"] - power_offset
# all['power_log'] = np.log1p(all['Power(kW)'])

all.replace(to_replace=99999, value=np.nan, inplace=True)

all = downcast_df_int_columns(all)
all = downcast_df_float_columns(all)

# all = all.loc[(all['Gearbox_T1_High_Speed_Shaft_Temperature'] > 20) | (all.is_train == 0)]
# all = all.loc[(all['Gearbox_T3_High_Speed_Shaft_Temperature'] >= 30) | (all.is_train == 0)]
# all = all.loc[(all['Gearbox_Oil-2_Temperature'] >= 25) | (all.is_train == 0)]
# all = all.loc[(all['Gearbox_Oil-1_Temperature'] >= 25) | (all.is_train == 0)]
# all = all.loc[(all['Gearbox_Distributor_Temperature'] >= 25) | (all.is_train == 0)]
# all = all.loc[(all['Moment D Filtered'] >= -500) | (all.is_train == 0)]
# all = all.loc[(all['Moment D Direction'] >= -500) | (all.is_train == 0)]
# all = all.loc[(all['Operating State'] >= 11) | (all.is_train == 0)]
# all = all.loc[(all['Voltage A-N'] >= 370) | (all.is_train == 0)]
# all = all.loc[(all['Voltage C-N'] >= 370) | (all.is_train == 0)]
# all = all.loc[(all['Voltage B-N'] >= 370) | (all.is_train == 0)]
# all = all.loc[(all['Temperature Axis Box-3'] >= 5) | (all.is_train == 0)]
# all = all.loc[(all['Temperature Axis Box-2'] >= 5) | (all.is_train == 0)]
# all = all.loc[(all['Temperature Axis Box-1'] >= 5) | (all.is_train == 0)]
# all = all.loc[(all['Temperature Battery Box-3'] >= 0) | (all.is_train == 0)]
# all = all.loc[(all['Temperature Battery Box-2'] >= 0) | (all.is_train == 0)]
# all = all.loc[(all['Temperature Battery Box-1'] >= 0) | (all.is_train == 0)]
# all = all.loc[(all['Hydraulic Prepressure'] >= 55) | (all.is_train == 0)]
# all = all.loc[(all['Angle Rotor Position'] < 370) | (all.is_train == 0)]
# all = all.loc[(all['Temperature Tower Base'] < 45) | (all.is_train == 0)]
# all = all.loc[(all['Pitch Offset Tower Feedback'] > -0.006) | (all.is_train == 0)]
# all = all.loc[(all['Line Frequency'] > 48) | (all.is_train == 0)]
# all = all.loc[(all['Tower Accelaration Normal Raw'] > -500) | (all.is_train == 0)]
# all = all.loc[(all['Tower Accelaration Lateral Raw'] < 500) | (all.is_train == 0)]
# all = all.loc[(all['External Power Limit'] >3000) | (all.is_train == 0)]
# all = all.loc[(all['Temperature Ambient'] < 39) | (all.is_train == 0)]
# all = all.loc[(all['Wind Deviation 1 seconds'] < 50) | (all.is_train == 0)]
# all = all.loc[(all['Wind Deviation 10 seconds'] < 50) | (all.is_train == 0)]
# all = all.loc[(all['Wind Deviation 1 seconds'] > -50) | (all.is_train == 0)]
# all = all.loc[(all['Wind Deviation 10 seconds'] > -50) | (all.is_train == 0)]
# all = all.loc[(all['Proxy Sensor_Degree-135'] >5.6) | (all.is_train == 0)]
# all = all.loc[(all['Proxy Sensor_Degree-225'] >5.6) | (all.is_train == 0)]
# all = all.loc[(all['Blade-3 Actual Value_Angle-A'] >-5) | (all.is_train == 0)]
# all = all.loc[(all['Blade-2 Actual Value_Angle-A'] >-5) | (all.is_train == 0)]
# all = all.loc[(all['Blade-1 Actual Value_Angle-A'] >-5) | (all.is_train == 0)]
# all = all.loc[(all['Scope CH 4'] > -5) | (all.is_train == 0)]
# all = all.loc[(all['Torque'].notna()) | (all.is_train == 0)]


lcap_threshold = 10
zero_threshold = 100
all["Timestamp"] = pd.to_datetime(all["Timestamp"])
sample_submission["Timestamp"] = pd.to_datetime(sample_submission["Timestamp"])

all["Internal Power Limit"] = all["Internal Power Limit"].fillna(value=2780)

all["Nacelle Position_Degree_sin"] = np.sin(
    all["Nacelle Position_Degree"] * np.pi / 180
).abs()
all["Nacelle Position_Degree_cos"] = np.cos(
    all["Nacelle Position_Degree"] * np.pi / 180
).abs()
all["Nacelle Revolution_sin"] = np.sin(all["Nacelle Revolution"] * np.pi).abs()
all["Nacelle Revolution_cos"] = np.cos(all["Nacelle Revolution"] * np.pi).abs()

all["operation_12"] = (all["Operating State"] == 12.0).astype("int")
all["turbine_1"] = (all["Turbine State"] == 1.0).astype("int")
all["turbine_2"] = (all["Turbine State"] == 2.0).astype("int")
all["turbine_3"] = (all["Turbine State"] == 3.0).astype("int")
all["turbine_4"] = (all["Turbine State"] == 4.0).astype("int")
all["turbine_5"] = (all["Turbine State"] == 5.0).astype("int")
all["operation_11"] = (all["Operating State"] == 11.0).astype("int")
all["operation_15"] = (all["Operating State"] == 15.0).astype("int")
all["operation_19"] = (all["Operating State"] == 19.0).astype("int")
all["operation_16"] = (all["Operating State"] == 16.0).astype("int")
all["turbine_prev"] = np.nan
all["turbine_next"] = np.nan

all.loc[all["Operating State"] == all["Operating State"].round(), "operation_prev"] = all["Operating State"]
all.loc[all["Operating State"] == all["Operating State"].round(), "operation_next"] = all["Operating State"]
all.loc[all["Operating State"].isna(), "operation_prev"] = 999
all.loc[all["Operating State"].isna(), "operation_next"] = 999
all["operation_prev"] = all["operation_prev"].fillna(method="ffill")
all["operation_next"] = all["operation_next"].fillna(method="bfill")
all["operation_prev"] = all["operation_prev"].astype(int)
all["operation_next"] = all["operation_next"].astype(int)
all["operation_state_transition"] = (all["operation_prev"].astype(str) + '_' + all["operation_next"].astype(str)).astype('category')
# all.loc[all["operation_prev"] == 999, "operation_state_transition"] = '999_999'
# all.loc[all["operation_next"] == 999, "operation_state_transition"] = '999_999'
all["operation_stationary"] = all["operation_prev"] == all["operation_next"]

all.loc[all["Turbine State"] == all["Turbine State"].round(), "turbine_prev"] = all["Turbine State"]
all.loc[all["Turbine State"] == all["Turbine State"].round(), "turbine_next"] = all["Turbine State"]
all.loc[all["Turbine State"].isna(), "turbine_prev"] = 999
all.loc[all["Turbine State"].isna(), "turbine_next"] = 999
all["turbine_prev"] = all["turbine_prev"].fillna(method="ffill")
all["turbine_next"] = all["turbine_next"].fillna(method="bfill")
all["turbine_prev"] = all["turbine_prev"].astype(int)
all["turbine_next"] = all["turbine_next"].astype(int)
all["turbine_state_transition"] = (all["turbine_prev"].astype(str) + '_' + all["turbine_next"].astype(str)).astype('category')
# all.loc[all["turbine_prev"] == 999, "turbine_state_transition"] = '999_999'
# all.loc[all["turbine_next"] == 999, "turbine_state_transition"] = '999_999'
all["turbine_stationary"] = all["turbine_prev"] == all["turbine_next"]

all["n-set_1_0"] = (all["N-set 1"] == 0.0).astype("int")
all["n-set_1_1735"] = (all["N-set 1"] == 1735.0).astype("int")
all["n-set_1_limit"] = all["N-set 1"] * 2780 / 1735.0
all["limit_difference"] = all["Internal Power Limit"] - all["Power(kW)"] - power_offset
all["limit_capped"] = (all["limit_difference"] < lcap_threshold).astype("int")
all["pf*reactive"] = all["Power Factor"] * all["Reactive Power"]
all["pf_abs"] = all["Power Factor"].abs()
all["pf_sign"] = (all["Power Factor"] > 0).astype(int)
all["torque_in_range"] = (all["Torque"].abs() < 100).astype(int)
all["Angle Rotor Position_2"] = (all["Angle Rotor Position"] - 180).abs()
all.loc[all["Torque"] == 0, "Torque"] = np.nan
all["Torque_abs"] = all["Torque"].abs()
all["Torque_log"] = np.log1p(all["Torque"])
all["Torque_sqrt"] = np.sqrt(all["Torque"])
all["Torque_is_0"] = (all["Torque"] == 0).astype(int)
all["Torque_sign"] = np.sign(all["Torque"])
all["Torque_square"] = np.power(all["Torque"], 2)
all["Torque_lag1"] = all["Torque"].shift(1)
all["Torque_lead1"] = all["Torque"].shift(-1)
all["Torque_sign_w_change"] = 0
all["Torque_sign_d_change"] = 0
all["Torque_sign_change"] = 0
all.loc[(all["Torque_lag1"] > 0) & (all["Torque"] < 0), "Torque_sign_d_change"] = 1
all.loc[(all["Torque_lag1"] > 0) & (all["Torque"] < 0), "Torque_sign_change"] = 1
all.loc[(all["Torque_lag1"] < 0) & (all["Torque"] > 0), "Torque_sign_d_change"] = 1
all.loc[(all["Torque_lag1"] < 0) & (all["Torque"] > 0), "Torque_sign_change"] = 1
all.loc[(all["Torque_lead1"] > 0) & (all["Torque"] < 0), "Torque_sign_w_change"] = 1
all.loc[(all["Torque_lead1"] > 0) & (all["Torque"] < 0), "Torque_sign_change"] = 1
all.loc[(all["Torque_lead1"] < 0) & (all["Torque"] > 0), "Torque_sign_w_change"] = 1
all.loc[(all["Torque_lead1"] < 0) & (all["Torque"] > 0), "Torque_sign_change"] = 1
all.loc[all["Torque_lead1"].isna(), "Torque_sign_w_change"] = np.nan
all.loc[all["Torque_lead1"].isna(), "Torque_sign_change"] = np.nan
all.loc[all["Torque"].isna(), "Torque_sign_d_change"] = np.nan
all.loc[all["Torque"].isna(), "Torque_sign_w_change"] = np.nan
all.loc[all["Torque"].isna(), "Torque_sign_change"] = np.nan
all.loc[all["Torque_lag1"].isna(), "Torque_sign_d_change"] = np.nan
all.loc[all["Torque_lag1"].isna(), "Torque_sign_change"] = np.nan
all.loc[all["Internal Power Limit"] - all["Power(kW)"] < -20, "target_group"] = 2
all.loc[all["Power(kW)"] < 33, "target_group"] = 0
all.loc[(all["Power(kW)"] >= 33) & (all["Internal Power Limit"] - all["Power(kW)"] >= -20), "target_group"] = 1

# features = ['pf_abs',
#             'Torque',
#             'Blade-3 Actual Value_Angle-A',
#             'Scope CH 4',
#             'Pitch Offset-2 Asymmetric Load Controller',
#             'Wind Deviation 1 seconds',
#             'Proxy Sensor_Degree-315',
#             'Gearbox_Oil-2_Temperature',
#              #'Temperature Trafo-2',
#             'Moment D Filtered',
#             'Proxy Sensor_Degree-45',
#             'Tower Acceleration Normal',
#             'Proxy Sensor_Degree-135',
#             'Gearbox_T1_Intermediate_Speed_Shaft_Temperature',
#             'Gearbox_T1_High_Speed_Shaft_Temperature',
#             'Converter Control Unit Voltage',
#             'Pitch Offset Tower Feedback',
#             'Pitch Demand Baseline_Degree',
#             'Tower Acceleration Lateral',
#             'Temperature Trafo-3',
#             'Operating State',
#             'N-set 1',
#             'Blade-1 Set Value_Degree',
#             'n-set_1_0',
#             'pf*reactive',
#             'Turbine State',
#             'Temperature Bearing_A',
#             'Blade-2 Set Value_Degree',
#             'Gearbox_T3_High_Speed_Shaft_Temperature',
#             'Gearbox_Oil-1_Temperature']

features = [
    "Gearbox_T1_High_Speed_Shaft_Temperature",
    "Gearbox_T3_High_Speed_Shaft_Temperature",
    "Gearbox_T1_Intermediate_Speed_Shaft_Temperature",
    "Temperature Gearbox Bearing Hollow Shaft",
    "Tower Acceleration Normal",
    "Gearbox_Oil-2_Temperature",
    "Tower Acceleration Lateral",
    "Temperature Bearing_A",
    "Temperature Trafo-3",
    "Gearbox_T3_Intermediate_Speed_Shaft_Temperature",
    "Gearbox_Oil-1_Temperature",
    "Gearbox_Oil_Temperature",
    "Torque",
    "Converter Control Unit Reactive Power",
    "Temperature Trafo-2",
    "Reactive Power",
    "Temperature Shaft Bearing-1",
    "Gearbox_Distributor_Temperature",
    "Moment D Filtered",
    "Moment D Direction",
    "N-set 1",
    "Operating State",
    "Power Factor",
    "Temperature Shaft Bearing-2",
    "Temperature_Nacelle",
    "Voltage A-N",
    "Temperature Axis Box-3",
    "Voltage C-N",
    "Temperature Axis Box-2",
    "Temperature Axis Box-1",
    "Voltage B-N",
    "Nacelle Position_Degree",
    "Converter Control Unit Voltage",
    "Temperature Battery Box-3",
    "Temperature Battery Box-2",
    "Temperature Battery Box-1",
    "Hydraulic Prepressure",
    "Angle Rotor Position",
    "Temperature Tower Base",
    "Pitch Offset-2 Asymmetric Load Controller",
    "Pitch Offset Tower Feedback",
    "Line Frequency",
    "Internal Power Limit",
    "Circuit Breaker cut-ins",
    "Particle Counter",
    "Tower Accelaration Normal Raw",
    "Torque Offset Tower Feedback",
    "External Power Limit",
    "Blade-2 Actual Value_Angle-B",
    "Blade-1 Actual Value_Angle-B",
    "Blade-3 Actual Value_Angle-B",
    "Temperature Heat Exchanger Converter Control Unit",
    "Tower Accelaration Lateral Raw",
    "Temperature Ambient",
    "Nacelle Revolution",
    "Pitch Offset-1 Asymmetric Load Controller",
    "Tower Deflection",
    "Pitch Offset-3 Asymmetric Load Controller",
    "Wind Deviation 1 seconds",
    "Wind Deviation 10 seconds",
    "Proxy Sensor_Degree-135",
    "State and Fault",
    "Proxy Sensor_Degree-225",
    "Blade-3 Actual Value_Angle-A",
    "Scope CH 4",
    "Blade-2 Actual Value_Angle-A",
    "Blade-1 Actual Value_Angle-A",
    "Blade-2 Set Value_Degree",
    "Pitch Demand Baseline_Degree",
    "Blade-1 Set Value_Degree",
    "Blade-3 Set Value_Degree",
    "Moment Q Direction",
    "Moment Q Filltered",
    "Proxy Sensor_Degree-45",
    "Turbine State",
    "Proxy Sensor_Degree-315",
    "Nacelle Position_Degree_sin",
    "Nacelle Position_Degree_cos",
    "Nacelle Revolution_sin",
    "Nacelle Revolution_cos",
    "operation_12",
    "operation_11",
    "operation_15",
    "operation_19",
    "operation_16",
    "n-set_1_0",
    "pf*reactive",
    "pf_abs",
    "pf_sign",
    "torque_in_range",
    "Angle Rotor Position_2",
    "Torque_log",
    "Torque_sqrt",
    "Torque_square",
    "Torque_sign",
]

target_feature = "Power(kW)"
submission_feature = "Power(kW)"

train_x = all[all.is_train == 1][features]
train_y = all[all.is_train == 1][target_feature]
test_x = all[all.is_train == 0][features]

if isolation_forest:
    model = IsolationForest(
        n_estimators=500,
        max_samples="auto",
        contamination=float(0.002),
        max_features=1.0,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)
    train_x["preds"] = model.predict(train_x)
    drop_contaminated_indexes = train_x[train_x.preds == -1].index
    all = all.drop(index=drop_contaminated_indexes)

all.replace(to_replace=99999, value=np.nan, inplace=True)


def prep(df, features):
    new_features = features.copy()
    for feature in features:
        df[feature + "_lag1"] = df[feature].shift(1)
        df[feature + "_lag2"] = df[feature].shift(2)
        df[feature + "_lag1day"] = df[feature].shift(24 * 6)
        df[feature + "_lead1"] = df[feature].shift(-1)
        df[feature + "_diff1"] = df[feature] - df[feature + "_lag1"]
        df[feature + "_rolling2"] = df[feature].rolling(2).mean()
        df[feature + "_rolling3"] = df[feature].rolling(3).mean()
        df[feature + "_rolling6"] = df[feature].rolling(6).mean()
        new_features.append(feature + "_lag1")
        new_features.append(feature + "_lead1")
        new_features.append(feature + "_diff1")
        new_features.append(feature + "_rolling2")
        new_features.append(feature + "_rolling3")
        new_features.append(feature + "_rolling6")
        new_features.append(feature + "_lag1day")

    return df, new_features


all, new_features = prep(all, features)

features_cat = [
    "pf_abs",
    "Torque",
    "Torque_lag1",
    "Torque_lag2",
    "Torque_diff1",
    "Torque_lead1",
    "Torque_rolling2",
    "Torque_log",
    "Torque_sqrt",
    "Blade-3 Actual Value_Angle-A",
    "Blade-3 Actual Value_Angle-A_lag1",
    "Blade-3 Actual Value_Angle-A_rolling2",
    "Scope CH 4",
    "Scope CH 4_rolling2",
    "Scope CH 4_lead1",
    "Pitch Offset-2 Asymmetric Load Controller",
    "Wind Deviation 1 seconds",
    "Proxy Sensor_Degree-315",
    "Proxy Sensor_Degree-315_rolling2",
    "Gearbox_Oil-2_Temperature",
    "Gearbox_Oil-2_Temperature_lead1",
    # #'Temperature Trafo-2',
    # 'Moment D Filtered',
    # 'Proxy Sensor_Degree-45',
    # 'Proxy Sensor_Degree-135',
    "Gearbox_T1_Intermediate_Speed_Shaft_Temperature",
    "Gearbox_T1_Intermediate_Speed_Shaft_Temperature_lead1",
    "Gearbox_T1_High_Speed_Shaft_Temperature",
    "Gearbox_T1_High_Speed_Shaft_Temperature_lead1",
    # 'Converter Control Unit Voltage',
    "Pitch Offset Tower Feedback",
    # 'Pitch Offset Tower Feedback_diff1',
    "Pitch Demand Baseline_Degree",
    "Pitch Demand Baseline_Degree_lag1",
    "Pitch Demand Baseline_Degree_rolling2",
    "Tower Acceleration Lateral",
    "Tower Acceleration Normal",
    "Tower Acceleration Normal_rolling2",
    # #'Temperature Trafo-3',
    "Operating State",
    "operation_11",
    "operation_12",
    "operation_15",
    "operation_16",
    "operation_19",
    "Operating State_rolling2",
    "Operating State_lead1",
    "N-set 1",
    "N-set 1_rolling2",
    "Blade-1 Set Value_Degree",
    "Blade-1 Set Value_Degree_lag1",
    "n-set_1_0",
    "n-set_1_1735",
    "n-set_1_0_lag1",
    "n-set_1_0_rolling2",
    "pf*reactive",
    "Turbine State",
    "Turbine State_rolling2",
    #'hour',
    "torque_in_range",
    "Temperature Bearing_A_lead1",
    "Blade-2 Set Value_Degree",
    "Blade-2 Set Value_Degree_rolling2",
    "Gearbox_T3_High_Speed_Shaft_Temperature",
    "Gearbox_Oil-1_Temperature",
    "Gearbox_Oil-1_Temperature_lead1",
    # #'Nacelle Revolution_cos',
]

features_xgb = [
            'pf_abs',
            'Torque',
            'Torque_lag1',
            'Torque_lead1',
            'Torque_rolling2',
            'Torque_log',
            'Torque_sqrt',
             #'Torque_abs',
            'Torque_is_0',
             # 'Torque_sign_change',
             # 'Torque_sign_d_change',
             # 'Torque_sign_w_change',
            'Blade-3 Actual Value_Angle-A',
            'Blade-3 Actual Value_Angle-A_rolling2',
            'Scope CH 4',
            'Pitch Offset-2 Asymmetric Load Controller',
            'Wind Deviation 1 seconds',
            'Proxy Sensor_Degree-315',
            'Gearbox_Oil-2_Temperature',
            'Gearbox_Oil-2_Temperature_lead1',
            # #'Temperature Trafo-2',
            # 'Moment D Filtered',
            # 'Proxy Sensor_Degree-45',
            # 'Proxy Sensor_Degree-135',
            'Gearbox_T1_Intermediate_Speed_Shaft_Temperature',
            'Gearbox_T1_Intermediate_Speed_Shaft_Temperature_lead1',
            'Gearbox_T1_High_Speed_Shaft_Temperature',
            'Gearbox_T1_High_Speed_Shaft_Temperature_lead1',
            # 'Converter Control Unit Voltage',
            'Pitch Offset Tower Feedback',
            # 'Pitch Offset Tower Feedback_diff1',
            'Pitch Demand Baseline_Degree',
            'Tower Acceleration Lateral',
            'Tower Acceleration Normal',
            'Tower Acceleration Normal_rolling2',
            # #'Temperature Trafo-3',
            'Operating State',
            "operation_11",
            "operation_12",
            "operation_15",
            "operation_16",
            "operation_19",
             #"operation_state_transition",
             #"turbine_state_transition",
            "turbine_stationary",
            "operation_stationary",
            "turbine_1",
            "turbine_2",
            "turbine_3",
            "turbine_4",
            "turbine_5",
            'Operating State_rolling2',
            'Operating State_lead1',
            'N-set 1',
            'N-set 1_rolling2',
            'Blade-1 Set Value_Degree',
            'n-set_1_0',
            'n-set_1_1735',
            'n-set_1_0_lag1',
            'n-set_1_0_rolling2',
            'pf*reactive',
            'Turbine State',
            'Turbine State_rolling2',
             #'hour',
            'torque_in_range',
            'Temperature Bearing_A_lead1',
            'Blade-2 Set Value_Degree',
            'Gearbox_T3_High_Speed_Shaft_Temperature',
            'Gearbox_Oil-1_Temperature',
            # #'Nacelle Revolution_cos',
            ]

# features_lgbm = [
#     'Tower Accelaration Lateral Raw',
#     'Torque Offset Tower Feedback_lag1day',
#     'Gearbox_Distributor_Temperature_diff1',
#     'Tower Accelaration Normal Raw_rolling2',
#     'Line Frequency_lead1',
#     'pf*reactive',
#     'Temperature Battery Box-2_diff1',
#     'Tower Accelaration Lateral Raw_lag1day',
#     'Pitch Offset Tower Feedback_rolling3',
#     'Tower Accelaration Normal Raw_lag1day',
#     'Tower Accelaration Lateral Raw_rolling3',
#     'Torque Offset Tower Feedback_rolling3',
#     'Angle Rotor Position',
#     'Tower Accelaration Lateral Raw_lag1',
#     'Torque Offset Tower Feedback',
#     'Torque Offset Tower Feedback_rolling2',
#     'Temperature Bearing_A_diff1',
#     'Reactive Power_diff1',
#     'Line Frequency_lag1day',
#     'Temperature Trafo-2_diff1',
#     'Torque Offset Tower Feedback_rolling6',
#     'Moment Q Direction',
#     'Tower Accelaration Lateral Raw_diff1',
#     'Voltage B-N_diff1',
#     'Tower Accelaration Normal Raw_lag1',
#     'Tower Accelaration Normal Raw',
#     'Gearbox_T3_High_Speed_Shaft_Temperature_diff1',
#     'Angle Rotor Position_2_lead1',
#     'Moment Q Filltered',
#     'Tower Accelaration Lateral Raw_rolling2',
#     'Tower Acceleration Normal_lead1',
#     'Moment D Filtered_lead1',
#     'Tower Accelaration Normal Raw_diff1',
#     'Temperature Tower Base_diff1',
#     'Moment D Filtered_diff1',
#     'Nacelle Revolution_sin_diff1',
#     'Converter Control Unit Reactive Power_diff1',
#     'Moment D Direction_diff1',
#     'Angle Rotor Position_lead1',
#     'Angle Rotor Position_2_diff1',
#     'pf*reactive_diff1',
#     'Angle Rotor Position_diff1',
#     'Pitch Demand Baseline_Degree',
#     'Hydraulic Prepressure_diff1',
#     'Nacelle Position_Degree_lag1',
#     'Tower Accelaration Lateral Raw_rolling6',
#     'Proxy Sensor_Degree-135_lead1',
#     'Temperature Shaft Bearing-1_diff1',
#     'Pitch Offset Tower Feedback_rolling6',
#     'Pitch Offset Tower Feedback_rolling2',
#     'Tower Accelaration Lateral Raw_lead1',
#     'Tower Acceleration Lateral',
#     'Tower Accelaration Normal Raw_rolling6',
#     'Nacelle Position_Degree_diff1',
#     'Pitch Offset Tower Feedback_lead1',
#     'Temperature Battery Box-3_diff1',
#     'Nacelle Revolution_diff1',
#     'Temperature Axis Box-3_diff1',
#     'Tower Acceleration Normal',
#     'Torque_rolling3',
#     'Temperature Battery Box-1_diff1',
#     'Torque_log_lead1',
#     'Moment Q Direction_diff1',
#     'Moment Q Filltered_diff1',
#     'pf_abs_diff1',
#     'Proxy Sensor_Degree-315_lead1',
#     'Proxy Sensor_Degree-45_diff1',
#     'Tower Acceleration Normal_diff1',
#     'Operating State_diff1',
#     'Proxy Sensor_Degree-135_diff1',
#     'Proxy Sensor_Degree-315',
#     'Torque_log_rolling2',
#     'Temperature Shaft Bearing-2_diff1',
#     'Tower Acceleration Lateral_diff1',
#     'Turbine State_diff1',
#     'Voltage A-N_diff1',
#     'operation_11_diff1',
#     'Converter Control Unit Voltage_diff1',
#     'Gearbox_T1_High_Speed_Shaft_Temperature_diff1',
#     'Pitch Offset Tower Feedback_diff1',
#     'Torque_lag1',
#     'Gearbox_T1_Intermediate_Speed_Shaft_Temperature_diff1',
#     'Tower Accelaration Normal Raw_lead1',
#     'Temperature Ambient_diff1',
#     'Proxy Sensor_Degree-315_diff1',
#     'Torque_rolling2',
#     'Torque_log_diff1',
#     'Torque_lead1',
#     'Proxy Sensor_Degree-225_diff1',
#     'State and Fault_diff1',
#     'Internal Power Limit_diff1',
#     'N-set 1_diff1',
#     'Pitch Offset Tower Feedback',
#     'Circuit Breaker cut-ins_diff1',
#     'Temperature Trafo-3_diff1',
#     'Temperature Heat Exchanger Converter Control Unit_diff1',
#     'Torque_log',
#     'Torque_diff1',
#     'Temperature Axis Box-2_diff1',
#     'Torque'
# ]

features_lgbm = [
    "pf_abs",
    "Torque",
    "Torque_lag1",
    "Torque_lead1",
    "Torque_rolling2",
    "Torque_log",
    "Blade-3 Actual Value_Angle-A",
    "Blade-3 Actual Value_Angle-A_rolling2",
    "Scope CH 4",
    "Pitch Offset-2 Asymmetric Load Controller",
    "Wind Deviation 1 seconds",
    "Proxy Sensor_Degree-315",
    "Gearbox_Oil-2_Temperature",
    "Gearbox_Oil-2_Temperature_lead1",
    # #'Temperature Trafo-2',
    # 'Moment D Filtered',
    # 'Proxy Sensor_Degree-45',
    # 'Proxy Sensor_Degree-135',
    "Gearbox_T1_Intermediate_Speed_Shaft_Temperature",
    "Gearbox_T1_Intermediate_Speed_Shaft_Temperature_lead1",
    "Gearbox_T1_High_Speed_Shaft_Temperature",
    "Gearbox_T1_High_Speed_Shaft_Temperature_lead1",
    # 'Converter Control Unit Voltage',
    "Pitch Offset Tower Feedback",
    # 'Pitch Offset Tower Feedback_diff1',
    "Pitch Demand Baseline_Degree",
    "Tower Acceleration Lateral",
    "Tower Acceleration Normal",
    "Tower Acceleration Normal_rolling2",
    # #'Temperature Trafo-3',
    "Operating State",
    "operation_16",
    "Operating State_rolling2",
    "Operating State_lead1",
    "N-set 1",
    "N-set 1_rolling2",
    "Blade-1 Set Value_Degree",
    "n-set_1_0",
    "n-set_1_0_lag1",
    "n-set_1_0_rolling2",
    "pf*reactive",
    "Turbine State",
    "Turbine State_rolling2",
    "torque_in_range",
    "Temperature Bearing_A_lead1",
    "Blade-2 Set Value_Degree",
    "Gearbox_T3_High_Speed_Shaft_Temperature",
    "Gearbox_Oil-1_Temperature",
    # #'Nacelle Revolution_cos',
]


if model_type == "lgbm":
    new_features = features_lgbm
elif model_type == "xgb":
    new_features = features_xgb
elif model_type == "cat":
    new_features = features_cat

print(f"Number of features: {len(new_features)}")

object_cols = [
    "n-set_1_0",
    "n-set_1_1735",
    "n-set_1_0_lag1",
    "n-set_1_0_rolling2",
    "Torque_sign_d_change",
    "Torque_sign_w_change",
    "Torque_sign_change",
    "Torque_is_0",
    "torque_in_range",
    "operation_11",
    "operation_11_diff1",
    "operation_12",
    "operation_15",
    "operation_16",
    "operation_19",
    "turbine_state_transition",
    "operation_state_transition"
    "turbine_stationary",
    "operation_stationary",
]
numeric_cols = list(set(all[new_features].columns) - set(object_cols))
print("numeric_cols: ", numeric_cols)
print("object_cols: ", object_cols)

if use_scaler:
    sc = StandardScaler()
    all[numeric_cols] = sc.fit_transform(all[numeric_cols])


def objective_xgb(trial):

    # To select which parameters to optimize, please look at the XGBoost documentation:
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    pruning_callback = optuna.integration.XGBoostPruningCallback(
        trial, observation_key="validation_0-rmse"
    )
    param = {
        "tree_method": "gpu_hist",  # Use GPU acceleration
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 1e2),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 1e2),
        "colsample_bytree": trial.suggest_categorical(
            "colsample_bytree", [0.8, 0.9, 1.0]
        ),
        "subsample": trial.suggest_categorical("subsample", [0.6, 0.7, 0.8, 1.0]),
        "learning_rate": trial.suggest_float("learning_rate", 0.002, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 2000, 6000, 400),
        "max_depth": trial.suggest_int("max_depth", 8, 15),
        "random_state": 42,
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 0.1, 10),
        "objective": "reg:squarederror",
        "enable_categorical": True,
    }
    model = XGBRegressor(**param)

    N_FOLDS = 5

    kf = KFold(N_FOLDS, shuffle=True, random_state=22)
    y_oof = np.zeros(train_x.shape[0])

    ix = 0
    for train_ind, val_ind in kf.split(train_x):
        print(f"******* Fold {ix} ******* ")
        tr_x, val_x = (
            train_x.iloc[train_ind].reset_index(drop=True),
            train_x.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            train_y.iloc[train_ind].reset_index(drop=True),
            train_y.iloc[val_ind].reset_index(drop=True),
        )

        model.fit(
            tr_x,
            tr_y,
            eval_set=[(val_x, val_y)],
            eval_metric="rmse",
            early_stopping_rounds=100,
            callbacks=[pruning_callback],
            verbose=100,
        )

        preds = model.predict(val_x)
        y_oof[val_ind] = y_oof[val_ind] + preds
        ix = ix + 1

    rmse = mse(train_y, y_oof, squared=False)

    return rmse


def objective_lgbm(trial):

    # To select which parameters to optimize, please look at the XGBoost documentation:
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, metric="rmse")
    param = {
        "metric": "rmse",
        "random_state": 42,
        "n_estimators": 5000,
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
        "colsample_bytree": trial.suggest_categorical(
            "colsample_bytree", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        "subsample": trial.suggest_categorical(
            "subsample", [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
        ),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [0.006, 0.008, 0.01, 0.014, 0.017, 0.02]
        ),
        "max_depth": trial.suggest_categorical("max_depth", [10, 20, 100]),
        "num_leaves": trial.suggest_int("num_leaves", 1, 1000),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 300),
    }
    model = LGBMRegressor(**param)

    N_FOLDS = 5

    kf = KFold(N_FOLDS, shuffle=True, random_state=22)
    y_oof = np.zeros(train_x.shape[0])

    ix = 0
    for train_ind, val_ind in kf.split(train_x):
        print(f"******* Fold {ix} ******* ")
        tr_x, val_x = (
            train_x.iloc[train_ind].reset_index(drop=True),
            train_x.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            train_y.iloc[train_ind].reset_index(drop=True),
            train_y.iloc[val_ind].reset_index(drop=True),
        )

        model.fit(
            tr_x,
            tr_y,
            eval_set=[(val_x, val_y)],
            eval_metric="rmse",
            early_stopping_rounds=100,
            callbacks=[pruning_callback],
            verbose=100,
        )

        preds = model.predict(val_x)
        y_oof[val_ind] = y_oof[val_ind] + preds
        ix = ix + 1

    rmse = mse(train_y, y_oof, squared=False)

    return rmse


def objective_cat(trial):

    # To select which parameters to optimize, please look at the XGBoost documentation:
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    param = {
        "iterations": 2000,
        "objective": "RMSE",
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical(
            "boosting_type", ["Ordered", "Plain"]
        ),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)
    model = CatBoostRegressor(**param)

    N_FOLDS = 5

    kf = KFold(N_FOLDS, shuffle=True, random_state=22)
    y_oof = np.zeros(train_x.shape[0])

    ix = 0
    for train_ind, val_ind in kf.split(train_x):
        print(f"******* Fold {ix} ******* ")
        tr_x, val_x = (
            train_x.iloc[train_ind].reset_index(drop=True),
            train_x.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            train_y.iloc[train_ind].reset_index(drop=True),
            train_y.iloc[val_ind].reset_index(drop=True),
        )

        model.fit(
            tr_x,
            tr_y,
            eval_set=[(val_x, val_y)],
            early_stopping_rounds=100,
            verbose=100,
        )

        preds = model.predict(val_x)
        y_oof[val_ind] = y_oof[val_ind] + preds
        ix = ix + 1

    rmse = mse(train_y, y_oof, squared=False)

    return rmse


def objective_et(trial):

    param = {
        "ccp_alpha": trial.suggest_loguniform("ccp_alpha", 1e-3, 0.1),
        "max_depth": trial.suggest_int("max_depth", 30, 100),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 1000),
        "max_features": trial.suggest_uniform("max_features", 0.15, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 14),
    }
    model = ExtraTreeRegressor(**param, random_state=42)

    N_FOLDS = 5

    kf = KFold(N_FOLDS, shuffle=True, random_state=42)
    y_oof = np.zeros(train_x.shape[0])

    ix = 0
    for train_ind, val_ind in kf.split(train_x):
        print(f"******* Fold {ix} ******* ")
        tr_x, val_x = (
            train_x.iloc[train_ind].reset_index(drop=True),
            train_x.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            train_y.iloc[train_ind].reset_index(drop=True),
            train_y.iloc[val_ind].reset_index(drop=True),
        )

        model.fit(
            tr_x,
            tr_y,
        )

        preds = model.predict(val_x)
        y_oof[val_ind] = y_oof[val_ind] + preds
        ix = ix + 1

    rmse = mse(train_y, y_oof, squared=False)

    return rmse


if run_optuna:
    study = optuna.create_study(direction="minimize")
    if model_type == "xgb":
        study.optimize(objective_xgb, n_trials=optuna_trials)
    elif model_type == "et":
        study.optimize(objective_et, n_trials=optuna_trials)
    elif model_type == "lgbm":
        study.optimize(objective_lgbm, n_trials=optuna_trials)
    elif model_type == "cat":
        study.optimize(objective_cat, n_trials=optuna_trials)
    else:
        study.optimize(objective_xgb, n_trials=optuna_trials)
    print("Number of finished trials:", len(study.trials))
    print("Best trial:", study.best_trial.params)

    best_params = study.best_params

else:
    if model_type == "xgb":
        best_params = {
            "reg_lambda": 10.748677791318165,
            "reg_alpha": 0.35338174927315275,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "learning_rate": 0.06048152407526179,
            "n_estimators": 6000,
            "max_depth": 8,
            "min_child_weight": 4.294149553496176,
        }
        best_params["tree_method"] = "gpu_hist"
        best_params["random_state"] = 42
        best_params["enable_categorical"] = True
    elif model_type == "cat":
        best_params = {
            "colsample_bylevel": 0.08679127245770515,
            "depth": 8,
            "boosting_type": "Ordered",
            "bootstrap_type": "Bernoulli",
            "subsample": 0.33006860711310543,
        }
        best_params["iterations"] = 20000
        best_params["objective"] = "RMSE"
    elif model_type == "lgbm":
        best_params = {
            "reg_alpha": 0.06554399897557625,
            "reg_lambda": 0.007645226648294985,
            "colsample_bytree": 0.6,
            "subsample": 0.7,
            "learning_rate": 0.02,
            "max_depth": 20,
            "num_leaves": 133,
            "min_child_samples": 81,
        }
        best_params["metric"] = "rmse"
        best_params["random_state"] = 42
        best_params["n_estimators"] = 20000
    else:
        best_params = {
            "ccp_alpha": 0.004109034886479211,
            "max_depth": 31,
            "min_samples_split": 2,
            "max_leaf_nodes": 986,
            "max_features": 0.9928403752860953,
            "min_samples_leaf": 1,
        }
        best_params["iterations"] = 10000
        best_params["objective"] = "RMSE"


## Target group classifications

# bonus_feats = ["operation_11", "operation_15", "operation_19", "operation_16"]
# classification_feats = new_features.copy()
# for f in bonus_feats:
#     classification_feats.append(f)
# classification_feats = list(set(classification_feats))
# classification_feature = "target_group"
# train_x = all[all.is_train == 1][classification_feats]
# train_y = all[all.is_train == 1][classification_feature]
# test_x = all[all.is_train == 0][classification_feats]
#
# N_FOLDS = 5
#
# skf = StratifiedKFold(N_FOLDS, random_state=42, shuffle=True)
# y_oof = np.zeros(train_x.shape[0])
# y_test = np.zeros(test_x.shape[0])
#
# ix = 0
#
# for train_ind, val_ind in skf.split(train_x, train_y):
#     print(f"******* Fold {ix} ******* ")
#     model_classifier = XGBClassifier(tree_method="gpu_hist")
#     tr_x, val_x = train_x.iloc[train_ind].reset_index(drop=True), train_x.iloc[val_ind].reset_index(drop=True)
#     tr_y, val_y = train_y.iloc[train_ind].reset_index(drop=True), train_y.iloc[val_ind].reset_index(drop=True)
#
#     model_classifier.fit(
#         tr_x,
#         tr_y,
#     )
#
#     preds = model_classifier.predict(val_x)
#     y_oof[val_ind] = y_oof[val_ind] + preds
#
#     if plot_importance:
#         importances = model_classifier.feature_importances_
#         indices = np.argsort(importances)
#         indices = indices[-30:]
#
#         plt.figure(figsize=(20, 10))
#         plt.title('Feature Importances')
#         plt.barh(range(len(indices)), importances[indices], color='b', align='center')
#         plt.yticks(range(len(indices)), [classification_feats[i] for i in indices])
#         plt.xlabel('Relative Importance')
#         plt.show()
#
#     test_preds = model_classifier.predict(test_x)
#     y_test = y_test + test_preds / N_FOLDS
#     ix = ix + 1
#
#
# auc = accuracy_score(train_y, y_oof)
# print(f"Classification val Score: {auc}")
#
#
# all.loc[all.is_train == 0, classification_feature] = y_test
# all.loc[all[classification_feature] < 0.5, classification_feature] = 0
# all.loc[(all[classification_feature] > 0.5) & (all[classification_feature] < 1.5), classification_feature] = 1
# all.loc[all[classification_feature] > 1.5, classification_feature] = 2
#
# limit_capped_pred = np.count_nonzero(all.loc[all.is_train == 0][classification_feature] == 1)
# print(f"Classification l-capped count: {limit_capped_pred}")
# new_features.append(classification_feature)

## Target group classifications

# =====================================================================
train_x = all[all.is_train == 1][new_features]
train_y = all[all.is_train == 1][target_feature]
test_x = all[all.is_train == 0][new_features]

N_FOLDS = 5

kf = KFold(N_FOLDS, shuffle=True, random_state=42)
y_oof = np.zeros(train_x.shape[0])
y_test = np.zeros(test_x.shape[0])

ix = 0

if model_type == "xgb":
    for train_ind, val_ind in kf.split(train_x):
        print(f"******* Fold {ix} ******* ")
        model = XGBRegressor(**(best_params))
        tr_x, val_x = (
            train_x.iloc[train_ind].reset_index(drop=True),
            train_x.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            train_y.iloc[train_ind].reset_index(drop=True),
            train_y.iloc[val_ind].reset_index(drop=True),
        )

        model.fit(
            tr_x,
            tr_y,
            eval_set=[(val_x, val_y)],
            eval_metric="rmse",
            early_stopping_rounds=200,
            verbose=100,
        )

        if plot_importance:
            importances = model.feature_importances_
            indices = np.argsort(importances)
            indices = indices[-100:]
            for indice in indices:
                print(f"'{new_features[indice]}',")

            plt.figure(figsize=(20, 10))
            plt.title("Feature Importances")
            plt.barh(
                range(len(indices)), importances[indices], color="b", align="center"
            )
            plt.yticks(range(len(indices)), [new_features[i] for i in indices])
            plt.xlabel("Relative Importance")
            plt.show()

        preds = model.predict(val_x)
        y_oof[val_ind] = y_oof[val_ind] + preds

        test_preds = model.predict(test_x)
        y_test = y_test + test_preds / N_FOLDS
        ix = ix + 1

    plt.figure(figsize=(32, 12))
    sns.distplot(train_y, hist=True, color="blue", kde=True, bins=30, label="Power(kW)")
    sns.distplot(y_oof, hist=True, color="red", kde=True, bins=30, label="predictions")
    plt.legend()
    plt.show()
    rmse = mse(train_y, y_oof, squared=False)

    train_x["pred"] = y_oof
    train_x["power"] = train_y
    train_x["pred_diff"] = (train_y - y_oof).abs()
    train_x.sort_values(by="pred_diff", ascending=False)
    train_x.to_csv(
        f"fuck_up_xgb.csv", sep=",", index=False
    )

    rmse_updated = mse(train_x[10:]['power'], train_x[10:]['pred'], squared=False)
    print(f"Val Score worst 10 removed: {rmse_updated}")

    plt.figure(figsize=(32, 12))
    sns.distplot(
        train_x[:100]["power"],
        hist=True,
        color="blue",
        kde=True,
        bins=30,
        label="Power(kW)",
    )
    sns.distplot(
        train_x[:100]["pred"],
        hist=True,
        color="red",
        kde=True,
        bins=30,
        label="predictions",
    )
    plt.legend()
    plt.show()

    sns.scatterplot(x=train_x["pred"], y=train_x["power"], hue=train_x['Torque'].isna())
    plt.show()

    sns.scatterplot(x=train_x["pred_diff"], y=train_x["power"], hue=train_x['Torque'].isna())
    plt.show()
    print(f"Val Score: {rmse}")

elif model_type == "lgbm":
    for train_ind, val_ind in kf.split(train_x):
        print(f"******* Fold {ix} ******* ")
        model = LGBMRegressor(**(best_params))
        tr_x, val_x = (
            train_x.iloc[train_ind].reset_index(drop=True),
            train_x.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            train_y.iloc[train_ind].reset_index(drop=True),
            train_y.iloc[val_ind].reset_index(drop=True),
        )

        model.fit(
            tr_x,
            tr_y,
            eval_set=[(val_x, val_y)],
            eval_metric="rmse",
            early_stopping_rounds=200,
            verbose=100,
        )

        if plot_importance:
            importances = model.feature_importances_
            indices = np.argsort(importances)
            indices = indices[-200:]
            for indice in indices:
                print(f"'{new_features[indice]}'")

            plt.figure(figsize=(20, 10))
            plt.title("Feature Importances")
            plt.barh(
                range(len(indices)), importances[indices], color="b", align="center"
            )
            plt.yticks(range(len(indices)), [new_features[i] for i in indices])
            plt.xlabel("Relative Importance")
            plt.show()

        preds = model.predict(val_x)
        y_oof[val_ind] = y_oof[val_ind] + preds

        test_preds = model.predict(test_x)
        y_test = y_test + test_preds / N_FOLDS
        ix = ix + 1

    plt.figure(figsize=(32, 12))
    sns.distplot(train_y, hist=True, color="blue", kde=True, bins=30, label="Power(kW)")
    sns.distplot(y_oof, hist=True, color="red", kde=True, bins=30, label="predictions")
    plt.legend()
    plt.show()
    rmse = mse(train_y, y_oof, squared=False)

    train_x["pred"] = y_oof
    train_x["power"] = train_y
    train_x["pred_diff"] = (train_y - y_oof).abs()
    train_x.sort_values(by="pred_diff", ascending=False)
    train_x.to_csv(
        f"fuck_up_{trial}.csv", sep=",", index=False
    )

    rmse_updated = mse(train_x[10:]['power'], train_x[10:]['pred'], squared=False)
    print(f"Val Score worst 10 removed: {rmse_updated}")

    plt.figure(figsize=(32, 12))
    sns.distplot(
        train_x[:100]["power"],
        hist=True,
        color="blue",
        kde=True,
        bins=30,
        label="Power(kW)",
    )
    sns.distplot(
        train_x[:100]["pred"],
        hist=True,
        color="red",
        kde=True,
        bins=30,
        label="predictions",
    )
    plt.legend()
    plt.show()

    sns.scatterplot(x=train_x["pred"], y=train_x["power"], hue=train_x['Torque'].isna())
    plt.show()
    sns.scatterplot(x=train_x["pred"], y=train_x["power"], hue=train_x['target_group'])
    plt.show()

    sns.scatterplot(x=train_x["pred_diff"], y=train_x["power"], hue=train_x['Torque'].isna())
    plt.show()
    sns.scatterplot(x=train_x["pred_diff"], y=train_x["power"], hue=train_x['target_group'])
    plt.show()
    print(f"Val Score: {rmse}")

elif model_type == "cat":
    for train_ind, val_ind in kf.split(train_x):
        print(f"******* Fold {ix} ******* ")
        model = CatBoostRegressor(**(best_params))
        tr_x, val_x = (
            train_x.iloc[train_ind].reset_index(drop=True),
            train_x.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            train_y.iloc[train_ind].reset_index(drop=True),
            train_y.iloc[val_ind].reset_index(drop=True),
        )

        model.fit(
            tr_x,
            tr_y,
            eval_set=[(val_x, val_y)],
            early_stopping_rounds=200,
            verbose=100,
        )

        if plot_importance:
            importances = model.feature_importances_
            indices = np.argsort(importances)
            indices = indices[-50:]

            plt.figure(figsize=(20, 10))
            plt.title("Feature Importances")
            plt.barh(
                range(len(indices)), importances[indices], color="b", align="center"
            )
            plt.yticks(range(len(indices)), [new_features[i] for i in indices])
            plt.xlabel("Relative Importance")
            plt.show()

        preds = model.predict(val_x)
        y_oof[val_ind] = y_oof[val_ind] + preds

        test_preds = model.predict(test_x)
        y_test = y_test + test_preds / N_FOLDS
        ix = ix + 1

    plt.figure(figsize=(32, 12))
    sns.distplot(train_y, hist=True, color="blue", kde=True, bins=30, label="Power(kW)")
    sns.distplot(y_oof, hist=True, color="red", kde=True, bins=30, label="predictions")
    plt.legend()
    plt.show()
    rmse = mse(train_y, y_oof, squared=False)

    train_x["pred"] = y_oof
    train_x["power"] = train_y
    train_x["pred_diff"] = (train_y - y_oof).abs()
    train_x.sort_values(by="pred_diff", ascending=False)

    plt.figure(figsize=(32, 12))
    sns.distplot(
        train_x[:200]["power"],
        hist=True,
        color="blue",
        kde=True,
        bins=30,
        label="Power(kW)",
    )
    sns.distplot(
        train_x[:200]["pred"],
        hist=True,
        color="red",
        kde=True,
        bins=30,
        label="predictions",
    )
    plt.legend()
    plt.show()

else:
    for train_ind, val_ind in kf.split(train_x):
        print(f"******* Fold {ix} ******* ")
        model = ExtraTreeRegressor(random_state=42, **best_params)
        tr_x, val_x = (
            train_x.iloc[train_ind].reset_index(drop=True),
            train_x.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            train_y.iloc[train_ind].reset_index(drop=True),
            train_y.iloc[val_ind].reset_index(drop=True),
        )

        model.fit(tr_x, tr_y)

        if plot_importance:
            importances = model.feature_importances_
            indices = np.argsort(importances)
            indices = indices[-30:]
            for i in indices[-20:]:
                print(new_features[i])

            plt.figure(figsize=(20, 10))
            plt.title("Feature Importances")
            plt.barh(
                range(len(indices)), importances[indices], color="b", align="center"
            )
            plt.yticks(range(len(indices)), [new_features[i] for i in indices])
            plt.xlabel("Relative Importance")
            plt.show()

        preds = model.predict(val_x)
        y_oof[val_ind] = y_oof[val_ind] + preds

        test_preds = model.predict(test_x)
        y_test = y_test + test_preds / N_FOLDS
        ix = ix + 1


sample_submission[submission_feature] = y_test + power_offset
sample_submission = pd.merge(
    sample_submission, all.drop(submission_feature, axis=1), on="Timestamp", how="left"
)

# sample_submission[sample_submission[classification_feature] == 1][submission_feature] = np.clip(sample_submission[sample_submission[classification_feature] == 1][submission_feature], sample_submission[sample_submission[classification_feature] == 1]['Internal Power Limit'] - lcap_threshold, sample_submission[sample_submission[classification_feature] == 1]['Internal Power Limit'] - 0.57666015625)
sample_submission[submission_feature] = np.clip(
    sample_submission[submission_feature],
    power_offset,
    sample_submission["Internal Power Limit"] - 0.57666015625,
)

# clipped_count = np.count_nonzero(sample_submission[sample_submission.target_group == 2][submission_feature] > sample_submission[sample_submission.target_group == 2]['Internal Power Limit'] - 0.57666015625)
# print(f"1 - {clipped_count} preds are clipped.")
# sample_submission[sample_submission.target_group == 2][submission_feature] = np.clip(sample_submission[sample_submission.target_group == 2][submission_feature], sample_submission[sample_submission.target_group == 2]['Internal Power Limit'] - lcap_threshold, sample_submission[sample_submission.target_group == 2]['Internal Power Limit'] - 0.57666015625)
#
# clipped_count = np.count_nonzero((sample_submission[sample_submission.target_group == 1][submission_feature] > sample_submission[sample_submission.target_group == 1]['Internal Power Limit'] - lcap_threshold) & (sample_submission[sample_submission.target_group == 1].target_group != 2))
# print(f"2 - {clipped_count} preds are clipped.")
# sample_submission[sample_submission.target_group == 1][submission_feature] = np.clip(sample_submission[sample_submission.target_group == 1][submission_feature], zero_threshold + power_offset, sample_submission[sample_submission.target_group == 1]['Internal Power Limit'] - lcap_threshold)
#
# clipped_count = np.count_nonzero(sample_submission[sample_submission.target_group == 0][submission_feature] > zero_threshold + power_offset)
# print(f"3 - {clipped_count} preds are clipped.")
# sample_submission[sample_submission.target_group == 0][submission_feature] = np.clip(sample_submission[sample_submission.target_group == 0][submission_feature], power_offset, zero_threshold + power_offset)


print(f"Submission mean: {sample_submission[submission_feature].mean()}")
sample_submission[["Timestamp", submission_feature]].to_csv(
    f"submissions/submission_{model_type}_{trial}.csv", sep=",", index=False
)
