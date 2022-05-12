import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
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
from tqdm import tqdm
from constants import (
    run_optuna,
    run_adv,
    use_nfolds,
    trial,
    validation_strategy,
    validation_files_index,
    check_val_results,
    run_imputer,
    use_scaler,
    optuna_trials,
    plot_importance,
    model_type,
    isolation_forest,
)
import warnings
import matplotlib.pyplot as plt
from stratified_kfold_reg import StratifiedKFoldReg

warnings.filterwarnings(action="ignore")
from utils import downcast_df_int_columns, downcast_df_float_columns

def analyze_column(input_series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(input_series):
        return 'numeric'
    else:
        return 'categorical'


class LGBMImputer:

    '''
    Regression imputer using LightGBM
    '''

    def __init__(self, cat_features=[], feature_list=[], n_iter=15000, verbose=False):
        self.n_iter = n_iter
        self.cat_features = cat_features
        self.verbose = verbose
        self.n_features = None
        self.feature_names = feature_list
        self.feature_with_missing = None
        self.imputers = {}
        self.offsets = {}
        self.objectives = {}

    def fit_transform(self, X, y=None):
        output_X = X.copy()
        self.n_features = X.shape[1]
        self.feature_with_missing = [col for col in self.feature_names if X[col].isnull().sum() > 0]

        for icol, col in enumerate(self.feature_with_missing):
            if icol in self.cat_features:
                nuni = X[col].dropna().nunique()
                if nuni == 2:
                    params = {
                        'objective': 'binary'
                    }
                elif nuni > 2:
                    params = {
                        'objective': 'multiclass',
                        'num_class': nuni + 1
                    }
            else:  # automatic analyze column
                if analyze_column(X[col]) == 'numeric':
                    params = {
                        'objective': 'regression'
                    }
                else:
                    nuni = X[col].dropna().nunique()
                    if nuni == 2:
                        params = {
                            'objective': 'binary'
                        }
                    elif nuni > 2:
                        params = {
                            'objective': 'multiclass',
                            'num_class': nuni + 1
                        }
                    else:
                        print(f'column {col} has only one unique value.')
                        continue

            params['verbosity'] = -1
            null_idx = X[col].isnull()
            x_train = X.loc[~null_idx].drop(col, axis=1)
            x_test = X.loc[null_idx].drop(col, axis=1)
            y_offset = X[col].min()
            y_train = X.loc[~null_idx, col].astype(int) - y_offset
            dtrain = lgb.Dataset(
                data=x_train,
                label=y_train
            )

            early_stopping_rounds = 50
            model = lgb.train(
                params, dtrain, valid_sets=[dtrain],
                num_boost_round=self.n_iter,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=0,
            )

            y_test = model.predict(x_test)
            if params['objective'] == 'multiclass':
                y_test = np.argmax(y_test, axis=1).astype(float)
            elif params['objective'] == 'binary':
                y_test = (y_test > 0.5).astype(float)
            y_test += y_offset
            output_X.loc[null_idx, col] = y_test
            if params['objective'] in ['multiclass', 'binary']:
                output_X[col] = output_X[col].astype(int)
            self.imputers[col] = model
            self.offsets[col] = y_offset
            self.objectives[col] = params['objective']
            if self.verbose:
                print(f'{col}:\t{self.objectives[col]}...iter{model.best_iteration}/{self.n_iter}')

        return output_X

if run_imputer:

    features = pd.read_csv("data/features.csv")
    power = pd.read_csv("data/power.csv")


    all = pd.merge(features, power, on="Timestamp", how="left")
    all["is_train"] = all["Power(kW)"].notnull()
    all = all.drop(index=60686)
    all.replace(to_replace=99999, value=np.nan, inplace=True)
    all.loc[all["Torque"] == 0, "Torque"] = np.nan

    corr = all.drop("is_train", axis=1).corr()
    corr_unstack = corr.unstack().abs().reset_index()
    corr_unstack.columns = ["feat_1", "feat_2", "corr"]
    corr_unstack = corr_unstack[corr_unstack["feat_1"] != corr_unstack["feat_2"]]
    corr_unstack.sort_values('corr', ascending=False, inplace=True)
    most_correlated_features = corr_unstack["feat_1"].to_numpy().tolist()
    most_correlated_features = list(dict.fromkeys(most_correlated_features))

    imputer = LGBMImputer(verbose=True, feature_list=most_correlated_features)
    all.loc[:, all.columns[1:-1]] = imputer.fit_transform(all[all.columns[1:-1]])
    all.to_csv('all_lgbm_imputed.csv', index=False)

else:
    all = pd.read_csv('all_lgbm_imputed.csv')

sample_submission = pd.read_csv("data/sample_submission.csv")
original_features = all.columns.tolist()
power_offset = all["Power(kW)"].min()
power_max = all["Power(kW)"].max()
all["Power(kW)"] = all["Power(kW)"] - power_offset
all['power_log'] = np.log1p(all['Power(kW)'])

all = downcast_df_int_columns(all)
all = downcast_df_float_columns(all)

scope_cut = all[all['Scope CH 4'] < -20]['Power(kW)'].max()
gearbox_t1_cut = all[all['Gearbox_T1_High_Speed_Shaft_Temperature'] < 10]['Power(kW)'].max()
gearbox_t3_cut = all[all['Gearbox_T3_High_Speed_Shaft_Temperature'] < 10]['Power(kW)'].max()
gearbox_t1_int_cut = all[all['Gearbox_T1_Intermediate_Speed_Shaft_Temperature'] < 10]['Power(kW)'].max()
gearbox_t3_int_cut = all[all['Gearbox_T3_Intermediate_Speed_Shaft_Temperature'] < 10]['Power(kW)'].max()
gearbox_oil_2_cut = all[all['Gearbox_Oil-2_Temperature'] < 0]['Power(kW)'].max()
gearbox_oil_1_cut = all[all['Gearbox_Oil-1_Temperature'] < 0]['Power(kW)'].max()
temp_bearing_a_cut = all[all['Temperature Bearing_A'] < 0]['Power(kW)'].max()
temp_bearing_1_cut = all[all['Temperature Shaft Bearing-1'] < 0]['Power(kW)'].max()
temp_bearing_2_cut = all[all['Temperature Shaft Bearing-2'] < 0]['Power(kW)'].max()
temp_nacelle_cut = all[all['Temperature_Nacelle'] < 0]['Power(kW)'].max()
torque_cut = all[all['Torque'] < -250]['Power(kW)'].max()
temp_gearbox_dist_cut = all[all['Gearbox_Distributor_Temperature'] < 0]['Power(kW)'].max()
mom_fil_cut = all[all['Moment D Filtered'] < -1000]['Power(kW)'].max()
mom_dir_cut = all[all['Moment D Direction'] < -1000]['Power(kW)'].max()
temp_box3_cut = all[all['Temperature Axis Box-3'] < -10]['Power(kW)'].max()
temp_box2_cut = all[all['Temperature Axis Box-2'] < -10]['Power(kW)'].max()
temp_box1_cut = all[all['Temperature Axis Box-1'] < -10]['Power(kW)'].max()
temp_bbox3_cut = all[all['Temperature Battery Box-3'] < -15]['Power(kW)'].max()
temp_bbox2_cut = all[all['Temperature Battery Box-2'] < -15]['Power(kW)'].max()
temp_bbox1_cut = all[all['Temperature Battery Box-1'] < -15]['Power(kW)'].max()
hydra_cut = all[all['Hydraulic Prepressure'] < 50]['Power(kW)'].max()
rotot_cut = all[all['Angle Rotor Position'] >= 400]['Power(kW)'].max()
temp_tbase_cut = all[all['Temperature Tower Base'] > 42]['Power(kW)'].max()
ta_normal_cut = all[all['Tower Accelaration Normal Raw'] < -500]['Power(kW)'].max()
ta_lateral_cut = all[all['Tower Accelaration Lateral Raw'] > 500]['Power(kW)'].max()
epl_cut = all[all['External Power Limit'] < 3000]['Power(kW)'].max()
emp_amb_cut = all[all['Temperature Ambient'] < -10]['Power(kW)'].max()
wind1g_cut = all[all['Wind Deviation 1 seconds'] > 60]['Power(kW)'].max()
wind1l_cut = all[all['Wind Deviation 1 seconds'] < -100]['Power(kW)'].max()
wind10g_cut = all[all['Wind Deviation 10 seconds'] > 60]['Power(kW)'].max()
wind10l_cut = all[all['Wind Deviation 10 seconds'] < -100]['Power(kW)'].max()
proxy_135_cut = all[all['Proxy Sensor_Degree-135'] <= 5.5]['Power(kW)'].max()
proxy_225_cut = all[all['Proxy Sensor_Degree-225'] <= 5.5]['Power(kW)'].max()
blade1_cut = all[all['Blade-1 Actual Value_Angle-A'] < -5]['Power(kW)'].max()
blade2_cut = all[all['Blade-2 Actual Value_Angle-A'] < -5]['Power(kW)'].max()
blade3_cut = all[all['Blade-3 Actual Value_Angle-A'] < -5]['Power(kW)'].max()



# print(all.shape[0])
# all = all.loc[(all['Gearbox_T1_High_Speed_Shaft_Temperature'] >= 10) | (all.is_train == 0) | (all['Gearbox_T1_High_Speed_Shaft_Temperature'].isna())]
# all = all.loc[(all['Gearbox_T3_High_Speed_Shaft_Temperature'] >= 10) | (all.is_train == 0)| (all['Gearbox_T3_High_Speed_Shaft_Temperature'].isna())]
# all = all.loc[(all['Gearbox_T1_Intermediate_Speed_Shaft_Temperature'] >= 10) | (all.is_train == 0)| (all['Gearbox_T1_Intermediate_Speed_Shaft_Temperature'].isna())]
# all = all.loc[(all['Gearbox_T3_Intermediate_Speed_Shaft_Temperature'] >= 10) | (all.is_train == 0)| (all['Gearbox_T3_Intermediate_Speed_Shaft_Temperature'].isna())]
# all = all.loc[(all['Gearbox_Oil-2_Temperature'] >= 0) | (all.is_train == 0)| (all['Gearbox_Oil-2_Temperature'].isna())]
# all = all.loc[(all['Gearbox_Oil-1_Temperature'] >= 0) | (all.is_train == 0)| (all['Gearbox_Oil-1_Temperature'].isna())]
# all = all.loc[(all['Gearbox_Oil_Temperature'] >= 0) | (all.is_train == 0)| (all['Gearbox_Oil_Temperature'].isna())]
# all = all.loc[(all['Temperature Bearing_A'] >= 0) | (all.is_train == 0)| (all['Temperature Bearing_A'].isna())]
# all = all.loc[(all['Temperature Shaft Bearing-1'] >= 0) | (all.is_train == 0)| (all['Temperature Shaft Bearing-1'].isna())]
# all = all.loc[(all['Temperature Shaft Bearing-2'] >= 0) | (all.is_train == 0)| (all['Temperature Shaft Bearing-2'].isna())]
# all = all.loc[(all['Temperature_Nacelle'] >= 0) | (all.is_train == 0) | (all['Temperature_Nacelle'].isna())]
# all = all.loc[(all['Torque'] >= -250) | (all.is_train == 0)| (all['Torque'].isna())]
# all = all.loc[(all['Gearbox_Distributor_Temperature'] >= 0) | (all.is_train == 0)| (all['Gearbox_Distributor_Temperature'].isna())]
all = all.loc[(all['Moment D Filtered'] >= -1000) | (all.is_train == 0)| (all['Moment D Filtered'].isna())]
all = all.loc[(all['Moment D Direction'] >= -1000) | (all.is_train == 0)| (all['Moment D Direction'].isna())]
all = all.loc[(all['Operating State'] >= 10) | (all.is_train == 0)]
# all = all.loc[(all['Voltage A-N'] >= 370) | (all.is_train == 0)]
# all = all.loc[(all['Voltage C-N'] >= 370) | (all.is_train == 0)]
# all = all.loc[(all['Voltage B-N'] >= 370) | (all.is_train == 0)]
# all = all.loc[(all['Temperature Axis Box-3'] >= -10) | (all.is_train == 0) | (all['Temperature Axis Box-3'].isna())]
# all = all.loc[(all['Temperature Axis Box-2'] >= -10) | (all.is_train == 0) | (all['Temperature Axis Box-2'].isna())]
# all = all.loc[(all['Temperature Axis Box-1'] >= -10) | (all.is_train == 0) | (all['Temperature Axis Box-1'].isna())]
all = all.loc[(all['Temperature Battery Box-3'] >= -15) | (all.is_train == 0) | (all['Temperature Battery Box-3'].isna())]
all = all.loc[(all['Temperature Battery Box-2'] >= -15) | (all.is_train == 0) | (all['Temperature Battery Box-2'].isna())]
all = all.loc[(all['Temperature Battery Box-1'] >= -15) | (all.is_train == 0) | (all['Temperature Battery Box-1'].isna())]
# all = all.loc[(all['Hydraulic Prepressure'] >= 50) | (all.is_train == 0) | (all['Hydraulic Prepressure'].isna())]
# all = all.loc[(all['Angle Rotor Position'] < 400) | (all.is_train == 0) | (all['Angle Rotor Position'].isna())]
# all = all.loc[(all['Temperature Tower Base'] < 42) | (all.is_train == 0) | (all['Temperature Tower Base'].isna())]
# all = all.loc[(all['Tower Accelaration Normal Raw'] > -500) | (all.is_train == 0) | (all['Tower Accelaration Normal Raw'].isna())]
all = all.loc[(all['Tower Accelaration Lateral Raw'] < 500) | (all.is_train == 0) | (all['Tower Accelaration Lateral Raw'].isna())]
all = all.loc[(all['Tower Accelaration Normal Raw'] > - 2000) | (all.is_train == 0) | (all['Tower Accelaration Lateral Raw'].isna())]
all = all.loc[(all['External Power Limit'] > 3000) | (all.is_train == 0) | (all['External Power Limit'].isna())]
all = all.loc[(all['Temperature Ambient'] > -10) | (all.is_train == 0) | (all['Temperature Ambient'].isna())]
# all = all.loc[(all['Wind Deviation 1 seconds'] < 60) | (all.is_train == 0) | (all['Wind Deviation 1 seconds'].isna())]
# all = all.loc[(all['Wind Deviation 10 seconds'] < 60) | (all.is_train == 0) | (all['Wind Deviation 10 seconds'].isna())]
# all = all.loc[(all['Wind Deviation 1 seconds'] > -100) | (all.is_train == 0) | (all['Wind Deviation 1 seconds'].isna())]
# all = all.loc[(all['Wind Deviation 10 seconds'] > -100) | (all.is_train == 0) | (all['Wind Deviation 10 seconds'].isna())]
# all = all.loc[(all['Proxy Sensor_Degree-135'] > 5.5) | (all.is_train == 0) | (all['Proxy Sensor_Degree-135'].isna())]
# all = all.loc[(all['Proxy Sensor_Degree-225'] > 5.5) | (all.is_train == 0) | (all['Proxy Sensor_Degree-225'].isna())]
# all = all.loc[(all['Blade-3 Actual Value_Angle-A'] > -5) | (all.is_train == 0) | (all['Blade-3 Actual Value_Angle-A'].isna())]
# all = all.loc[(all['Blade-2 Actual Value_Angle-A'] > -5) | (all.is_train == 0) | (all['Blade-2 Actual Value_Angle-A'].isna())]
# all = all.loc[(all['Blade-1 Actual Value_Angle-A'] > -5) | (all.is_train == 0) | (all['Blade-1 Actual Value_Angle-A'].isna())]
# all = all.loc[(all['Scope CH 4'] > -20) | (all.is_train == 0) | (all['Scope CH 4'].isna())]
# all = all.loc[(all['Pitch Offset-1 Asymmetric Load Controller'] > -0.075) | (all.is_train == 0) | (all['Temperature Ambient'].isna())]
# all = all.loc[(all['Pitch Offset-2 Asymmetric Load Controller'] < -0.05) | (all.is_train == 0) | (all['Temperature Ambient'].isna())]
# all = all.loc[(all['Pitch Offset-3 Asymmetric Load Controller'] > -0.05) | (all.is_train == 0) | (all['Temperature Ambient'].isna())]
print(all.shape[0])


def print_mean(mean):
    print(f"Mean is {mean} now")

def apply_limiters(df, is_validation=False):
    limited_feature = 'Power(kW)'
    if is_validation:
        limited_feature = "pred"
    print_mean(df[limited_feature].mean())
    df[limited_feature] = np.clip(df[limited_feature], 0, df["n-set_1_limit"])
    print_mean(df[limited_feature].mean())
    df[limited_feature] = np.clip(df[limited_feature], 0, df["pf_limit"])
    print_mean(df[limited_feature].mean())
    df[limited_feature] = np.clip(df[limited_feature], 0, df["sf_limit"])
    print_mean(df[limited_feature].mean())
    df[limited_feature] = np.clip(df[limited_feature], 0, df["os_limit"])
    print_mean(df[limited_feature].mean())
    df[limited_feature] = np.clip(df[limited_feature], 0, df["turbine_limit"])
    print_mean(df[limited_feature].mean())
    df[limited_feature] = np.clip(df[limited_feature], 0, df["Internal Power Limit"] - power_offset - 0.57666015625)
    print_mean(df[limited_feature].mean())
    df.loc[df['Scope CH 4'] < -20, limited_feature] = scope_cut
    print_mean(df[limited_feature].mean())
    df.loc[df['Gearbox_T1_High_Speed_Shaft_Temperature'] < 10, limited_feature] = np.clip(df[limited_feature], 0, gearbox_t1_cut)
    df.loc[df['Gearbox_T3_High_Speed_Shaft_Temperature'] < 10, limited_feature] = np.clip(df[limited_feature], 0, gearbox_t3_cut)
    df.loc[df['Gearbox_T1_Intermediate_Speed_Shaft_Temperature'] < 10, limited_feature] = np.clip(df[limited_feature], 0, gearbox_t1_int_cut)
    df.loc[df['Gearbox_T3_Intermediate_Speed_Shaft_Temperature'] < 10, limited_feature] = np.clip(df[limited_feature], 0, gearbox_t3_int_cut)
    df.loc[df['Gearbox_Oil-2_Temperature'] < 0, limited_feature] = np.clip(df[limited_feature], 0, gearbox_oil_2_cut)
    df.loc[df['Gearbox_Oil-1_Temperature'] < 0, limited_feature] = np.clip(df[limited_feature], 0, gearbox_oil_1_cut)
    df.loc[df['Temperature Bearing_A'] < 0, limited_feature] = np.clip(df[limited_feature], 0, temp_bearing_a_cut)
    df.loc[df['Temperature Shaft Bearing-1'] < 0, limited_feature] = np.clip(df[limited_feature], 0, temp_bearing_1_cut)
    df.loc[df['Temperature Shaft Bearing-2'] < 0, limited_feature] = np.clip(df[limited_feature], 0, temp_bearing_2_cut)
    df.loc[df['Temperature_Nacelle'] < 0, limited_feature] = np.clip(df[limited_feature], 0, temp_nacelle_cut)
    df.loc[df['Torque'] < -250, limited_feature] = np.clip(df[limited_feature], 0, torque_cut)
    df.loc[df['Gearbox_Distributor_Temperature'] < 0, limited_feature] = np.clip(df[limited_feature], 0, temp_gearbox_dist_cut)
    df.loc[df['Moment D Filtered'] < -1000, limited_feature] = np.clip(df[limited_feature], 0, mom_fil_cut)
    df.loc[df['Moment D Direction'] < -1000, limited_feature] = np.clip(df[limited_feature], 0, mom_dir_cut)
    df.loc[df['Temperature Axis Box-3'] < -10, limited_feature] = np.clip(df[limited_feature], 0, temp_box3_cut)
    df.loc[df['Temperature Axis Box-2'] < -10, limited_feature] = np.clip(df[limited_feature], 0, temp_box2_cut)
    df.loc[df['Temperature Axis Box-1'] < -10, limited_feature] = np.clip(df[limited_feature], 0, temp_box1_cut)
    df.loc[df['Temperature Battery Box-3'] < -15, limited_feature] = np.clip(df[limited_feature], 0, temp_bbox3_cut)
    df.loc[df['Temperature Battery Box-2'] < -15, limited_feature] = np.clip(df[limited_feature], 0, temp_bbox2_cut)
    df.loc[df['Temperature Battery Box-1'] < -15, limited_feature] = np.clip(df[limited_feature], 0, temp_bbox1_cut)
    df.loc[df['Hydraulic Prepressure'] < 50, limited_feature] = np.clip(df[limited_feature], 0, hydra_cut)
    df.loc[df['Angle Rotor Position'] >=400 , limited_feature] = np.clip(df[limited_feature], 0, rotot_cut)
    df.loc[df['Temperature Tower Base'] > 42, limited_feature] = np.clip(df[limited_feature], 0, temp_tbase_cut)
    df.loc[df['Tower Accelaration Normal Raw'] < -500, limited_feature] = np.clip(df[limited_feature], 0, ta_normal_cut)
    df.loc[df['Tower Accelaration Lateral Raw'] > 500, limited_feature] = np.clip(df[limited_feature], 0, ta_lateral_cut)
    df.loc[df['External Power Limit'] < 3000, limited_feature] = np.clip(df[limited_feature], 0, epl_cut)
    df.loc[df['Temperature Ambient'] < -10, limited_feature] = np.clip(df[limited_feature], 0, emp_amb_cut)
    df.loc[df['Wind Deviation 1 seconds'] > 60, limited_feature] = np.clip(df[limited_feature], 0, wind1g_cut)
    df.loc[df['Wind Deviation 1 seconds'] < -100, limited_feature] = np.clip(df[limited_feature], 0, wind1l_cut)
    df.loc[df['Wind Deviation 10 seconds'] > 60, limited_feature] = np.clip(df[limited_feature], 0, wind10g_cut)
    df.loc[df['Wind Deviation 10 seconds'] < -100, limited_feature] = np.clip(df[limited_feature], 0, wind10l_cut)
    df.loc[df['Proxy Sensor_Degree-135'] <= 5.5, limited_feature] = np.clip(df[limited_feature], 0, proxy_135_cut)
    df.loc[df['Proxy Sensor_Degree-225'] <= 5.5, limited_feature] = np.clip(df[limited_feature], 0, proxy_225_cut)
    df.loc[df['Blade-1 Actual Value_Angle-A'] < -5, limited_feature] = np.clip(df[limited_feature], 0, blade1_cut)
    df.loc[df['Blade-2 Actual Value_Angle-A'] < -5, limited_feature] = np.clip(df[limited_feature], 0, blade2_cut)
    df.loc[df['Blade-3 Actual Value_Angle-A'] < -5, limited_feature] = np.clip(df[limited_feature], 0, blade3_cut)
    print_mean(df[limited_feature].mean())
    df[limited_feature] = np.clip(df[limited_feature], 0, df["scope_limit"])
    return df

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

all.loc[all["Operating State"].isna() & (all["Turbine State"] == 1.0), "Operating State"] = 16.0
all.loc[all["Operating State"].isna() & (all["Turbine State"] == 5.0), "Operating State"] = 11.0
all.loc[all["Operating State"].isna() & (all["Turbine State"] == 2.0), "Operating State"] = 16.0
all.loc[all["Turbine State"].isna() & (all["Operating State"] == 12), "Turbine State"] = 3.0
all.loc[all["Turbine State"].isna() & (all["Operating State"] == 15), "Turbine State"] = 3.0



all["pf_abs"] = all["Power Factor"].abs()
all["pf_sign"] = (all["Power Factor"] > 0).astype(int)

all["n-set_1_0"] = (all["N-set 1"] == 0.0).astype("int")
all["n-set_1_1735"] = (all["N-set 1"] == 1735.0).astype("int")

all.loc[all["N-set 1"] > 0.0, "n-set_1_limit"] = all[all["N-set 1"] > 0.0]["N-set 1"] * 2780 / 1735.0 - power_offset - 0.57666
all.loc[all["N-set 1"] > 0.0, "n-set-diff"] = all[all["N-set 1"] > 0.0]["n-set_1_limit"] - all[all["N-set 1"] > 0.0]["Power(kW)"]



all["sf_bin"] = all["State and Fault"].round(0)
all.loc[all["State and Fault"] < 1100, "sf_limit"] = (1098 - all[all["State and Fault"] < 1100]["State and Fault"]) * (2828.02 / 1098) - power_offset + 98.882
all["sf-limit-diff"] = all["sf_limit"] - all["Power(kW)"]
all["sf_2"] = (all["State and Fault"] == 2.0).astype("int")
all["sf_l2"] = (all["State and Fault"] < 2.0).astype("int")


all.loc[all['Scope CH 4'] > 25, "scope_limit"] = (2828.02 - (all[all['Scope CH 4'] > 25]['Scope CH 4'] - 25) * (2828.02 / 75)) - power_offset - 353.28
all["scope-limit-diff"] = all["scope_limit"] - all["Power(kW)"]

all["pf_bin"] = all["Power Factor"].round(1)
pf_abs_bin_maxes = all.groupby('pf_bin')['Power(kW)'].max().reset_index()
pf_abs_bin_maxes = pf_abs_bin_maxes.rename(columns={'Power(kW)': 'pf_limit'})
all = all.merge(pf_abs_bin_maxes, on='pf_bin', how='left')

all["limit_difference"] = all["Internal Power Limit"] - all["Power(kW)"] - power_offset
all["limit_capped"] = (all["limit_difference"] < lcap_threshold).astype("int")
# all["torque_in_range"] = (all["Torque"].abs() < 100).astype(int)
all["Angle Rotor Position_2"] = (all["Angle Rotor Position"] - 180).abs()
all.loc[all["Torque"] == 0, "Torque"] = np.nan
all["Reactive Power"] = all["Reactive Power"].abs()
all["pf*reactive"] = all["Power Factor"] * all["Reactive Power"]

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
    #"operation_12",
    # "operation_11",
    # "operation_15",
    # "operation_19",
    # "operation_16",
    "n-set_1_0",
    "pf*reactive",
    "pf_abs",
    "pf_sign",
    #"torque_in_range",
    "Angle Rotor Position_2",
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

def prep(df, features):
    new_features = features.copy()
    for feature in features:
        df[feature + "_lag1"] = df[feature].shift(1).fillna(method='bfill')
        df[feature + "_lag2"] = df[feature].shift(2).fillna(method='bfill')
        df[feature + "_lead1"] = df[feature].shift(-1).fillna(method='ffill')
        df[feature + "_diff1"] = df[feature + "_lead1"] - df[feature]
        df[feature + "_diff-1"] = df[feature] - df[feature + "_lag1"]
        df[feature + "_avg2"] = (df[feature + "_lead1"] + df[feature]) / 2
        df[feature + "_avg3"] = (df[feature + "_lead1"] + df[feature] + df[feature + "_lag1"]) / 32
        df[feature + "_rolling2"] = df[feature].rolling(2).mean().fillna(method='bfill')
        df[feature + "_rolling3"] = df[feature].rolling(3).mean().fillna(method='bfill')
        new_features.append(feature + "_lag1")
        new_features.append(feature + "_lag2")
        new_features.append(feature + "_lead1")
        new_features.append(feature + "_diff1")
        new_features.append(feature + "_rolling2")
        new_features.append(feature + "_rolling3")
        new_features.append(feature + "_avg2")
        new_features.append(feature + "_avg3")

    return df, new_features


all, new_features = prep(all, features)

for i in range(50):
    all.loc[(all['Operating State'].isna()) & (all['Turbine State_diff-1'] == 0) & (all['Operating State_lag1'].notna()), 'Operating State'] = all['Operating State_lag1']
    all.loc[(all['Operating State'].isna()) & (all['Turbine State_diff1'] == 0) & (all['Operating State_lead1'].notna()), 'Operating State'] = all['Operating State_lead1']
    all.loc[(all['Turbine State'].isna()) & (all['Operating State_diff-1'] == 0) & (all['Operating State_lag1'].notna()), 'Turbine State'] = all['Turbine State_lag1']
    all.loc[(all['Turbine State'].isna()) & (all['Operating State_diff1'] == 0) & (all['Operating State_lead1'].notna()), 'Turbine State'] = all['Turbine State_lead1']

all.loc[(all['Operating State'].isna()), ['Operating State', 'Turbine State']] = [16.0, 1.0]
all.loc[all['Turbine State'].isna() & (all['Operating State'] == 16.0), 'Turbine State'] = 1.0
all.loc[all['Turbine State'].isna() & (all['Operating State'] == 11.0), 'Turbine State'] = 4.0

all.loc[(all["Operating State"] >= 11) & (all["Operating State"] <= 16), "os_limit"] = (all[(all["Operating State"] >= 11) & (all["Operating State"] <= 16)]["Operating State"] - 11) * (2780 / 5) - power_offset + 0.07833
all.loc[(all["Operating State"] >= 11) & (all["Operating State"] <= 16), "os-limit-diff"] = all[(all["Operating State"] >= 11) & (all["Operating State"] <= 16)]["os_limit"] - all[(all["Operating State"] >= 11) & (all["Operating State"] <= 16)]["Power(kW)"]

all.loc[all["Turbine State"] > 2, "turbine_limit"] = (2097.69 - (all[all["Turbine State"] > 2]["Turbine State"] - 2) * (2097.69 / 3)) - power_offset + 56.97
all["turbine-limit-diff"] = all["turbine_limit"] - all["Power(kW)"]

all["operation_12"] = (all["Operating State"] == 12.0).astype("int")
all["turbine_1"] = (all["Turbine State"] == 1.0).astype("int")
all["turbine_2"] = (all["Turbine State"] == 2.0).astype("int")
all["turbine_3"] = (all["Turbine State"] == 3.0).astype("int")
all["turbine_4"] = (all["Turbine State"] == 4.0).astype("int")
all["turbine_5"] = (all["Turbine State"] == 5.0).astype("int")
all["turbine_12"] = (all["Turbine State"] <= 2.0).astype("int")
all["operation_11"] = (all["Operating State"] == 11.0).astype("int")
all["operation_15"] = (all["Operating State"] == 15.0).astype("int")
all["operation_19"] = (all["Operating State"] == 19.0).astype("int")
all["operation_16"] = (all["Operating State"] == 16.0).astype("int")

all["turbine_prev"] = np.nan
all["turbine_next"] = np.nan
all["operation_prev"] = np.nan
all["operation_next"] = np.nan

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

all["Torque_abs"] = all["Torque"].abs()
all["Torque_log"] = np.log1p(all["Torque_abs"])
all["Torque_sqrt"] = np.sqrt(all["Torque_abs"])
all["Torque_sign"] = np.sign(all["Torque"])
all["Torque_square"] = np.power(all["Torque"], 2)
all["Torque_lag1"] = all["Torque"].shift(1).fillna(method='bfill')
all["Torque_lead1"] = all["Torque"].shift(-1).fillna(method='ffill')
all["Torque_avg_2"] = (all["Torque_lead1"] + all["Torque"]) / 2
all["Torque_avg_3"] = (all["Torque_lead1"] + all["Torque"] + all["Torque_lag1"]) / 3
all["Torque_diff1"] = (all["Torque"] - all["Torque_lag1"])
all["Torque_diff-1"] = all["Torque"] - all["Torque_lead1"]
all["Torque_diff_total"] = all["Torque_diff1"].abs() + all["Torque_diff-1"].abs()
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
#all["Torque"] = all["Torque_abs"]

features_cat = [
    "State and Fault",
    "sf_2",
    "sf_l2",
    "turbine_stationary",
    "operation_stationary",
    "pf_abs",
    "Torque",
    "Torque_lag1",
    "Torque_lag2",
    "Torque_diff1",
    "Torque_diff-1",
    "Torque_lead1",
    "Torque_rolling2",
    "Torque_log",
    "Torque_sqrt",
    "Torque_avg2",
    "Torque_avg3",
    'Voltage A-N',
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
    'Voltage A-N_avg3',
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
    #"torque_in_range",
    "Temperature Bearing_A_lead1",
    "Blade-2 Set Value_Degree",
    "Blade-2 Set Value_Degree_rolling2",
    "Gearbox_T3_High_Speed_Shaft_Temperature",
    "Gearbox_Oil-1_Temperature",
    "Gearbox_Oil-1_Temperature_lead1",
    # #'Nacelle Revolution_cos',
]

features_xgb = [
            "State and Fault",
            "sf_2",
            "sf_l2",
            'Pitch Offset Tower Feedback_diff1',
            "Temperature Gearbox Bearing Hollow Shaft",
            'Gearbox_T1_High_Speed_Shaft_Temperature_avg3',
            'Voltage A-N_avg3',
            'Voltage A-N_rolling2',
            'Operating State_rolling3',
            'Voltage A-N',
            'pf_abs',
             'Torque',
            #'Torque_abs',
             #'Torque_int',
            'Torque_lag1',
            'Torque_lead1',
            'Torque_diff1',
            'Torque_diff-1',
             #'Torque_diff_total',
            'Torque_rolling2',
            'Torque_log',
            'Torque_sqrt',
            'Torque_avg_2',
            'Torque_avg_3',
             #'Torque_is_0',
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
            #"operation_15",
            "operation_16",
            #"operation_19",
             #"operation_state_transition",
             #"turbine_state_transition",
            "turbine_stationary",
            "operation_stationary",
            "turbine_1",
            "turbine_2",
            "turbine_12",
            'Operating State_rolling2',
            'Operating State_lead1',
            'N-set 1',
            'N-set 1_rolling2',
            # 'N-set 1_rolling3',
            # 'N-set 1_avg3',
            'Blade-1 Set Value_Degree',
            'n-set_1_0',
            'n-set_1_1735',
            'n-set_1_0_lag1',
            'n-set_1_0_rolling2',
            'pf*reactive',
            'Turbine State',
            'Turbine State_rolling2',
             #'hour',
             #'torque_in_range',
            'Temperature Bearing_A_lead1',
            'Blade-2 Set Value_Degree',
            'Gearbox_T3_High_Speed_Shaft_Temperature',
            'Gearbox_Oil-1_Temperature',
            # #'Nacelle Revolution_cos',
            ]

features_lgbm = [
    'Temperature Axis Box-3_diff1',
    'Tower Accelaration Normal Raw_lag2',
    'Converter Control Unit Reactive Power_diff1',
    'Gearbox_Distributor_Temperature_diff1',
    'Torque Offset Tower Feedback_lag2',
    'Line Frequency_avg2',
    'Tower Acceleration Lateral_lag2',
    'Moment Q Direction',
    'Nacelle Revolution_sin_diff1',
    'Voltage B-N_diff1',
    'Torque Offset Tower Feedback_lead1',
    'Angle Rotor Position',
    'Blade-2 Set Value_Degree',
    'Moment Q Direction_lag2',
    'Tower Acceleration Normal_lead1',
    'Tower Accelaration Lateral Raw_lead1',
    'Pitch Offset Tower Feedback_lag2',
    'Pitch Offset Tower Feedback_avg2',
    'Gearbox_T1_Intermediate_Speed_Shaft_Temperature_diff1',
    'Pitch Offset Tower Feedback_lag1',
    'Angle Rotor Position_2_rolling3',
    'Tower Acceleration Normal_lag2',
    'Nacelle Revolution_diff1',
    'Angle Rotor Position_diff1',
    'Tower Accelaration Lateral Raw_rolling3',
    'Temperature Ambient_diff1',
    'Temperature Shaft Bearing-1_diff1',
    'Tower Accelaration Normal Raw_diff1',
    'Tower Accelaration Lateral Raw_rolling2',
    'Tower Accelaration Normal Raw_avg3',
    'Voltage A-N',
    'Temperature Tower Base_diff1',
    'Tower Accelaration Normal Raw_avg2',
    'pf*reactive',
    'Nacelle Position_Degree_cos_diff1',
    'Temperature Trafo-2_lead1',
    'Temperature Battery Box-2_diff1',
    'Tower Accelaration Normal Raw_rolling2',
    'Tower Accelaration Lateral Raw_avg2',
    'Blade-1 Set Value_Degree',
    'Line Frequency_lag2',
    'Blade-3 Actual Value_Angle-A',
    'Pitch Offset Tower Feedback_lead1',
    'Voltage A-N_diff1',
    'Moment Q Direction_diff1',
    'Tower Accelaration Lateral Raw',
    'Blade-2 Actual Value_Angle-A',
    'State and Fault',
    'Torque Offset Tower Feedback_rolling3',
    'Tower Accelaration Lateral Raw_lag1',
    'Tower Acceleration Normal_avg2',
    'Blade-1 Actual Value_Angle-A',
    'Operating State',
    'Temperature Axis Box-1_diff1',
    'N-set 1_diff1',
    'Proxy Sensor_Degree-315_lead1',
    'Angle Rotor Position_2_lag2',
    'Temperature Battery Box-1_diff1',
    'Tower Acceleration Normal_diff1',
    'Angle Rotor Position_2_diff1',
    'Pitch Offset Tower Feedback_rolling2',
    'Gearbox_T1_High_Speed_Shaft_Temperature_diff1',
    'Pitch Offset Tower Feedback_avg3',
    'Pitch Offset Tower Feedback_rolling3',
    'Moment D Filtered_diff1',
    'Converter Control Unit Voltage_diff1',
    'Temperature Trafo-2_diff1',
    'Moment Q Filltered_diff1',
    'Temperature Axis Box-2_diff1',
    'Temperature Battery Box-3_diff1',
    'Hydraulic Prepressure_diff1',
    'Angle Rotor Position_2',
    'pf_abs_diff1',
    'Temperature Shaft Bearing-2_diff1',
    'Torque Offset Tower Feedback',
    'Internal Power Limit_diff1',
    'Tower Accelaration Normal Raw',
    'Proxy Sensor_Degree-45_diff1',
    'Gearbox_Oil-2_Temperature_diff1',
    'Torque_lag1',
    'Tower Acceleration Lateral',
    'Tower Acceleration Lateral_diff1',
    'Temperature Bearing_A_diff1',
    'Proxy Sensor_Degree-315_diff1',
    'Pitch Demand Baseline_Degree',
    'Proxy Sensor_Degree-135_diff1',
    'Torque_avg3',
    'Torque_lead1',
    'Proxy Sensor_Degree-315',
    'Proxy Sensor_Degree-225_diff1',
    'Torque_rolling2',
    'Temperature_Nacelle_diff1',
    'Tower Acceleration Normal',
    'Temperature Trafo-3_diff1',
    'Gearbox_T3_High_Speed_Shaft_Temperature_diff1',
    'Pitch Offset Tower Feedback',
    'N-set 1',
    'Torque_avg2',
    'Torque_diff1',
    'Torque'
]


if model_type == "lgbm":
    new_features = features
elif model_type == "xgb":
    new_features = features_xgb
elif model_type == "cat":
    new_features = features
elif model_type == "et":
    new_features = features

print(f"Number of features: {len(new_features)}")

object_cols = [
    "sf_2",
    "sf_l2",
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
    "turbine_1",
    "turbine_2",
    "turbine_3",
    "turbine_4",
    "turbine_5",
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

plot_features = new_features.copy()
for f in original_features:
    if f not in plot_features:
        plot_features.append(f)
plot_features.append('os_limit')
plot_features.append('pf_limit')
plot_features.append('sf_limit')
plot_features.append('turbine_limit')
plot_features.append('n-set_1_limit')
plot_features.append('scope_limit')


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
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.8, 0.9, 1.0]),
        "subsample": trial.suggest_categorical("subsample", [0.6, 0.7, 0.8, 1.0]),
        "learning_rate": trial.suggest_float("learning_rate", 0.002, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 2000, 6000, 200),
        "max_depth": trial.suggest_int("max_depth", 8, 15),
        "random_state": 555,
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 0.1, 10),
        "objective": "reg:squarederror",
        "enable_categorical": True,
    }
    model = XGBRegressor(**param)
    y_oof = np.zeros(train_x.shape[0])

    ix = 0
    for train_ind, val_ind in kf.split(train_x, train_y):
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


    plot_df = all[all.is_train == 1][plot_features]
    if use_scaler:
        plot_df[numeric_cols] = sc.inverse_transform(plot_df[numeric_cols])
    plot_df["pred"] = y_oof
    plot_df["power"] = train_y

    plot_df = apply_limiters(plot_df, is_validation=True)
    plot_df["pred_diff"] = (plot_df["pred"] - plot_df["power"]).abs()
    plot_df.sort_values("pred_diff", ascending=False)

    sns.scatterplot(x=plot_df["pred_diff"], y=plot_df["power"], hue=plot_df['Torque'].isna())
    plt.show()

    rmse = mse(plot_df["power"], plot_df["pred"], squared=False)

    return rmse


def objective_lgbm(trial):

    # To select which parameters to optimize, please look at the XGBoost documentation:
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, metric="rmse")
    param = {
        "metric": "rmse",
        "random_state": 555,
        "n_estimators": 3000,
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.6, 0.7, 0.8, 0.9, 1.0]),
        "subsample": trial.suggest_categorical("subsample", [0.6, 0.7, 0.8, 1.0]),
        "learning_rate": trial.suggest_float("learning_rate", 0.002, 0.1),
        "max_depth": trial.suggest_int("max_depth", 8, 15),
        "num_leaves": trial.suggest_int("num_leaves", 1, 1000),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 300),
    }
    model = LGBMRegressor(**param)

    model.fit(
        train_optuna_x,
        train_optuna_y,
        eval_set=[(val_optuna_x, val_optuna_y)],
        eval_metric="rmse",
        early_stopping_rounds=100,
        callbacks=[pruning_callback],
        verbose=100,
    )

    preds = model.predict(val_optuna_x)

    rmse = mse(val_optuna_y, preds, squared=False)

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

    kf = KFold(N_FOLDS, shuffle=True, random_state=555)
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
    model = ExtraTreeRegressor(**param, random_state=555)

    N_FOLDS = 5

    kf = KFold(N_FOLDS, shuffle=True, random_state=555)
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


def objective_hgbt(trial):

    train_x = all[all.is_train == 1][new_features]
    train_y = all[all.is_train == 1][target_feature]

    N_FOLDS = 5

    kf = KFold(N_FOLDS, shuffle=True, random_state=555)

    # To select which parameters to optimize, please look at the XGBoost documentation:
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    param = {
        "learning_rate": trial.suggest_float("learning_rate", 0.002, 0.05),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 50, 250, 5),
        "max_depth": trial.suggest_int("max_depth", 10, 20, 1),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100, 1),
        "loss": 'squared_error'
    }
    model = HistGradientBoostingRegressor(**param)

    y_oof = np.zeros(train_x.shape[0])

    ix = 0
    for train_ind, val_ind in kf.split(train_x, train_y):
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

    plot_df = all[all.is_train == 1][plot_features]
    if use_scaler:
        plot_df[numeric_cols] = sc.inverse_transform(plot_df[numeric_cols])
    plot_df["pred"] = y_oof
    plot_df["power"] = train_y

    plot_df = apply_limiters(plot_df, is_validation=True)
    plot_df["pred_diff"] = (plot_df["pred"] - plot_df["power"]).abs()
    plot_df.sort_values("pred_diff", ascending=False)

    sns.scatterplot(x=plot_df["pred_diff"], y=plot_df["power"], hue=plot_df['Torque'].isna())
    plt.show()

    rmse = mse(plot_df["power"], plot_df["pred"], squared=False)

    return rmse


if run_optuna:
    N_FOLDS = 5

    kf = StratifiedKFoldReg(N_FOLDS, shuffle=True, random_state=555)
    optuna_index, optuna_val_index = next(kf.split(train_x, train_y))
    train_optuna_x = train_x.loc[optuna_index]
    train_optuna_y = train_y.loc[optuna_index]
    val_optuna_x = train_x.loc[optuna_val_index]
    val_optuna_y = train_y.loc[optuna_val_index]

    study = optuna.create_study(direction="minimize")
    if model_type == "xgb":
        study.optimize(objective_xgb, n_trials=optuna_trials)
    elif model_type == "et":
        study.optimize(objective_et, n_trials=optuna_trials)
    elif model_type == "lgbm":
        study.optimize(objective_lgbm, n_trials=optuna_trials)
    elif model_type == "cat":
        study.optimize(objective_cat, n_trials=optuna_trials)
    elif model_type == "hgbt":
        study.optimize(objective_hgbt, n_trials=optuna_trials)
    else:
        print(f"Not a valid model type: {model_type}")
        exit(1)
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
        best_params["random_state"] = 555
        best_params["enable_categorical"] = True
    elif model_type == "cat":
        best_params = {'colsample_bylevel': 0.09972653071196472, 'depth': 10, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}
        best_params["iterations"] = 8000
        best_params["objective"] = "RMSE"
    elif model_type == "lgbm":
        best_params = {'reg_alpha': 0.05743055364152163, 'reg_lambda': 7.400420760838195, 'colsample_bytree': 0.9, 'subsample': 0.7, 'learning_rate': 0.03581775161211641, 'max_depth': 11, 'num_leaves': 550, 'min_child_samples': 140}
        best_params["metric"] = "rmse"
        best_params["random_state"] = 555
        best_params["n_estimators"] = 7000
    elif model_type == "et":
        best_params = {'ccp_alpha': 0.08500021149054458, 'max_depth': 58, 'min_samples_split': 7, 'max_leaf_nodes': 984, 'max_features': 0.9615571697348897, 'min_samples_leaf': 6}
        best_params["n_estimators"] = 2000
    elif model_type == "hgbt":
        best_params = {'learning_rate': 0.04945816004518069, 'max_leaf_nodes': 140, 'max_depth': 20, 'min_samples_leaf': 43}

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

kf = StratifiedKFoldReg(N_FOLDS, shuffle=True, random_state=555)
y_oof = np.zeros(train_x.shape[0])
y_test = np.zeros(test_x.shape[0])

ix = 0

if model_type == "xgb":
    for train_ind, val_ind in kf.split(train_x, train_y):
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


    for f in original_features:
        if f not in new_features:
            new_features.append(f)
    new_features.append('os_limit')
    new_features.append('pf_limit')
    new_features.append('sf_limit')
    new_features.append('turbine_limit')
    new_features.append('n-set_1_limit')
    new_features.append('scope_limit')
    plot_df = all[all.is_train == 1][new_features]
    if use_scaler:
        plot_df[numeric_cols] = sc.inverse_transform(plot_df[numeric_cols])
    plot_df["pred"] = y_oof
    plot_df["power"] = train_y

    plot_df = apply_limiters(plot_df, is_validation=True)
    plot_df["pred_diff"] = (plot_df["pred"] - plot_df["power"]).abs()
    plot_df.to_csv(f"fuck_up_{trial}.csv", sep=",", index=False)
    plot_df.sort_values("pred_diff", ascending=False)

    plt.figure(figsize=(32, 12))
    sns.distplot(
        plot_df[:100]["power"],
        hist=True,
        color="blue",
        kde=True,
        bins=30,
        label="Power(kW)",
    )
    sns.distplot(
        plot_df[:100]["pred"],
        hist=True,
        color="red",
        kde=True,
        bins=30,
        label="predictions",
    )
    plt.legend()
    plt.show()

    sns.scatterplot(x=plot_df["pred"], y=plot_df["power"], hue=plot_df['Torque'].isna())
    plt.show()

    sns.scatterplot(x=plot_df["pred_diff"], y=plot_df["power"], hue=plot_df['Torque'].isna())
    plt.show()
    rmse = mse(plot_df["power"], plot_df["pred"], squared=False)
    print(f"Val Score: {rmse}")

elif model_type == "lgbm":
    for train_ind, val_ind in kf.split(train_x, train_y):
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
            early_stopping_rounds=50,
            verbose=100,
        )

        if plot_importance:
            importances = model.feature_importances_
            indices = np.argsort(importances)
            importances_sorted = np.sort(importances)
            # for indice in indices[-100:]:
            #     print(f"'{new_features[indice]}'")
            # print(f"Lowest index score = {importances_sorted[-100]}")
            indices = indices[-60:]
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

    for f in original_features:
        if f not in new_features:
            new_features.append(f)
    new_features.append('os_limit')
    new_features.append('pf_limit')
    new_features.append('sf_limit')
    new_features.append('turbine_limit')
    new_features.append('n-set_1_limit')
    new_features.append('scope_limit')
    plot_df = all[all.is_train == 1][new_features]
    if use_scaler:
        plot_df[numeric_cols] = sc.inverse_transform(plot_df[numeric_cols])
    plot_df["pred"] = y_oof
    plot_df["power"] = train_y

    plot_df = apply_limiters(plot_df, is_validation=True)
    plot_df["pred_diff"] = (plot_df["pred"] - plot_df["power"]).abs()
    plot_df.to_csv(f"fuck_up_{trial}.csv", sep=",", index=False)
    plot_df.sort_values("pred_diff", ascending=False)

    plt.figure(figsize=(32, 12))
    sns.distplot(
        plot_df[:100]["power"],
        hist=True,
        color="blue",
        kde=True,
        bins=30,
        label="Power(kW)",
    )
    sns.distplot(
        plot_df[:100]["pred"],
        hist=True,
        color="red",
        kde=True,
        bins=30,
        label="predictions",
    )
    plt.legend()
    plt.show()

    sns.scatterplot(x=plot_df["pred"], y=plot_df["power"], hue=plot_df['Torque'].isna())
    plt.show()

    sns.scatterplot(x=plot_df["pred_diff"], y=plot_df["power"], hue=plot_df['Torque'].isna())
    plt.show()
    rmse = mse(plot_df["power"], plot_df["pred"], squared=False)
    print(f"Val Score: {rmse}")

elif model_type == "cat":
    for train_ind, val_ind in kf.split(train_x, train_y):
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

    for f in original_features:
        if f not in new_features:
            new_features.append(f)
    new_features.append('os_limit')
    new_features.append('pf_limit')
    new_features.append('sf_limit')
    new_features.append('turbine_limit')
    new_features.append('n-set_1_limit')
    new_features.append('scope_limit')
    plot_df = all[all.is_train == 1][new_features]
    if use_scaler:
        plot_df[numeric_cols] = sc.inverse_transform(plot_df[numeric_cols])
    plot_df["pred"] = y_oof
    plot_df["power"] = train_y

    plot_df = apply_limiters(plot_df, is_validation=True)
    plot_df["pred_diff"] = (plot_df["pred"] - plot_df["power"]).abs()
    plot_df.to_csv(f"fuck_up_{trial}.csv", sep=",", index=False)
    plot_df.sort_values("pred_diff", ascending=False)

    plt.figure(figsize=(32, 12))
    sns.distplot(
        plot_df[:100]["power"],
        hist=True,
        color="blue",
        kde=True,
        bins=30,
        label="Power(kW)",
    )
    sns.distplot(
        plot_df[:100]["pred"],
        hist=True,
        color="red",
        kde=True,
        bins=30,
        label="predictions",
    )
    plt.legend()
    plt.show()

    sns.scatterplot(x=plot_df["pred"], y=plot_df["power"], hue=plot_df['Torque'].isna())
    plt.show()

    sns.scatterplot(x=plot_df["pred_diff"], y=plot_df["power"], hue=plot_df['Torque'].isna())
    plt.show()
    rmse = mse(plot_df["power"], plot_df["pred"], squared=False)
    print(f"Val Score: {rmse}")

elif model_type == "et":
    for train_ind, val_ind in kf.split(train_x, train_y):
        print(f"******* Fold {ix} ******* ")
        model = ExtraTreeRegressor(random_state=555, **best_params)
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

    plt.figure(figsize=(32, 12))
    sns.distplot(train_y, hist=True, color="blue", kde=True, bins=30, label="Power(kW)")
    sns.distplot(y_oof, hist=True, color="red", kde=True, bins=30, label="predictions")
    plt.legend()
    plt.show()

    for f in original_features:
        if f not in new_features:
            new_features.append(f)
    new_features.append('os_limit')
    new_features.append('pf_limit')
    new_features.append('sf_limit')
    new_features.append('turbine_limit')
    new_features.append('n-set_1_limit')
    new_features.append('scope_limit')
    plot_df = all[all.is_train == 1][new_features]
    if use_scaler:
        plot_df[numeric_cols] = sc.inverse_transform(plot_df[numeric_cols])
    plot_df["pred"] = y_oof
    plot_df["power"] = train_y

    plot_df = apply_limiters(plot_df, is_validation=True)
    plot_df["pred_diff"] = (plot_df["pred"] - plot_df["power"]).abs()
    plot_df.to_csv(f"fuck_up_{trial}.csv", sep=",", index=False)
    plot_df.sort_values("pred_diff", ascending=False)

    plt.figure(figsize=(32, 12))
    sns.distplot(
        plot_df[:100]["power"],
        hist=True,
        color="blue",
        kde=True,
        bins=30,
        label="Power(kW)",
    )
    sns.distplot(
        plot_df[:100]["pred"],
        hist=True,
        color="red",
        kde=True,
        bins=30,
        label="predictions",
    )
    plt.legend()
    plt.show()

    sns.scatterplot(x=plot_df["pred"], y=plot_df["power"], hue=plot_df['Torque'].isna())
    plt.show()

    sns.scatterplot(x=plot_df["pred_diff"], y=plot_df["power"], hue=plot_df['Torque'].isna())
    plt.show()
    rmse = mse(plot_df["power"], plot_df["pred"], squared=False)
    print(f"Val Score: {rmse}")

elif model_type == "hgbt":
    for train_ind, val_ind in kf.split(train_x, train_y):
        print(f"******* Fold {ix} ******* ")
        model = HistGradientBoostingRegressor(random_state=555, **best_params)
        tr_x, val_x = (
            train_x.iloc[train_ind].reset_index(drop=True),
            train_x.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            train_y.iloc[train_ind].reset_index(drop=True),
            train_y.iloc[val_ind].reset_index(drop=True),
        )

        model.fit(tr_x, tr_y)

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

    for f in original_features:
        if f not in new_features:
            new_features.append(f)
    new_features.append('os_limit')
    new_features.append('pf_limit')
    new_features.append('sf_limit')
    new_features.append('turbine_limit')
    new_features.append('n-set_1_limit')
    new_features.append('scope_limit')
    plot_df = all[all.is_train == 1][new_features]
    if use_scaler:
        plot_df[numeric_cols] = sc.inverse_transform(plot_df[numeric_cols])
    plot_df["pred"] = y_oof
    plot_df["power"] = train_y

    plot_df = apply_limiters(plot_df, is_validation=True)
    plot_df["pred_diff"] = (plot_df["pred"] - plot_df["power"]).abs()
    plot_df.to_csv(f"fuck_up_{trial}.csv", sep=",", index=False)
    plot_df.sort_values("pred_diff", ascending=False)

    plt.figure(figsize=(32, 12))
    sns.distplot(
        plot_df[:100]["power"],
        hist=True,
        color="blue",
        kde=True,
        bins=30,
        label="Power(kW)",
    )
    sns.distplot(
        plot_df[:100]["pred"],
        hist=True,
        color="red",
        kde=True,
        bins=30,
        label="predictions",
    )
    plt.legend()
    plt.show()

    sns.scatterplot(x=plot_df["pred"], y=plot_df["power"], hue=plot_df['Torque'].isna())
    plt.show()

    sns.scatterplot(x=plot_df["pred_diff"], y=plot_df["power"], hue=plot_df['Torque'].isna())
    plt.show()
    rmse = mse(plot_df["power"], plot_df["pred"], squared=False)
    print(f"Val Score: {rmse}")

elif model_type == "stacking":
    y_cat = np.zeros(train_x.shape[0])
    y_lgb = np.zeros(train_x.shape[0])
    y_cat_test = np.zeros(test_x.shape[0])
    y_lgb_test = np.zeros(test_x.shape[0])
    for train_ind, val_ind in kf.split(train_x, train_y):
        best_params = {
        "learning_rate": 0.02,
        "num_leaves": 64,
        "colsample_bytree": 0.9,
        "subsample": 0.9,
        "verbosity": -1,
        "n_estimators": 7000,
        "early_stopping_rounds": 50,
        "random_state": 42,
        "objective": "regression",
        "metric": "rmse",
    }
        print(f"******* Fold {ix} ******* ")
        model_1 = LGBMRegressor(**(best_params))
        model_2 = CatBoostRegressor(
            iterations=5000,
            random_state=42,
            early_stopping_rounds=50,
        )
        tr_x, val_x = (
            train_x.iloc[train_ind].reset_index(drop=True),
            train_x.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            train_y.iloc[train_ind].reset_index(drop=True),
            train_y.iloc[val_ind].reset_index(drop=True),
        )

        model_1.fit(
            tr_x,
            tr_y,
            eval_set=[(val_x, val_y)],
            eval_metric="rmse",
            early_stopping_rounds=50,
            verbose=100,
        )
        model_2.fit(
            tr_x,
            tr_y,
            eval_set=[(val_x, val_y)],
            early_stopping_rounds=50,
            verbose=100,
        )

        cat_pred = model_2.predict(val_x)
        lgb_pred = model_1.predict(val_x)
        preds = (cat_pred + lgb_pred) / 2
        y_cat[val_ind] = y_cat[val_ind] + cat_pred
        y_lgb[val_ind] = y_lgb[val_ind] + lgb_pred
        y_oof[val_ind] = y_oof[val_ind] + preds

        cat_pred_test = model_2.predict(test_x)
        lgb_pred_test = model_1.predict(test_x)
        test_preds = (cat_pred_test + lgb_pred_test) / 2
        y_test = y_test + test_preds / N_FOLDS
        y_cat_test = y_cat_test + cat_pred_test / N_FOLDS
        y_lgb_test = y_lgb_test + lgb_pred_test / N_FOLDS

        ix = ix + 1

    rmse = mse(y_oof, train_y, squared=False)
    print(f"Val Score after Level-1: {rmse}")

    train_x['cat_pred'] = y_cat
    train_x['lgb_pred'] = y_lgb
    test_x['cat_pred'] = y_cat_test
    test_x['lgb_pred'] = y_lgb_test


    y_cat = np.zeros(train_x.shape[0])
    y_lgb = np.zeros(train_x.shape[0])
    y_cat_test = np.zeros(test_x.shape[0])
    y_lgb_test = np.zeros(test_x.shape[0])
    y_oof = np.zeros(train_x.shape[0])
    y_test = np.zeros(test_x.shape[0])
    ix = 0

    for train_ind, val_ind in kf.split(train_x, train_y):
        print(f"******* Fold {ix} ******* ")
        model_1 = LGBMRegressor(**(best_params))
        model_2 = CatBoostRegressor(
            iterations=5000,
            random_state=42,
            early_stopping_rounds=50,
        )
        tr_x, val_x = (
            train_x.iloc[train_ind].reset_index(drop=True),
            train_x.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            train_y.iloc[train_ind].reset_index(drop=True),
            train_y.iloc[val_ind].reset_index(drop=True),
        )

        model_1.fit(
            tr_x,
            tr_y,
            eval_set=[(val_x, val_y)],
            eval_metric="rmse",
            early_stopping_rounds=50,
            verbose=100,
        )
        model_2.fit(
            tr_x,
            tr_y,
            eval_set=[(val_x, val_y)],
            early_stopping_rounds=50,
            verbose=100,
        )

        cat_pred = model_2.predict(val_x)
        lgb_pred = model_1.predict(val_x)
        preds = (cat_pred + lgb_pred) / 2
        y_cat[val_ind] = y_cat[val_ind] + cat_pred
        y_lgb[val_ind] = y_lgb[val_ind] + lgb_pred
        y_oof[val_ind] = y_oof[val_ind] + preds


        cat_pred_test = model_2.predict(test_x)
        lgb_pred_test = model_1.predict(test_x)
        test_preds = (cat_pred_test + lgb_pred_test) / 2
        y_test = y_test + test_preds / N_FOLDS
        y_cat_test = y_cat_test + cat_pred_test / N_FOLDS
        y_lgb_test = y_lgb_test + lgb_pred_test / N_FOLDS

        ix = ix + 1

    rmse = mse(y_oof, train_y, squared=False)
    print(f"Val Score: {rmse}")








sample_submission[submission_feature] = y_test
sample_submission = pd.merge(sample_submission, all.drop(submission_feature, axis=1), on="Timestamp", how="left")
sample_submission = apply_limiters(sample_submission)
sample_submission[submission_feature] = sample_submission[submission_feature] + power_offset
# sample_submission[sample_submission[classification_feature] == 1][submission_feature] = np.clip(sample_submission[sample_submission[classification_feature] == 1][submission_feature], sample_submission[sample_submission[classification_feature] == 1]['Internal Power Limit'] - lcap_threshold, sample_submission[sample_submission[classification_feature] == 1]['Internal Power Limit'] - 0.57666015625)
# sample_submission[submission_feature] = np.clip(
#     sample_submission[submission_feature],
#     power_offset,
#     sample_submission["Internal Power Limit"] - 0.57666015625,
# )

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
sample_submission[["Timestamp", submission_feature]].to_csv(f"submissions/submission_{model_type}_{trial}.csv", sep=",", index=False)
