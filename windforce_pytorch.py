##########################Load Libraries  ####################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import random
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torchmetrics

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
import os
import gc

gc.collect()
torch.cuda.empty_cache()
device = 'cuda'

# features = pd.read_csv('data/features.csv')
# power = pd.read_csv('data/power.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')

# all = pd.merge(features, power, on="Timestamp", how='left')
# all["is_train"] = all['Power(kW)'].notnull()
#
# power_offset = all['Power(kW)'].min()
#
# all['Timestamp'] = pd.to_datetime(all['Timestamp'])

# all.replace(to_replace=99999, value=np.nan, inplace=True)
# all = all.fillna(method='bfill')

# features = [
#        'Gearbox_T1_High_Speed_Shaft_Temperature',
#        'Gearbox_T3_High_Speed_Shaft_Temperature',
#        'Gearbox_T1_Intermediate_Speed_Shaft_Temperature',
#        'Temperature Gearbox Bearing Hollow Shaft', 'Tower Acceleration Normal',
#        'Gearbox_Oil-2_Temperature', 'Tower Acceleration Lateral',
#        'Temperature Bearing_A', 'Temperature Trafo-3',
#        'Gearbox_T3_Intermediate_Speed_Shaft_Temperature',
#        'Gearbox_Oil-1_Temperature', 'Gearbox_Oil_Temperature', 'Torque',
#        'Converter Control Unit Reactive Power', 'Temperature Trafo-2',
#        'Reactive Power', 'Temperature Shaft Bearing-1',
#        'Gearbox_Distributor_Temperature', 'Moment D Filtered',
#        'Moment D Direction', 'N-set 1', 'Operating State', 'Power Factor',
#        'Temperature Shaft Bearing-2', 'Temperature_Nacelle', 'Voltage A-N',
#        'Temperature Axis Box-3', 'Voltage C-N', 'Temperature Axis Box-2',
#        'Temperature Axis Box-1', 'Voltage B-N', 'Nacelle Position_Degree',
#        'Converter Control Unit Voltage', 'Temperature Battery Box-3',
#        'Temperature Battery Box-2', 'Temperature Battery Box-1',
#        'Hydraulic Prepressure', 'Angle Rotor Position',
#        'Temperature Tower Base', 'Pitch Offset-2 Asymmetric Load Controller',
#        'Pitch Offset Tower Feedback', 'Line Frequency', 'Internal Power Limit',
#        'Circuit Breaker cut-ins', 'Particle Counter',
#        'Tower Accelaration Normal Raw', 'Torque Offset Tower Feedback',
#        'External Power Limit', 'Blade-2 Actual Value_Angle-B',
#        'Blade-1 Actual Value_Angle-B', 'Blade-3 Actual Value_Angle-B',
#        'Temperature Heat Exchanger Converter Control Unit',
#        'Tower Accelaration Lateral Raw', 'Temperature Ambient',
#        'Nacelle Revolution', 'Pitch Offset-1 Asymmetric Load Controller',
#        'Tower Deflection', 'Pitch Offset-3 Asymmetric Load Controller',
#        'Wind Deviation 1 seconds', 'Wind Deviation 10 seconds',
#        'Proxy Sensor_Degree-135', 'State and Fault', 'Proxy Sensor_Degree-225',
#        'Blade-3 Actual Value_Angle-A', 'Scope CH 4',
#        'Blade-2 Actual Value_Angle-A', 'Blade-1 Actual Value_Angle-A',
#        'Blade-2 Set Value_Degree', 'Pitch Demand Baseline_Degree',
#        'Blade-1 Set Value_Degree', 'Blade-3 Set Value_Degree',
#        'Moment Q Direction', 'Moment Q Filltered', 'Proxy Sensor_Degree-45',
#        'Turbine State', 'Proxy Sensor_Degree-315'
# ]

features = [
       'Gearbox_T1_High_Speed_Shaft_Temperature', 'Nacelle Position_Degree_cos', 'Nacelle Revolution_sin', 'Nacelle Revolution_cos','Nacelle Revolution', 'Nacelle Position_Degree',
       'Gearbox_T3_High_Speed_Shaft_Temperature', 'Operating State', #"operation_12", "operation_11","operation_15","operation_16","operation_19",
       'Gearbox_T1_Intermediate_Speed_Shaft_Temperature', "n-set_1_0",
       'Temperature Gearbox Bearing Hollow Shaft', 'Tower Acceleration Normal',
       'Gearbox_Oil-2_Temperature', 'Tower Acceleration Lateral',
       'Temperature Bearing_A', 'Temperature Trafo-3',
       'Gearbox_T3_Intermediate_Speed_Shaft_Temperature',
       'Gearbox_Oil-1_Temperature', 'Gearbox_Oil_Temperature',
        'Torque',
       'Converter Control Unit Reactive Power', 'Temperature Trafo-2',
       'Reactive Power', 'Temperature Shaft Bearing-1',
       'Gearbox_Distributor_Temperature', 'Moment D Filtered',
       'Moment D Direction', 'N-set 1', 'Power Factor',
        #'Temperature Shaft Bearing-2', 'Temperature_Nacelle',
        #'Voltage A-N',
       #'Temperature Axis Box-3',
       #'Voltage C-N',
       'Temperature Axis Box-2',
       'Temperature Axis Box-1', #'Voltage B-N',
       'Converter Control Unit Voltage',
       #'Temperature Battery Box-3',
       #'Temperature Battery Box-2', 'Temperature Battery Box-1',
        #'Hydraulic Prepressure',
        'Angle Rotor Position',
       #'Temperature Tower Base',
      'Pitch Offset-2 Asymmetric Load Controller',
       'Pitch Offset Tower Feedback',
       #'Line Frequency',
       # 'Internal Power Limit',
       #'Circuit Breaker cut-ins',
       #'Particle Counter',
       #'Tower Accelaration Normal Raw',
       #'Torque Offset Tower Feedback',
       #'Blade-2 Actual Value_Angle-B', #'External Power Limit',
       #'Blade-1 Actual Value_Angle-B', 'Blade-3 Actual Value_Angle-B',
       'Temperature Heat Exchanger Converter Control Unit',
       'Tower Accelaration Lateral Raw', #'Temperature Ambient',
        'Pitch Offset-1 Asymmetric Load Controller',
       #'Tower Deflection',
        'Pitch Offset-3 Asymmetric Load Controller',
       'Wind Deviation 1 seconds', 'Wind Deviation 10 seconds',
       'Proxy Sensor_Degree-135', 'State and Fault', 'Proxy Sensor_Degree-225',
       'Blade-3 Actual Value_Angle-A', 'Scope CH 4',
       'Blade-2 Actual Value_Angle-A', 'Blade-1 Actual Value_Angle-A',
       'Blade-2 Set Value_Degree', 'Pitch Demand Baseline_Degree',
       'Blade-1 Set Value_Degree', 'Blade-3 Set Value_Degree',
       'Moment Q Direction', 'Moment Q Filltered', 'Proxy Sensor_Degree-45',
       'Turbine State', 'Proxy Sensor_Degree-315',
       #'month', 'hour', 'week', 'dayofweek', 'dayofyear'
]

all = pd.read_csv("all_knn.csv")

power_offset = -48.5966682434082

all['Timestamp'] = pd.to_datetime(all['Timestamp'])
sample_submission['Timestamp'] = pd.to_datetime(sample_submission['Timestamp'])

all['month'] = all['Timestamp'].dt.month
all['hour'] = all['Timestamp'].dt.hour
all['week'] = all['Timestamp'].dt.weekofyear
all['dayofweek'] = all['Timestamp'].dt.dayofweek
all['dayofyear'] = all['Timestamp'].dt.dayofyear
all['minute'] = all['Timestamp'].dt.hour * 60 + all['Timestamp'].dt.minute

all.replace(to_replace=99999, value=np.nan, inplace=True)
# all = all.fillna(method='ffill')
all['Internal Power Limit'] = all['Internal Power Limit'].fillna(method='ffill')

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


target_feature = "Power(kW)"
submission_feature = "Power(kW)"

def prep(df, features):
    new_features = features.copy()
    for feature in features:
        df[feature + '_lag1'] = df[feature].shift(1)
        df[feature + '_lead1'] = df[feature].shift(-1)
        df.fillna(0, inplace=True)
        df[feature + '_diff1'] = df[feature] - df[feature + '_lag1']
        new_features.append(feature + '_lag1')
        new_features.append(feature + '_lead1')
        new_features.append(feature + '_diff1')

    return df, new_features

all, new_features = prep(all, features)

train_df = all[all.is_train == 1]
test_df = all[all.is_train == 0]
test_df[target_feature] = 0

sc = StandardScaler()
train_df[new_features] = sc.fit_transform(train_df[new_features])
test_df[new_features] = sc.transform(test_df[new_features])

# F_train = df_train[features].values.reshape(df_train.shape[0] // 60, 60, len(features))
# F_test = df_test[features].values.reshape(df_test.shape[0] // 60, 60, len(features))

SEED = 22
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


trial = 'pt_2'
patiance = 7
seed_everything(SEED)

class GlobalMaxPooling1D(nn.Module):

    def __init__(self, data_format='channels_last'):
        super(GlobalMaxPooling1D, self).__init__()
        self.data_format = data_format
        self.step_axis = 1 if self.data_format == 'channels_last' else 2

    def forward(self, input):
        return torch.max(input, axis=self.step_axis).values

class GlobalAvgPooling1D(nn.Module):

    def __init__(self, data_format='channels_last'):
        super(GlobalAvgPooling1D, self).__init__()
        self.data_format = data_format
        self.step_axis = 1 if self.data_format == 'channels_last' else 2

    def forward(self, input):
        return torch.mean(input, dim=self.step_axis)


class WindforceNet(pl.LightningModule):
    def __init__(self):
        super(WindforceNet, self).__init__()

        self.bi_lstm1 = nn.LSTM(228, 768, bidirectional=True, batch_first=True, dropout=0.2)
        self.bi_lstm21 = nn.LSTM(1536, 512, bidirectional=True, batch_first=True)
        self.bi_lstm22 = nn.LSTM(228, 512, bidirectional=True, batch_first=True)
        self.bi_lstm31 = nn.LSTM(2048, 384, bidirectional=True, batch_first=True)
        self.bi_lstm32 = nn.LSTM(1024, 384, bidirectional=True, batch_first=True)

        self.dense = nn.Sequential(
            nn.BatchNorm1d(5120),
            nn.Linear(in_features=5120, out_features=128),
            nn.SELU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=1),
        )
        self.criterion = nn.MSELoss(reduce=True, reduction='mean')

        self.train_metric = torchmetrics.MeanSquaredError(squared=False)
        self.val_metric = torchmetrics.MeanSquaredError(squared=False)

    def forward(self, x):
        x_1, _ = self.bi_lstm1(x)
        x_21, _ = self.bi_lstm21(x_1)
        x_22, _ = self.bi_lstm22(x)
        x_2 = torch.cat([x_21, x_22], dim=1)

        x_31, _ = self.bi_lstm31(x_2)
        x_32, _ = self.bi_lstm32(x_21)
        x_3 = torch.cat([x_31, x_32], dim=1)

        x_5 = torch.cat([x_1, x_2, x_3], dim=1)
        output = self.dense(x_5)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_rmse_epoch"}

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.train_metric(y_hat, y.to(torch.int))
        self.log('train_metric', self.train_metric, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs):
        self.train_metric.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.val_metric(y_hat, y.to(torch.int))
        self.log('val_metric', self.val_metric.compute(), on_step=True, on_epoch=False)
        return loss

    def validation_epoch_end(self, outputs):
        self.log('valid_rmse_epoch', self.val_metric.compute(), prog_bar=True)
        self.val_metric.reset()

    def predict_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        return self(x)

class WindForcceDataset(Dataset):
    def __init__(self, df, is_test=False):
        self.indices = df.reset_index()['index']
        self.targets = df[target_feature]
        self.is_test = is_test
        self.df = df

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        X = self.df.iloc[self.indices[idx]][new_features]
        y = self.targets[idx]
        return torch.FloatTensor(X), torch.FloatTensor([y])

class WindForceDataLoader(pl.LightningDataModule):
    def __init__(self, df, batch_size=512, fold=None):
        super().__init__()
        self.batch_size = batch_size
        self.df = df
        self.fold = fold

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        dataset = WindForcceDataset(self.df)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=8, shuffle=True, drop_last=False)
        return train_loader

    def valid_dataloader(self):
        dataset = WindForcceDataset(self.df)
        valid_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=8, shuffle=False, drop_last=False)
        return valid_loader

    def test_dataloader(self):
        dataset = WindForcceDataset(self.df, is_test=True)
        test_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=8, shuffle=False, drop_last=False)
        return test_loader



N_FOLDS = 5

kf = KFold(N_FOLDS, shuffle=True, random_state=42)
y_oof = np.zeros(train_df.shape[0])
y_test = np.zeros(test_df.shape[0])

test_df = test_df.reset_index(drop=True)

ix = 0
for train_ind, val_ind in kf.split(train_df):
    print(f"******* Fold {ix} ******* ")
    tr_df, val_df = train_df.iloc[train_ind].reset_index(drop=True), train_df.iloc[val_ind].reset_index(drop=True)

    test_loader = WindForceDataLoader(test_df).test_dataloader()
    train_loader = WindForceDataLoader(tr_df).train_dataloader()
    val_loader = WindForceDataLoader(val_df).valid_dataloader()


    model = WindforceNet()

    early_stop_callback = EarlyStopping(monitor='valid_rmse_epoch', min_delta=0.00, patience=6, verbose=True, mode='min')
    rich_progress_bar_callback = RichProgressBar()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(limit_train_batches=0.5, callbacks=[early_stop_callback], max_epochs=50, gpus=1)
    trainer.fit(model, train_loader, val_loader)
    val_pred_list = trainer.predict(model, val_loader)
    val_pred = torch.cat(val_pred_list, dim=0).detach().cpu().numpy().ravel()
    test_pred_list = trainer.predict(model, test_loader)
    test_pred = torch.cat(test_pred_list, dim=0).detach().cpu().numpy().ravel()
    y_oof[val_ind] = val_pred
    y_test += test_pred / N_FOLDS
    ix = ix + 1

cv_mse = np.round(mse(train_df[target_feature], y_oof, squared=False), 4)
print("CV Val AUC:", cv_mse)
sample_submission[submission_feature] = y_test + power_offset
print(f"Submission mean: {sample_submission[submission_feature].mean()}")
sample_submission.to_csv(f"submissions/submission_{trial}.csv", sep=",", index=False)


