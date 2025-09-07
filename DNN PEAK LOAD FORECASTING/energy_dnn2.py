#MEKA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


#Load
data = pd.read_csv('energydata_complete.csv')

#Parse time + engineer features
# Expect a 'date' column like '2016-01-11 17:00:00'
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')
    data['hour'] = data['date'].dt.hour
    data['dayofweek'] = data['date'].dt.dayofweek
    data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)
else:
    # Fallback: assume data already in order
    data['hour'] = 0
    data['dayofweek'] = 0
    data['is_weekend'] = 0

#Core predictors (you can add more rooms if available)
base_feats = ['T1','RH_1','T2','RH_2','T3','RH_3', 'T4','RH_4', 'T5','RH_5','T6','RH_6','T7',
              'RH_7','T8','RH_8','T9','RH_9','T_out','RH_out',
              'Press_mm_hg','Windspeed','Visibility','Tdewpoint', "rv1", "rv2",
              'hour','dayofweek','is_weekend']

#Target
y_col = 'Appliances'
assert y_col in data.columns, "Column 'Appliances' missing."

#Lags & rolling statistics (helps a lot)
def add_lags(df, col, lags = [1, 2, 3]):
    for L in lags:
        df[f'{col}_lag{L}'] = df[col].shift(L)
def add_rolling(df, col, windows=[3,6,12]):
    for w in windows:
        df[f'{col}_roll{w}'] = df[col].rolling(w, min_periods=1).mean()

add_lags(data, y_col, lags=[1,2,3])
add_rolling(data, y_col, windows=[3,6,12])
feat_cols = base_feats + [f'{y_col}_lag1',f'{y_col}_lag2',f'{y_col}_lag3',
                          f'{y_col}_roll3',f'{y_col}_roll6',f'{y_col}_roll12']

#Drop rows with NaNs created by lags
data = data.dropna(subset=feat_cols + [y_col]).reset_index(drop=True)

X = data[feat_cols].values
y = data[y_col].values.reshape(-1,1)

#Chronological split: 70% train, 15% val, 15% test
n = len(data)
n_train = int(0.70 * n)
n_val   = int(0.85 * n)

X_train, X_val, X_test = X[:n_train], X[n_train:n_val], X[n_val:]
y_train, y_val, y_test = y[:n_train], y[n_train:n_val], y[n_val:]

# 8) Fit scalers **only on train** (avoid leakage)
scaler_X = MinMaxScaler()
scaler_y = StandardScaler()   # std scaling often stabilizes regression

X_train_s = scaler_X.fit_transform(X_train)
X_val_s   = scaler_X.transform(X_val)
X_test_s  = scaler_X.transform(X_test)

y_train_s = scaler_y.fit_transform(y_train)
y_val_s   = scaler_y.transform(y_val)
y_test_s  = scaler_y.transform(y_test)

#Naive baseline (persistence using lag1 already in features)
#    Compare to simply predicting previous value
naive_pred = data[f'{y_col}_lag1'].values[n_val:].reshape(-1,1)
naive_rmse = np.sqrt(mean_squared_error(y_test.ravel(), naive_pred.ravel()))

#Build compact, regularized DNN
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_s.shape[1],)),
    Dropout(0.025), ##MA:0.1
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=5e-3), loss='mse', metrics=[]) ##MA: 1e-3

#Train with early stopping + LR scheduling
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-5, verbose=1)
]
history = model.fit(
    X_train_s, y_train_s,
    validation_data=(X_val_s, y_val_s),
    epochs=550,
    batch_size=32,
    verbose=1,
    callbacks=callbacks
)


#valuate
y_pred_s = model.predict(X_test_s)
y_pred = scaler_y.inverse_transform(y_pred_s)
mse = mean_squared_error(y_test.ravel(), y_pred.ravel())
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test.ravel(), y_pred.ravel())

print(f"Naive RMSE: {naive_rmse:.3f}")
print(f"Test MSE:   {mse:.3f}")
print(f"Test RMSE:  {rmse:.3f}")
print(f"Test MAE:   {mae:.3f}")

#Get matching time index for test set
time_index = data["date"].iloc[n_val:].reset_index(drop=True) #matches y_test

#Predictions
y_pred_s = model.predict(X_test_s)
y_pred = scaler_y.inverse_transform(y_pred_s)

#Plot actual vs predicted
plt.figure(figsize=(14,6))
plt.plot(time_index[:200], y_test[:200], label="Actual Peak Load", marker="o")
plt.plot(time_index[:200], y_pred[:200], label="Predicted Peak Load", marker="x")

#Formatting
plt.title("DNN Peak Load Forecasting", fontsize=14)
plt.xlabel("Time")
plt.ylabel("Appliance Load (Wh)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

#Format x-axis to show hours (HH:MM)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.savefig('DNN Peak Load Forecasting.png', dpi=300)
plt.show()


# (Optional) Save
model.save('energy_dnn_regularized.h5')
print("Model saved as 'energy_dnn_regularized.h5'")