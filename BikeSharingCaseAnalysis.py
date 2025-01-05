# BIKE RENTAL CASE ANALYSIS - TEAM22 FINAL PROJECT

# -----------------------------
# 1. Import Required Libraries
# -----------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tensorflow.keras.backend as K

# -----------------------------
# 2. Load and Preview Dataset
# -----------------------------
data = pd.read_csv('hour.csv')
print(data.head())
print(data.info())

# -----------------------------
# 3. Exploratory Data Analysis
# -----------------------------
# Summary statistics
print(data.describe())

# Set the plot style
sns.set(style="whitegrid")

# Histogram of Bike-Sharing Count
plt.figure(figsize=(10, 6))
sns.histplot(data['cnt'], bins=100, color='salmon')
plt.title('Histogram of Bike-Sharing Count')
plt.xlabel('Bikes Rented')
plt.ylabel('Count')
plt.show()

# Box Plot of Bike-Sharing Count
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['cnt'], color='green')
plt.title('Box Plot of Bike-Sharing Count')
plt.xlabel('Bikes Rented in an Hour')
plt.tight_layout()
plt.show()

# Correlation Heatmap
correlation_matrix = data.select_dtypes(include='number').corr()
plt.figure(figsize=(10, 10))
ax = sns.heatmap(
    correlation_matrix, annot=True, cmap='flare', fmt=".2f", annot_kws={'size': 8}
)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.setp(ax.get_yticklabels(), rotation=0)
plt.title('Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.show()

# Time Series Plot of Total Rentals
data['dteday'] = pd.to_datetime(data['dteday'])
plt.figure(figsize=(14, 6))
sns.lineplot(data=data, x='dteday', y='cnt', label='Total Rentals')
plt.title('Total Rentals Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Rentals')
plt.show()

# Distribution of Bike Rentals per Hour
plt.figure(figsize=(15, 6))
sns.boxplot(x='hr', y='cnt', data=data)
plt.title('Distribution of Bike Rentals per Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Bike Rentals Count')
plt.show()

# Seasonal Trends in Bike Rentals
data['season_code'] = data['season'].map({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
plt.figure(figsize=(15, 6))
sns.barplot(x='season_code', y='cnt', data=data, errorbar=None, color='green')
plt.title('Bike Rentals Across Seasons')
plt.xlabel('Season')
plt.ylabel('Total Rentals')
plt.show()

# Monthly Rental Trends
data['month_name'] = data['mnth'].map({
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
    5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
    9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
})
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='month_name', y='cnt', estimator='mean', errorbar=None, label='Average Rentals')
plt.title('Monthly Rental Trends')
plt.xlabel('Month')
plt.ylabel('Average Rentals')
sns.despine()
plt.show()

# Weather Impact on Rentals
data['weather_description'] = data['weathersit'].map({
    1: 'Clear/Partly Cloudy',
    2: 'Mist/Cloudy',
    3: 'Light Snow/Light Rain',
    4: 'Heavy Rain/Snow/Fog'
})
plt.figure(figsize=(15, 6))
sns.boxplot(data=data, x='weather_description', y='cnt')
plt.title('Weather Impact on Rentals')
plt.xlabel('Weather Situation')
plt.ylabel('Number of Rentals')
plt.show()

# -----------------------------
# 4. Feature Engineering
# -----------------------------
# Rename and drop unnecessary columns
data = data.rename(columns={'instant': 'ID'})
data = data.drop(columns=['dteday', 'casual', 'registered', 'atemp'])

# One-hot encode categorical features
data = pd.get_dummies(data, columns=['season', 'weathersit', 'weekday'], dtype=int)

# Normalize numeric columns
for col in ['temp', 'hum', 'windspeed']:
    data[col] = (data[col] - data[col].mean()) / data[col].std()

# Create a normalized target variable
data['cnt_normalized'] = data['cnt'] / 100

# Train-test split
features = [col for col in data.columns if col != 'cnt_normalized']
target = 'cnt_normalized'
X = data[features].values
y = data[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
train_features = torch.tensor(X_train, dtype=torch.float32)
train_labels = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
test_features = torch.tensor(X_test, dtype=torch.float32)
test_labels = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

# -----------------------------
# 5. Define Loss Functions
# -----------------------------
# RMSLE Loss
class RMSLELoss(nn.Module):
    def __init__(self):
        super(RMSLELoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=0)
        y_true = torch.clamp(y_true, min=0)
        log_pred = torch.log1p(y_pred)
        log_true = torch.log1p(y_true)
        return torch.sqrt(torch.mean((log_pred - log_true) ** 2))

criterion = RMSLELoss()

# Metrics
def calculate_mape(y_pred, y_true):
    """Mean Absolute Percentage Error (MAPE)."""
    mape = torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100
    return mape.item()

def calculate_r_squared(y_pred, y_true):
    """R-squared (Coefficient of Determination)."""
    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total).item()

# -----------------------------
# 6. Training Function
# -----------------------------
def train_with_validation(model, train_loader, valid_loader, num_epochs, learning_rate=0.001, weight_decay=0.01):
    train_losses = []
    valid_losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Average training loss
        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        total_valid_loss = 0.0
        with torch.no_grad():
            for features, labels in valid_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                total_valid_loss += loss.item()

        # Average validation loss
        valid_loss = total_valid_loss / len(valid_loader)
        valid_losses.append(valid_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")

    return train_losses, valid_losses

# -----------------------------
# 7. Baseline Model: Linear Regression
# -----------------------------

# Prepare data loaders
batch_size = 64
train_data = TensorDataset(train_features, train_labels)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

valid_data = TensorDataset(test_features, test_labels)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

# Build Linear Model
linear_model = nn.Linear(train_features.shape[1], 1)

# Move model to device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
linear_model = linear_model.to(device)

# Train Linear Model
train_losses_lin, valid_losses_lin = train_with_validation(
    linear_model, train_loader, valid_loader, num_epochs=100, learning_rate=0.001
)

# Plot Training and Validation Losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses_lin, label='Training Loss')
plt.plot(valid_losses_lin, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('RMSLE')
plt.title('Linear Model Losses')
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 8. Multi-Layer Perceptron (MLP) Model
# -----------------------------

# Build Multi-Layer Perceptron (MLP) Model
mlp_model = nn.Sequential(
    nn.Linear(train_features.shape[1], 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 32),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 1)
).to(device)

# Train MLP Model
train_losses_mlp, valid_losses_mlp = train_with_validation(
    mlp_model, train_loader, valid_loader, num_epochs=100, learning_rate=0.001
)

# Plot Training and Validation Losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses_mlp, label='Training Loss')
plt.plot(valid_losses_mlp, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('RMSLE')
plt.title('MLP Model Losses')
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 9. Recurrent Neural Network (RNN) Model
# -----------------------------

# Prepare time series data for RNN
def create_rnn_dataset(data, time_step=24):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data.iloc[i:(i + time_step)][features].values)
        Y.append(data.iloc[i + time_step][target])
    return np.array(X), np.array(Y)

time_step = 24
X_train_rnn, y_train_rnn = create_rnn_dataset(data.iloc[:len(X_train)], time_step)
X_test_rnn, y_test_rnn = create_rnn_dataset(data.iloc[len(X_train):], time_step)

# Reshape input to [samples, time_steps, features]
X_train_rnn = X_train_rnn.reshape(X_train_rnn.shape[0], time_step, len(features))
X_test_rnn = X_test_rnn.reshape(X_test_rnn.shape[0], time_step, len(features))

# Build RNN Model
rnn_model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(time_step, len(features))),
    Dense(1)
])
rnn_model.compile(optimizer='adam', loss=rmsle)

# Train RNN Model
rnn_history = rnn_model.fit(
    X_train_rnn, y_train_rnn, validation_split=0.2,
    epochs=50, batch_size=32, verbose=1
)

# Extract Losses
train_loss_rnn = rnn_history.history['loss']
val_loss_rnn = rnn_history.history['val_loss']

# Plot Training and Validation Losses
plt.figure(figsize=(10, 6))
plt.plot(train_loss_rnn, label='Training Loss')
plt.plot(val_loss_rnn, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('RMSLE')
plt.title('RNN Model Losses')
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 10. Long Short-Term Memory (LSTM) Model
# -----------------------------

# Build LSTM Model
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(time_step, len(features))),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss=rmsle)

# Train LSTM Model
lstm_history = lstm_model.fit(
    X_train_rnn, y_train_rnn, validation_split=0.2,
    epochs=50, batch_size=32, verbose=1
)

# Extract Losses
train_loss_lstm = lstm_history.history['loss']
val_loss_lstm = lstm_history.history['val_loss']

# Plot Training and Validation Losses
plt.figure(figsize=(10, 6))
plt.plot(train_loss_lstm, label='Training Loss')
plt.plot(val_loss_lstm, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('RMSLE')
plt.title('LSTM Model Losses')
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 11. Predictions and Evaluation
# -----------------------------

# Predictions for RNN and LSTM Models
y_test_pred_rnn = rnn_model.predict(X_test_rnn)
y_test_pred_lstm = lstm_model.predict(X_test_rnn)

# Plot Predictions vs Actual for RNN
plt.figure(figsize=(15, 6))
plt.plot(y_test_rnn[:100], label="True Values", color='black')
plt.plot(y_test_pred_rnn[:100], label="RNN Predictions", color='orange')
plt.title("RNN Predictions vs True Values")
plt.xlabel("Samples")
plt.ylabel("Normalized Bike Count")
plt.legend()
plt.show()

# Plot Predictions vs Actual for LSTM
plt.figure(figsize=(15, 6))
plt.plot(y_test_rnn[:100], label="True Values", color='black')
plt.plot(y_test_pred_lstm[:100], label="LSTM Predictions", color='green')
plt.title("LSTM Predictions vs True Values")
plt.xlabel("Samples")
plt.ylabel("Normalized Bike Count")
plt.legend()
plt.show()

# Print Metrics for RNN
train_mape_rnn = calculate_mape(torch.tensor(y_train_rnn), torch.tensor(rnn_model.predict(X_train_rnn)))
test_mape_rnn = calculate_mape(torch.tensor(y_test_rnn), torch.tensor(y_test_pred_rnn))
print(f"RNN Train MAPE: {train_mape_rnn:.2f}%, Test MAPE: {test_mape_rnn:.2f}%")

# Print Metrics for LSTM
train_mape_lstm = calculate_mape(torch.tensor(y_train_rnn), torch.tensor(lstm_model.predict(X_train_rnn)))
test_mape_lstm = calculate_mape(torch.tensor(y_test_rnn), torch.tensor(y_test_pred_lstm))
print(f"LSTM Train MAPE: {train_mape_lstm:.2f}%, Test MAPE: {test_mape_lstm:.2f}%")
