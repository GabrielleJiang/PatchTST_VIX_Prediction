import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Dict, List, Optional
import time
import random
from model import PatchTST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VIXDataset(Dataset):
    """
    Custom dataset for VIX time series data.

    Parameters:
    -----------
    data : np.ndarray
        A NumPy array of shape (num_samples, sequence_length) containing the input time series
        data (e.g., historical VIX values).
    targets : np.ndarray
        A NumPy array of shape (num_samples, prediction_length) containing the target values
        corresponding to the input sequences.
    """

    def __init__(self, data: np.ndarray, targets: np.ndarray):
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    This function reads an Excel file specified by 'file_path', processes the data for time 
    series forecasting, and splits it into training, validation, and test sets. The expected 
    Excel file should contain at least the columns 'dt' and 'vix', which are renamed to 'time' 
    and 'VIX' respectively.

    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing VIX data.

    Returns:
    --------
    pd.DataFrame
        Preprocessed DataFrame with VIX data.
    Dict
        Dictionary containing train, validation, and test split information.
    """
    try:
        logger.info(f"Loading data from {file_path}")

        file_path = os.path.expanduser(file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        df = pd.read_excel(file_path)
        df = df.rename(columns={'dt': 'time', 'vix': 'VIX'})
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        df = df.set_index('time').resample('D').asfreq()
        df['VIX'] = df['VIX'].fillna(method='ffill').fillna(method='bfill')
        if df['VIX'].isna().any():
            mean_vix = df['VIX'].mean()
            df['VIX'] = df['VIX'].fillna(mean_vix)
            logger.warning(f"Filled missing values with mean: {mean_vix}")
        df = df.reset_index()
        train_end = pd.Timestamp('2015-12-31')
        val_end = pd.Timestamp('2016-12-31')
        split_info = {
            'train': df[df['time'] <= train_end],
            'val': df[(df['time'] > train_end) & (df['time'] <= val_end)],
            'test': df[df['time'] > val_end]
        }

        return df, split_info

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def create_supervised_samples(df: pd.DataFrame, seq_len: int, pred_len: int) -> Tuple[Dict[str, np.ndarray], StandardScaler]:
    """
    Create supervised learning samples using a sliding window approach.

    This function takes pre-split data, standardizes the VIX values with a StandardScaler, and constructs input-target pairs by sliding a window
    of length `seq_len` to predict the next `pred_len` values.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with VIX data.
    seq_len : int
        Input sequence length (lookback window).
    pred_len : int
        Prediction sequence length (forecast horizon).

    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing training, validation, and test datasets.
    StandardScaler
        Fitted scaler for inverse transformation.
    """
    try:
        train_data, val_data, test_data = df['train'], df['val'], df['test']
        logger.info(f"Creating supervised samples with seq_len={seq_len}, pred_len={pred_len}")
        if len(train_data) < seq_len + pred_len:
            raise ValueError(f"Training data length ({len(train_data)}) is less than required (seq_len + pred_len = {seq_len + pred_len})")
        
        train_values = train_data['VIX'].values
        val_values = val_data['VIX'].values
        test_values = test_data['VIX'].values
        
        logger.info(f"Data shapes - Train: {train_values.shape}, Val: {val_values.shape}, Test: {test_values.shape}")

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_values.reshape(-1, 1)).flatten()
        val_scaled = scaler.transform(val_values.reshape(-1, 1)).flatten()
        test_scaled = scaler.transform(test_values.reshape(-1, 1)).flatten()
        
        logger.info("Data standardization completed")

        def create_windows(data, seq_len, pred_len):
            X, y = [], []
            for i in range(len(data) - seq_len - pred_len + 1):
                X.append(data[i:i+seq_len])
                y.append(data[i+seq_len:i+seq_len+pred_len])
            return np.array(X), np.array(y)

        X_train, y_train = create_windows(train_scaled, seq_len, pred_len)
        X_val, y_val = create_windows(val_scaled, seq_len, pred_len)
        X_test, y_test = create_windows(test_scaled, seq_len, pred_len)
        
        logger.info(f"Created window samples - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        return {'train': {'X': X_train, 'y': y_train},
                'val': {'X': X_val, 'y': y_val},
                'test': {'X': X_test, 'y': y_test}}, scaler
                
    except Exception as e:
        logger.error(f"Error in create_supervised_samples: {str(e)}")
        raise


def mixup_data(x, y, alpha=0.2):
    """
    Performs mixup augmentation on the input data.
    
    Parameters:
    -----------
    x : torch.Tensor
        Input features of shape [batch_size, seq_len].
    y : torch.Tensor
        Target values of shape [batch_size, pred_len].
    alpha : float, default=0.2
        Parameter for beta distribution to sample lambda.
        
    Returns:
    --------
    mixed_x : torch.Tensor
        Mixed input features.
    mixed_y : torch.Tensor
        Mixed target values.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    
    batch_size = x.size(0)

    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y


def train_model(train_loader, val_loader, model, device, lr=0.001, max_epochs=100, patience=10, use_mixup=True, mixup_alpha=0.2):
    """
    Train the PatchTST model with optional mixup data augmentation.
    
    Parameters:
    -----------
    train_loader : DataLoader
        DataLoader for training data.
    val_loader : DataLoader
        DataLoader for validation data.
    model : nn.Module
        The model to train.
    device : torch.device
        Device to use for training.
    lr : float, default=0.001
        Learning rate.
    max_epochs : int, default=100
        Maximum number of epochs.
    patience : int, default=10
        Patience for early stopping.
    use_mixup : bool, default=True
        Whether to use mixup data augmentation.
    mixup_alpha : float, default=0.2
        Alpha parameter for mixup.
    """
    criterion = nn.MSELoss()
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_val_loss = float('inf')
    counter = 0
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            if use_mixup and random.random() < 0.5:
                X_batch, y_batch = mixup_data(X_batch, y_batch, alpha=mixup_alpha)
            
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()
        
        scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info("Best model saved")
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(torch.load('best_model.pth'))
                break


def evaluate_model(model, test_loader, scaler, device, y_true_scaled):
    """
    Evaluate the trained model on test data.
    
    Parameters:
    -----------
    model : nn.Module
        The trained model.
    test_loader : DataLoader
        DataLoader for test data.
    scaler : StandardScaler
        Scaler used to transform the data.
    device : torch.device
        Device to use for evaluation.
    y_true_scaled : np.ndarray
        True scaled values for the test set.
    """
    try:
        model.eval()
        predictions = []
        
        logger.info("Starting model evaluation...")
        
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                y_pred = model(X_batch)
                predictions.append(y_pred.cpu().numpy())
        
        y_pred_scaled = np.vstack(predictions)
        logger.info(f"Predictions shape: {y_pred_scaled.shape}, True values shape: {y_true_scaled.shape}")
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
        y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).reshape(y_true_scaled.shape)

        mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true.flatten(), y_pred.flatten())
        
        logger.info(f"Test MAE: {mae:.4f}")
        logger.info(f"Test RMSE: {rmse:.4f}")
        logger.info(f"Test R²: {r2:.4f}")
        
        plt.figure(figsize=(12, 6))
        sample_idx = 0
        plt.plot(y_true[sample_idx], label='Actual')
        plt.plot(y_pred[sample_idx], label='Predicted')
        plt.title('VIX Prediction - Test Sample')
        plt.xlabel('Days')
        plt.ylabel('VIX')
        plt.legend()
        plt.savefig('vix_prediction.png')
        plt.close()
        
        logger.info("Evaluation completed and plot saved.")
        
    except Exception as e:
        logger.error(f"Error in evaluate_model: {str(e)}")
        raise


def main():
    """
    Main function to execute the entire workflow: loading data, training the model, 
    evaluating it, and saving the results.
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        file_path = '~/Desktop/PS_FinHub/vixdata.xlsx'
        logger.info(f"Attempting to load data from: {file_path}")
        expanded_path = os.path.expanduser(file_path)
        logger.info(f"Expanded path: {expanded_path}")
        
        if not os.path.exists(expanded_path):
            logger.error(f"File not found: {expanded_path}")
            logger.info("Please ensure the file exists at the specified location.")
            return
            
        df, split_info = load_and_preprocess_data(file_path)
        logger.info(f"Data loaded successfully. Train size: {len(split_info['train'])}, Val size: {len(split_info['val'])}, Test size: {len(split_info['test'])}")

        data_dict, scaler = create_supervised_samples(split_info, seq_len=252, pred_len=21)
        train_dataset = VIXDataset(data_dict['train']['X'], data_dict['train']['y'])
        val_dataset = VIXDataset(data_dict['val']['X'], data_dict['val']['y'])
        test_dataset = VIXDataset(data_dict['test']['X'], data_dict['test']['y'])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = PatchTST(
            seq_len=252, 
            pred_len=21, 
            patch_len=21, 
            stride=21,
            d_model=128, 
            n_heads=8, 
            num_layers=2, 
            dropout=0.3
        ).to(device)

        train_model(
            train_loader, 
            val_loader, 
            model, 
            device, 
            lr=0.001,           # Learning rate
            max_epochs=100,     # Maximum number of epochs
            patience=10,        # Early stopping patience
            use_mixup=True,     # Enable mixup data augmentation
            mixup_alpha=0.2     # Mixup alpha parameter
        )

        evaluate_model(model, test_loader, scaler, device, data_dict['test']['y'])

        torch.save(model.state_dict(), 'vix_patchtst_model.pth')
        logger.info("Model saved successfully.")

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise


if __name__ == "__main__":
    main()
