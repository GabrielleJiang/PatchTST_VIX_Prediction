# PatchTST Time Series Prediction

## Overview
This project implements the PatchTST (Patch Time Series Transformer) model for time series forecasting. The model is specifically applied to VIX (Volatility Index) prediction, demonstrating how transformer-based architectures can be effectively used for financial time series forecasting.

## Features
- Implementation of PatchTST model for time series forecasting
- Data preprocessing pipeline for financial time series data
- Training with mixup data augmentation for improved generalization
- Evaluation metrics including MAE, RMSE, and R²
- Visualization of prediction results

## Project Structure
```
PatchTST_Prediction/
├── results/               # Model implementation and results
│   ├── model.py           # PatchTST model implementation
│   └── patchtst.py        # Training and evaluation pipeline
├── requirements.txt       # Project dependencies
└── README.txt             # Project documentation
```

## Installation
1. Clone the repository:
```
git clone https://github.com/yourusername/PatchTST_Prediction.git
cd PatchTST_Prediction
```

2. Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Usage
1. Prepare your data in Excel format with columns for date and VIX values.
2. Update the file path in the main function of `results/patchtst.py`.
3. Run the training and evaluation:
```
python results/patchtst.py
```

## Model Architecture
The PatchTST model consists of:
- Patch embedding layer to convert time series segments into embeddings
- Positional encoding to maintain temporal information
- Transformer encoder for capturing long-range dependencies
- Fully connected layers for prediction

## Hyperparameters
- Sequence length: 252 (approximately one trading year)
- Prediction length: 21 (approximately one trading month)
- Patch length: 21
- Stride: 21
- Model dimension: 128
- Number of attention heads: 8
- Number of transformer layers: 2
- Dropout rate: 0.3

## Results
The model is evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

