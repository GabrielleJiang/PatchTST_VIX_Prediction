import torch
import torch.nn as nn
import logging

# Basic logging setup
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class PatchTST(nn.Module):
    """
    PatchTST model for time series forecasting.
    
    This model uses patches and Transformer for time series prediction.
    
    Parameters:
    -----------
    seq_len: int, default=252
        Length of the input sequence.
    pred_len: int, default=21
        Length of the output prediction.
    patch_len: int, default=21
        Length of each patch (window).
    stride: int, default=21
        Step size for patch extraction.
    d_model: int, default=128
        Dimension of the Transformer model.
    n_heads: int, default=8
        Number of attention heads in the Transformer encoder.
    num_layers: int, default=2
        Number of Transformer encoder layers.
    dropout: float, default=0.3
        Dropout rate for regularization.
    """

    def __init__(
        self,
        seq_len: int = 252,
        pred_len: int = 21,
        patch_len: int = 21,
        stride: int = 21,
        d_model: int = 128,
        n_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super(PatchTST, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride

        self.num_patches = (seq_len - patch_len) // stride + 1
        logger.info(f"Input length: {seq_len}, Number of patches: {self.num_patches}")

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_len, d_model)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers
        self.fc1 = nn.Linear(d_model * self.num_patches, 128)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(128, pred_len)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PatchTST model.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape [batch_size, seq_len].

        Returns:
        --------
        torch.Tensor
            Predicted output of shape [batch_size, pred_len].
        """
        try:
            batch_size = x.size(0)

            if len(x.shape) == 3:
                x = x.squeeze(1)

            patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
            
            # Apply patch embedding
            embedded = self.patch_embedding(patches)
            
            # Add positional encoding
            embedded = embedded + self.pos_embedding

            transformed = self.transformer(embedded)

            flattened = transformed.reshape(batch_size, -1)

            x = self.fc1(flattened)
            x = self.layer_norm(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.fc2(x)

            return x

        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            logger.error(f"Input shape: {x.shape}")
            raise e

