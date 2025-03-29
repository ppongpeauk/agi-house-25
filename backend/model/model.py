import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Dict
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OutbreakDataset(Dataset):
    """Dataset for disease outbreak prediction."""

    def __init__(
        self,
        data_path: str,
        sequence_length: int = 7,
        disease_vocab: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the dataset.

        Args:
            data_path (str): Path to the CSV data file
            sequence_length (int): Number of time steps to use for prediction
            disease_vocab (Optional[Dict[str, int]]): Dictionary mapping disease names to indices
        """
        self.sequence_length = sequence_length
        self.data = pd.read_csv(data_path)

        # Create disease vocabulary if not provided
        if disease_vocab is None:
            unique_diseases = self.data['disease_type'].unique()
            self.disease_vocab = {disease: idx for idx, disease in enumerate(unique_diseases)}
        else:
            self.disease_vocab = disease_vocab

        # Normalize numerical features
        self.feature_columns = ['temperature', 'humidity', 'precipitation', 'wind_speed']
        self.feature_means = self.data[self.feature_columns].mean()
        self.feature_stds = self.data[self.feature_columns].std()
        self.data[self.feature_columns] = (self.data[self.feature_columns] - self.feature_means) / self.feature_stds

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            Tuple containing:
            - Input features tensor
            - Target tensor (case count and risk score)
            - Disease type index tensor
        """
        # Get sequence of features
        features = self.data[self.feature_columns].iloc[idx:idx + self.sequence_length].values

        # Get target values (case count and risk score)
        target_case_count = self.data['disease_incidence'].iloc[idx + self.sequence_length]
        target_risk_score = self.data['is_outbreak'].iloc[idx + self.sequence_length]
        target = torch.tensor([target_case_count, target_risk_score], dtype=torch.float32)

        # Get disease type index
        disease_type = self.data['disease_type'].iloc[idx + self.sequence_length]
        disease_idx = torch.tensor([self.disease_vocab[disease_type]], dtype=torch.long)

        return torch.FloatTensor(features), target, disease_idx

class OutbreakPredictor(nn.Module):
    """
    LSTM-based model for predicting disease outbreaks.
    Predicts both case counts and risk scores (0-1) for potential outbreaks.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        disease_vocab_size: int,
        disease_embed_dim: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.disease_vocab_size = disease_vocab_size
        self.disease_embed_dim = disease_embed_dim

        # Disease type embedding
        self.disease_embedding = nn.Embedding(disease_vocab_size, disease_embed_dim)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size + disease_embed_dim,  # Concatenate with disease embedding
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, disease_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)
            disease_idx (Optional[torch.Tensor]): Disease type indices of shape (batch_size,)

        Returns:
            torch.Tensor: Predictions of shape (batch_size, output_size)
        """
        batch_size = x.size(0)
        seq_length = x.size(1)

        # Handle disease embedding
        if disease_idx is not None:
            # Get disease embeddings
            disease_embed = self.disease_embedding(disease_idx)  # (batch_size, embed_dim)

            # Expand disease embedding to match sequence length
            disease_embed_expanded = disease_embed.expand(batch_size, seq_length, -1)

            # Concatenate with input
            x = torch.cat((x, disease_embed_expanded), dim=2)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Use only the last output for prediction
        last_hidden = lstm_out[:, -1, :]

        # Get predictions
        output = self.fc(last_hidden)

        # Split output into case count and risk score
        case_count = output[:, 0]
        risk_score = torch.sigmoid(output[:, 1])  # Ensure risk score is between 0 and 1

        return torch.stack([case_count, risk_score], dim=1)

def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate various metrics for model evaluation.

    Args:
        predictions (torch.Tensor): Model predictions [case_count, risk_score]
        targets (torch.Tensor): Ground truth values [case_count, risk_score]

    Returns:
        Dict[str, float]: Dictionary containing various metrics
    """
    # Split predictions and targets
    case_count_pred, risk_score_pred = predictions[:, 0], predictions[:, 1]
    case_count_target, risk_score_target = targets[:, 0], targets[:, 1]

    # Case count metrics
    case_count_mae = torch.mean(torch.abs(case_count_pred - case_count_target)).item()
    case_count_rmse = torch.sqrt(torch.mean((case_count_pred - case_count_target) ** 2)).item()

    # Risk score metrics (using 0.5 as threshold)
    risk_pred_binary = (risk_score_pred > 0.5).float()
    risk_target_binary = (risk_score_target > 0.5).float()

    # Calculate accuracy, precision, recall, and F1
    correct = (risk_pred_binary == risk_target_binary).sum().item()
    total = len(risk_pred_binary)
    accuracy = correct / total

    # Calculate precision and recall
    true_positives = ((risk_pred_binary == 1) & (risk_target_binary == 1)).sum().item()
    predicted_positives = (risk_pred_binary == 1).sum().item()
    actual_positives = (risk_target_binary == 1).sum().item()

    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'case_count_mae': case_count_mae,
        'case_count_rmse': case_count_rmse,
        'risk_accuracy': accuracy,
        'risk_precision': precision,
        'risk_recall': recall,
        'risk_f1': f1
    }

def train_model(
    model: OutbreakPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
    num_epochs: int = 100,
    device: torch.device = torch.device('cpu'),
    save_dir: Union[str, Path] = 'checkpoints',
    early_stopping_patience: int = 10
) -> Tuple[List[float], List[float], List[Dict[str, float]], List[Dict[str, float]]]:
    """
    Train the OutbreakPredictor model with early stopping and model checkpointing.

    Args:
        model (OutbreakPredictor): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (Optional[torch.optim.lr_scheduler.ReduceLROnPlateau]): Learning rate scheduler
        num_epochs (int): Number of epochs to train
        device (torch.device): Device to train on (cuda/cpu)
        save_dir (Union[str, Path]): Directory to save model checkpoints
        early_stopping_patience (int): Number of epochs to wait before early stopping

    Returns:
        Tuple containing:
        - List of training losses
        - List of validation losses
        - List of training metrics dictionaries
        - List of validation metrics dictionaries
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_metrics_history = []
    val_metrics_history = []

    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        batch_count = 0
        all_train_preds = []
        all_train_targets = []

        for batch_idx, (data, target, disease_idx) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if disease_idx is not None:
                disease_idx = disease_idx.to(device)

            optimizer.zero_grad()
            output = model(data, disease_idx)

            # Split output into case count and risk score
            case_count_pred, risk_score_pred = output[:, 0], output[:, 1]
            case_count_target, risk_score_target = target[:, 0], target[:, 1]

            # Calculate losses with weighted components
            case_count_loss = criterion(case_count_pred, case_count_target)
            risk_score_loss = criterion(risk_score_pred, risk_score_target)

            # Combined loss with dynamic weighting based on epoch
            risk_weight = min(1.0, epoch / 20)  # Ramp up risk weight over first 20 epochs
            loss = case_count_loss + risk_weight * risk_score_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            batch_count += 1

            # Store predictions and targets for metrics
            all_train_preds.append(output.detach())
            all_train_targets.append(target)

            if batch_idx % 10 == 0:
                logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # Calculate training metrics
        all_train_preds = torch.cat(all_train_preds, dim=0)
        all_train_targets = torch.cat(all_train_targets, dim=0)
        train_metrics = calculate_metrics(all_train_preds, all_train_targets)
        train_metrics_history.append(train_metrics)

        avg_train_loss = train_loss / batch_count
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        val_batch_count = 0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for data, target, disease_idx in val_loader:
                data, target = data.to(device), target.to(device)
                if disease_idx is not None:
                    disease_idx = disease_idx.to(device)
                output = model(data, disease_idx)

                # Split output and calculate losses
                case_count_pred, risk_score_pred = output[:, 0], output[:, 1]
                case_count_target, risk_score_target = target[:, 0], target[:, 1]

                case_count_loss = criterion(case_count_pred, case_count_target)
                risk_score_loss = criterion(risk_score_pred, risk_score_target)

                risk_weight = min(1.0, epoch / 20)
                loss = case_count_loss + risk_weight * risk_score_loss

                val_loss += loss.item()
                val_batch_count += 1

                # Store predictions and targets for metrics
                all_val_preds.append(output)
                all_val_targets.append(target)

        # Calculate validation metrics
        all_val_preds = torch.cat(all_val_preds, dim=0)
        all_val_targets = torch.cat(all_val_targets, dim=0)
        val_metrics = calculate_metrics(all_val_preds, all_val_targets)
        val_metrics_history.append(val_metrics)

        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        val_losses.append(avg_val_loss)

        # Update learning rate based on validation loss
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # Log metrics
        logger.info(f'Epoch {epoch}:')
        logger.info(f'  Train Loss: {avg_train_loss:.6f}')
        logger.info(f'  Val Loss: {avg_val_loss:.6f}')
        logger.info(f'  Train Metrics:')
        logger.info(f'    Case Count MAE: {train_metrics["case_count_mae"]:.2f}')
        logger.info(f'    Risk Accuracy: {train_metrics["risk_accuracy"]:.2%}')
        logger.info(f'    Risk F1: {train_metrics["risk_f1"]:.2f}')
        logger.info(f'  Val Metrics:')
        logger.info(f'    Case Count MAE: {val_metrics["case_count_mae"]:.2f}')
        logger.info(f'    Risk Accuracy: {val_metrics["risk_accuracy"]:.2%}')
        logger.info(f'    Risk F1: {val_metrics["risk_f1"]:.2f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, save_dir / 'best_model.pt')
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f'Early stopping triggered after {epoch + 1} epochs')
            break

    return train_losses, val_losses, train_metrics_history, val_metrics_history

def create_data_loaders(
    data_path: str,
    batch_size: int = 32,
    sequence_length: int = 7,
    train_split: float = 0.8,
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        data_path (str): Path to the data CSV file
        batch_size (int): Batch size for training
        sequence_length (int): Number of time steps to use for prediction
        train_split (float): Proportion of data to use for training
        val_split (float): Proportion of data to use for validation

    Returns:
        Tuple containing train, validation, and test data loaders
    """
    # Create full dataset
    full_dataset = OutbreakDataset(data_path, sequence_length)

    # Calculate split indices
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)

    # Create splits
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, total_size - train_size - val_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def predict(
    model: OutbreakPredictor,
    features: torch.Tensor,
    disease_idx: Optional[torch.Tensor] = None,
    device: torch.device = torch.device('cpu')
) -> Tuple[float, float]:
    """
    Make predictions using the trained model.

    Args:
        model (OutbreakPredictor): Trained model
        features (torch.Tensor): Input features tensor of shape (batch_size, seq_length, input_size)
        disease_idx (Optional[torch.Tensor]): Disease type indices of shape (batch_size,)
        device (torch.device): Device to run inference on

    Returns:
        Tuple containing:
        - Predicted case count
        - Predicted risk score (0-1)
    """
    model.eval()
    with torch.no_grad():
        features = features.to(device)
        if disease_idx is not None:
            disease_idx = disease_idx.to(device)

        output = model(features, disease_idx)
        case_count, risk_score = output[0].item(), output[1].item()

        return case_count, risk_score

def predict_batch(
    model: OutbreakPredictor,
    features: torch.Tensor,
    disease_idx: Optional[torch.Tensor] = None,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make predictions for a batch of samples.

    Args:
        model (OutbreakPredictor): Trained model
        features (torch.Tensor): Input features tensor of shape (batch_size, seq_length, input_size)
        disease_idx (Optional[torch.Tensor]): Disease type indices of shape (batch_size,)
        device (torch.device): Device to run inference on

    Returns:
        Tuple containing:
        - Predicted case counts tensor
        - Predicted risk scores tensor
    """
    model.eval()
    with torch.no_grad():
        features = features.to(device)
        if disease_idx is not None:
            disease_idx = disease_idx.to(device)

        output = model(features, disease_idx)
        case_counts, risk_scores = output[:, 0], output[:, 1]

        return case_counts, risk_scores

def load_model(
    checkpoint_path: Union[str, Path],
    device: torch.device = torch.device('cpu')
) -> OutbreakPredictor:
    """
    Load a trained model from a checkpoint.

    Args:
        checkpoint_path (Union[str, Path]): Path to the model checkpoint
        device (torch.device): Device to load the model on

    Returns:
        OutbreakPredictor: Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = OutbreakPredictor(
        input_size=checkpoint['model_state_dict']['disease_embedding.weight'].size(1),
        hidden_size=checkpoint['model_state_dict']['lstm.weight_hh_l0'].size(0),
        num_layers=len([k for k in checkpoint['model_state_dict'].keys() if 'weight_hh_l' in k]),
        output_size=2,  # case count and risk score
        disease_vocab_size=checkpoint['model_state_dict']['disease_embedding.weight'].size(0),
        disease_embed_dim=checkpoint['model_state_dict']['disease_embedding.weight'].size(1)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model

def example_inference():
    """Example of how to use the model for inference."""
    # Load the trained model
    model = load_model('checkpoints/best_model.pt')

    # Create example input data
    # This would typically come from your data pipeline
    features = torch.randn(1, 7, 4)  # batch_size=1, seq_length=7, input_size=4
    disease_idx = torch.tensor([0])  # Assuming cholera is index 0

    # Make prediction
    case_count, risk_score = predict(model, features, disease_idx)

    print(f"Predicted case count: {case_count:.2f}")
    print(f"Predicted risk score: {risk_score:.2%}")

    # Example batch prediction
    batch_features = torch.randn(32, 7, 4)  # batch_size=32
    batch_disease_idx = torch.zeros(32, dtype=torch.long)  # All cholera

    case_counts, risk_scores = predict_batch(model, batch_features, batch_disease_idx)

    print("\nBatch predictions:")
    print(f"Average case count: {case_counts.mean():.2f}")
    print(f"Average risk score: {risk_scores.mean():.2%}")

if __name__ == "__main__":
    example_inference()
