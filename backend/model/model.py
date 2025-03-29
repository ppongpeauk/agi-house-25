import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Dict
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OutbreakDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        sequence_length: int = 7,
    ):
        """
        Initialize the dataset.

        Args:
            data_path (str): Path to the CSV data file
            sequence_length (int): Number of time steps to use for prediction
        """
        self.sequence_length = sequence_length
        self.data = pd.read_csv(data_path)

        # Normalize numerical features
        self.feature_columns = ['temperature', 'humidity', 'precipitation', 'wind_speed']
        self.feature_means = self.data[self.feature_columns].mean()
        self.feature_stds = self.data[self.feature_columns].std()
        self.data[self.feature_columns] = (self.data[self.feature_columns] - self.feature_means) / self.feature_stds

        # Get disease columns
        self.disease_columns = [col for col in self.data.columns if col.startswith('disease_')]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            Tuple containing:
            - Input features tensor
            - Target tensor (disease presence for each disease)
        """
        # Get sequence of features
        features = self.data[self.feature_columns].iloc[idx:idx + self.sequence_length].values

        # Get target values (disease presence)
        target = self.data[self.disease_columns].iloc[idx + self.sequence_length].values

        return torch.FloatTensor(features), torch.FloatTensor(target)

class OutbreakPredictor(nn.Module):
    """
    LSTM-based model for predicting disease outbreaks.
    Predicts presence/absence of multiple diseases.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_diseases: int,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_diseases = num_diseases

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_diseases),
            nn.Sigmoid()  # Output probabilities for each disease
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            torch.Tensor: Predictions of shape (batch_size, num_diseases)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Use only the last output for prediction
        last_hidden = lstm_out[:, -1, :]

        # Get predictions
        output = self.fc(last_hidden)

        return output

def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate various metrics for model evaluation.

    Args:
        predictions (torch.Tensor): Model predictions [num_diseases]
        targets (torch.Tensor): Ground truth values [num_diseases]

    Returns:
        Dict[str, float]: Dictionary containing various metrics
    """
    # Convert predictions to binary using 0.5 threshold
    pred_binary = (predictions > 0.5).float()

    # Calculate metrics for each disease
    metrics = {}
    for i in range(predictions.size(1)):
        disease_pred = pred_binary[:, i]
        disease_target = targets[:, i]

        # Calculate accuracy, precision, recall, and F1
        correct = (disease_pred == disease_target).sum().item()
        total = len(disease_pred)
        accuracy = correct / total

        # Calculate precision and recall
        true_positives = ((disease_pred == 1) & (disease_target == 1)).sum().item()
        predicted_positives = (disease_pred == 1).sum().item()
        actual_positives = (disease_target == 1).sum().item()

        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / actual_positives if actual_positives > 0 else 0

        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics[f'disease_{i}_accuracy'] = accuracy
        metrics[f'disease_{i}_precision'] = precision
        metrics[f'disease_{i}_recall'] = recall
        metrics[f'disease_{i}_f1'] = f1

    # Calculate average metrics across all diseases
    metrics['avg_accuracy'] = np.mean([metrics[f'disease_{i}_accuracy'] for i in range(predictions.size(1))])
    metrics['avg_precision'] = np.mean([metrics[f'disease_{i}_precision'] for i in range(predictions.size(1))])
    metrics['avg_recall'] = np.mean([metrics[f'disease_{i}_recall'] for i in range(predictions.size(1))])
    metrics['avg_f1'] = np.mean([metrics[f'disease_{i}_f1'] for i in range(predictions.size(1))])

    return metrics

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

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            # Calculate losses
            loss = criterion(output, target)

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
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                # Calculate losses
                loss = criterion(output, target)

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
        logger.info(f'    Disease Accuracy: {train_metrics["avg_accuracy"]:.2%}')
        logger.info(f'    Disease F1: {train_metrics["avg_f1"]:.2f}')
        logger.info(f'  Val Metrics:')
        logger.info(f'    Disease Accuracy: {val_metrics["avg_accuracy"]:.2%}')
        logger.info(f'    Disease F1: {val_metrics["avg_f1"]:.2f}')

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
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Make predictions using the trained model.

    Args:
        model (OutbreakPredictor): Trained model
        features (torch.Tensor): Input features tensor of shape (batch_size, seq_length, input_size)
        device (torch.device): Device to run inference on

    Returns:
        torch.Tensor: Predictions of shape (batch_size, num_diseases)
    """
    model.eval()
    with torch.no_grad():
        features = features.to(device)
        output = model(features)
        return output

def predict_batch(
    model: OutbreakPredictor,
    features: torch.Tensor,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Make predictions for a batch of samples.

    Args:
        model (OutbreakPredictor): Trained model
        features (torch.Tensor): Input features tensor of shape (batch_size, seq_length, input_size)
        device (torch.device): Device to run inference on

    Returns:
        torch.Tensor: Predictions of shape (batch_size, num_diseases)
    """
    model.eval()
    with torch.no_grad():
        features = features.to(device)
        output = model(features)
        return output

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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract hidden size from the checkpoint
    hidden_size = checkpoint['model_state_dict']['lstm.weight_hh_l0'].size(1)  # Get the hidden size from the second dimension
    num_layers = len([k for k in checkpoint['model_state_dict'].keys() if 'weight_hh_l' in k])

    model = OutbreakPredictor(
        input_size=4,  # temperature, humidity, precipitation, wind_speed
        hidden_size=hidden_size,  # Use the hidden size from the checkpoint
        num_layers=num_layers,
        num_diseases=10,  # disease_0 through disease_9
        dropout=0.2
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model

def example_inference():
    """Example of how to use the model for inference."""
    # Load the trained model
    model = load_model('checkpoints/best_model.pt')

    # Create example input data
    features = torch.randn(1, 7, 4)  # batch_size=1, seq_length=7, input_size=4

    # Make prediction
    predictions = predict(model, features)

    print("Predicted disease probabilities:")
    for i in range(10):
        print(f"Disease {i}: {predictions[0, i]:.2%}")

    # Example batch prediction
    batch_features = torch.randn(32, 7, 4)  # batch_size=32
    batch_predictions = predict_batch(model, batch_features)

    print("\nBatch predictions:")
    print(f"Average probability for each disease:")
    for i in range(10):
        print(f"Disease {i}: {batch_predictions[:, i].mean():.2%}")

if __name__ == "__main__":
    example_inference()
