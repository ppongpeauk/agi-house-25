import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
from model import (
    OutbreakPredictor,
    train_model,
    calculate_metrics,
    create_data_loaders
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main training function."""
    torch.manual_seed(42)

    # Configuration
    config = {
        'data_path': 'merged_data.csv',
        'batch_size': 32,
        'sequence_length': 7,
        'input_size': 4,  # temperature, humidity, precipitation, wind_speed
        'hidden_size': 64,
        'num_layers': 2,
        'num_diseases': 10,  # disease_0 through disease_9
        'dropout': 0.2,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'early_stopping_patience': 10,
        'save_dir': 'checkpoints'
    }

    train_loader, val_loader, test_loader = create_data_loaders(
        config['data_path'],
        config['batch_size'],
        config['sequence_length']
    )

    # Initialize model
    model = OutbreakPredictor(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_diseases=config['num_diseases'],
        dropout=config['dropout']
    )

    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    # Train model
    train_losses, val_losses, train_metrics, val_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config['num_epochs'],
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        save_dir=config['save_dir'],
        early_stopping_patience=config['early_stopping_patience']
    )

    # Evaluate on test set
    model.eval()
    test_loss = 0
    all_test_preds = []
    all_test_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            all_test_preds.append(output)
            all_test_targets.append(target)

    all_test_preds = torch.cat(all_test_preds, dim=0)
    all_test_targets = torch.cat(all_test_targets, dim=0)
    test_metrics = calculate_metrics(all_test_preds, all_test_targets)

    # Log final results
    logger.info("Training completed!")
    logger.info(f"Final test loss: {test_loss / len(test_loader):.6f}")
    logger.info("Test metrics:")
    logger.info(f"  Average Accuracy: {test_metrics['avg_accuracy']:.4f}")
    logger.info(f"  Average Precision: {test_metrics['avg_precision']:.4f}")
    logger.info(f"  Average Recall: {test_metrics['avg_recall']:.4f}")
    logger.info(f"  Average F1: {test_metrics['avg_f1']:.4f}")

    # Log individual disease metrics
    for i in range(config['num_diseases']):
        logger.info(f"\nDisease {i} metrics:")
        logger.info(f"  Accuracy: {test_metrics[f'disease_{i}_accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics[f'disease_{i}_precision']:.4f}")
        logger.info(f"  Recall: {test_metrics[f'disease_{i}_recall']:.4f}")
        logger.info(f"  F1: {test_metrics[f'disease_{i}_f1']:.4f}")

if __name__ == "__main__":
    main()
