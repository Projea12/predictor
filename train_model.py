import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from text_dataset import TextDataset
from lstm_model import NextWordLSTM, LSTMTrainer
import time
import os


def train_language_model(
    text_corpus: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    sequence_length: int = 10,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3,
    train_split: float = 0.8,
    device: str = 'auto',
    save_model: bool = True,
    model_path: str = 'next_word_lstm.pth'
):
    """
    Complete training loop for the LSTM language model.
    
    Args:
        text_corpus (str): Input text for training
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for Adam optimizer
        sequence_length (int): Length of input sequences
        embedding_dim (int): Embedding dimension
        hidden_dim (int): LSTM hidden dimension
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate
        train_split (float): Fraction of data for training (rest for validation)
        device (str): Device to use ('auto', 'cpu', 'cuda', 'mps')
        save_model (bool): Whether to save the trained model
        model_path (str): Path to save the model
    
    Returns:
        tuple: (trained_model, dataset, training_history)
    """
    
    # Set device
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA GPU")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    else:
        device = torch.device(device)
        print(f"Using device: {device}")
    
    print("=" * 60)
    print("PREPARING DATA")
    print("=" * 60)
    
    # Create dataset
    dataset = TextDataset(
        text_corpus=text_corpus,
        sequence_length=sequence_length,
        min_word_freq=1,
        vocab_size=None  # Use all words
    )
    
    # Split dataset into training and validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Vocabulary size: {dataset.get_vocab_size()}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False
    )
    
    print("=" * 60)
    print("CREATING MODEL")
    print("=" * 60)
    
    # Create model
    model = NextWordLSTM(
        vocab_size=dataset.get_vocab_size(),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=False
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 10
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(batch_inputs)
            
            # Compute loss
            loss = criterion(logits, batch_targets)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # Update weights
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                
                # Forward pass
                logits, _ = model(batch_inputs)
                
                # Compute loss
                loss = criterion(logits, batch_targets)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rate'].append(current_lr)
        
        # Calculate time
        epoch_time = time.time() - epoch_start_time
        
        # Print progress
        print(f"Epoch [{epoch+1:3d}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            if save_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'vocab_size': dataset.get_vocab_size(),
                    'sequence_length': sequence_length,
                    'embedding_dim': embedding_dim,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'dropout': dropout,
                }, model_path)
                print(f"✓ Best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print()  # Empty line for readability
    
    total_time = time.time() - start_time
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Load best model
    if save_model and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Best model loaded from {model_path}")
    
    return model, dataset, history


def plot_training_history(history):
    """Plot training and validation loss curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Learning rate plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['learning_rate'], 'g-')
    plt.title('Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


def test_text_generation(model, dataset, seed_texts, device):
    """Test text generation with the trained model."""
    print("=" * 60)
    print("TEXT GENERATION EXAMPLES")
    print("=" * 60)
    
    model.eval()
    
    for seed in seed_texts:
        print(f"Seed: '{seed}'")
        
        # Generate with different temperatures
        for temp in [0.7, 1.0, 1.5]:
            generated = model.generate_text(
                dataset=dataset,
                seed_text=seed,
                max_length=20,
                temperature=temp,
                device=device
            )
            print(f"  T={temp}: {generated}")
        print()


# Main training script
if __name__ == "__main__":
    # Sample text corpus (you can replace with your own text file)
    sample_text = """
    The quick brown fox jumps over the lazy dog. The dog was sleeping under the tree.
    The fox was very clever and quick. It jumped over the dog again and again.
    The lazy dog finally woke up and chased the fox. The fox ran into the forest.
    The dog returned to sleep under the tree. The sun was shining bright in the sky.
    
    Machine learning is a fascinating field of artificial intelligence. Neural networks 
    can learn complex patterns from data. Deep learning has revolutionized many areas
    of computer science including natural language processing and computer vision.
    
    Python is a powerful programming language widely used in data science and machine learning.
    PyTorch is a popular deep learning framework that makes it easy to build and train neural networks.
    The LSTM architecture is particularly effective for sequential data like text and time series.
    
    Natural language processing involves teaching computers to understand and generate human language.
    Word embeddings capture semantic relationships between words in high-dimensional vector spaces.
    Recurrent neural networks can process sequences of variable length making them ideal for text.
    
    Training deep learning models requires careful tuning of hyperparameters like learning rate.
    Overfitting can be prevented using techniques like dropout and early stopping.
    Cross-validation helps assess model performance and generalization capability.
    """
    
    print("LSTM Language Model Training Script")
    print("=" * 60)
    
    # Training configuration
    config = {
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.001,
        'sequence_length': 8,
        'embedding_dim': 64,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'train_split': 0.8
    }
    
    print("Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Train the model
    model, dataset, history = train_language_model(
        text_corpus=sample_text,
        **config
    )
    
    # Plot training history (optional - requires matplotlib)
    try:
        plot_training_history(history)
        print("✓ Training plots saved as 'training_history.png'")
    except ImportError:
        print("⚠ Matplotlib not available - skipping plots")
    except Exception as e:
        print(f"⚠ Could not create plots: {e}")
    
    # Test text generation
    seed_texts = [
        "the quick brown",
        "machine learning is",
        "python is a",
        "neural networks can"
    ]
    
    device = next(model.parameters()).device
    test_text_generation(model, dataset, seed_texts, device)
    
    print("=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("The trained model has been saved as 'next_word_lstm.pth'")
    print("You can load it later using:")
    print("  checkpoint = torch.load('next_word_lstm.pth')")
    print("  model.load_state_dict(checkpoint['model_state_dict'])")
