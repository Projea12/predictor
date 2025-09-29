import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Optional


class NextWordLSTM(nn.Module):
    """
    LSTM-based language model for next-word prediction.
    
    Architecture:
    - Embedding layer: Maps word indices to dense vectors
    - LSTM layer(s): Processes sequential information
    - Linear classifier: Outputs probability distribution over vocabulary
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Initialize the LSTM model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimension of word embeddings
            hidden_dim (int): Hidden dimension of LSTM
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate for regularization
            bidirectional (bool): Whether to use bidirectional LSTM
        """
        super(NextWordLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # Assuming index 0 is used for padding if needed
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Calculate output size for linear layer
        lstm_output_size = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Linear classifier (output layer)
        self.linear = nn.Linear(lstm_output_size, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embedding weights
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Initialize linear layer weights
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
            hidden (tuple, optional): Hidden state tuple (h_0, c_0)
        
        Returns:
            tuple: (output, hidden_state)
                - output: Tensor of shape (batch_size, vocab_size)
                - hidden_state: Tuple of hidden and cell states
        """
        batch_size, seq_len = x.shape
        
        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM layer
        lstm_out, hidden_state = self.lstm(embedded, hidden)
        # lstm_out: (batch_size, seq_len, hidden_dim * num_directions)
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim * num_directions)
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Linear classifier
        logits = self.linear(last_output)  # (batch_size, vocab_size)
        
        return logits, hidden_state
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state for LSTM.
        
        Args:
            batch_size (int): Batch size
            device (torch.device): Device to create tensors on
        
        Returns:
            tuple: (h_0, c_0) hidden state tensors
        """
        num_directions = 2 if self.bidirectional else 1
        
        h_0 = torch.zeros(
            self.num_layers * num_directions, 
            batch_size, 
            self.hidden_dim,
            device=device
        )
        c_0 = torch.zeros(
            self.num_layers * num_directions, 
            batch_size, 
            self.hidden_dim,
            device=device
        )
        
        return h_0, c_0
    
    def generate_text(
        self, 
        dataset, 
        seed_text: str, 
        max_length: int = 50, 
        temperature: float = 1.0,
        device: torch.device = torch.device('cpu')
    ) -> str:
        """
        Generate text using the trained model.
        
        Args:
            dataset: TextDataset instance for encoding/decoding
            seed_text (str): Starting text
            max_length (int): Maximum length of generated text
            temperature (float): Sampling temperature (higher = more random)
            device (torch.device): Device for computation
        
        Returns:
            str: Generated text
        """
        self.eval()
        
        with torch.no_grad():
            # Encode seed text
            tokens = dataset.encode_text(seed_text)
            
            # If seed is shorter than sequence length, pad or truncate
            if len(tokens) < dataset.sequence_length:
                tokens = [0] * (dataset.sequence_length - len(tokens)) + tokens
            else:
                tokens = tokens[-dataset.sequence_length:]
            
            generated_tokens = tokens.copy()
            
            for _ in range(max_length):
                # Prepare input
                input_seq = torch.tensor([tokens[-dataset.sequence_length:]], device=device)
                
                # Forward pass
                logits, _ = self.forward(input_seq)
                
                # Apply temperature
                logits = logits / temperature
                
                # Sample next token
                probabilities = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1).item()
                
                generated_tokens.append(next_token)
                tokens.append(next_token)
                
                # Stop if we generate an end token (if applicable)
                if next_token == 0:  # Assuming 0 is a stop token
                    break
            
            # Decode generated tokens
            generated_text = dataset.decode_sequence(generated_tokens)
            
        return generated_text


# Training utilities
class LSTMTrainer:
    """Utility class for training the LSTM model."""
    
    def __init__(
        self, 
        model: NextWordLSTM, 
        device: torch.device = torch.device('cpu')
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(
        self, 
        dataloader: DataLoader, 
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Train the model for one epoch.
        
        Args:
            dataloader (DataLoader): Training data loader
            optimizer (torch.optim.Optimizer): Optimizer
        
        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = self.model(batch_inputs)
            
            # Compute loss
            loss = self.criterion(logits, batch_targets)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """
        Evaluate the model.
        
        Args:
            dataloader (DataLoader): Evaluation data loader
        
        Returns:
            float: Average loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in dataloader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                logits, _ = self.model(batch_inputs)
                
                # Compute loss
                loss = self.criterion(logits, batch_targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches


# Example usage
if __name__ == "__main__":
    # Import the TextDataset
    from text_dataset import TextDataset
    
    # Sample text
    sample_text = """
    The quick brown fox jumps over the lazy dog. The dog was sleeping under the tree.
    The fox was very clever and quick. It jumped over the dog again and again.
    The lazy dog finally woke up and chased the fox. The fox ran into the forest.
    The dog returned to sleep under the tree. The sun was shining bright.
    Machine learning is a fascinating field of artificial intelligence.
    Neural networks can learn complex patterns from data.
    """
    
    # Create dataset
    dataset = TextDataset(
        text_corpus=sample_text,
        sequence_length=5,
        min_word_freq=1,
        vocab_size=100
    )
    
    # Create model
    model = NextWordLSTM(
        vocab_size=dataset.get_vocab_size(),
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Vocabulary size: {dataset.get_vocab_size()}")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Test forward pass
    for batch_inputs, batch_targets in dataloader:
        print(f"Input shape: {batch_inputs.shape}")
        print(f"Target shape: {batch_targets.shape}")
        
        # Forward pass
        logits, hidden = model(batch_inputs)
        print(f"Output logits shape: {logits.shape}")
        
        # Test loss computation
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, batch_targets)
        print(f"Loss: {loss.item():.4f}")
        
        break
    
    # Test text generation (with untrained model)
    generated = model.generate_text(
        dataset, 
        seed_text="the quick brown", 
        max_length=10,
        temperature=1.0
    )
    print(f"\nGenerated text: {generated}")
