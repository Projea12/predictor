import torch
import torch.nn.functional as F
from text_dataset import TextDataset
from lstm_model import NextWordLSTM
import os
from typing import List, Tuple, Optional


class NextWordPredictor:
    """
    A predictor class that loads a trained LSTM model and predicts next words.
    """
    
    def __init__(self, model_path: str = 'next_word_lstm.pth', device: str = 'auto'):
        """
        Initialize the predictor by loading the trained model.
        
        Args:
            model_path (str): Path to the saved model checkpoint
            device (str): Device to run predictions on ('auto', 'cpu', 'cuda', 'mps')
        """
        self.model_path = model_path
        
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Load model and dataset
        self.model, self.dataset = self._load_model_and_dataset()
        
        print(f"✓ Model loaded successfully on {self.device}")
        print(f"✓ Vocabulary size: {self.dataset.get_vocab_size()}")
        print(f"✓ Sequence length: {self.dataset.sequence_length}")
    
    def _load_model_and_dataset(self) -> Tuple[NextWordLSTM, TextDataset]:
        """
        Load the trained model and recreate the dataset.
        
        Returns:
            Tuple[NextWordLSTM, TextDataset]: Loaded model and dataset
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model parameters from checkpoint
        vocab_size = checkpoint['vocab_size']
        sequence_length = checkpoint['sequence_length']
        embedding_dim = checkpoint['embedding_dim']
        hidden_dim = checkpoint['hidden_dim']
        num_layers = checkpoint['num_layers']
        dropout = checkpoint['dropout']
        
        # Create model with same architecture
        model = NextWordLSTM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False
        ).to(self.device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # We need to recreate the dataset with the same vocabulary
        # For now, we'll create a dummy dataset and manually set the vocabulary
        # In a production system, you'd want to save the vocabulary separately
        dummy_text = "dummy text for vocabulary reconstruction"
        dataset = TextDataset(
            text_corpus=dummy_text,
            sequence_length=sequence_length,
            min_word_freq=1
        )
        
        # This is a limitation - we need the original vocabulary
        # In practice, you'd save the vocabulary mapping separately
        print("⚠️  Note: Using reconstructed vocabulary. For best results, use the same text corpus.")
        
        return model, dataset
    
    def predict_next_words(
        self, 
        input_text: str, 
        top_k: int = 3,
        return_probabilities: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Predict the top-k most likely next words for the given input text.
        
        Args:
            input_text (str): Input sentence/text
            top_k (int): Number of top predictions to return
            return_probabilities (bool): Whether to return probabilities
        
        Returns:
            List[Tuple[str, float]]: List of (word, probability) tuples
        """
        self.model.eval()
        
        with torch.no_grad():
            # Encode the input text
            try:
                tokens = self.dataset.encode_text(input_text)
            except Exception as e:
                print(f"Warning: Could not encode text properly: {e}")
                # Fallback to simple tokenization
                words = input_text.lower().split()
                tokens = []
                for word in words:
                    if word in self.dataset.word2idx:
                        tokens.append(self.dataset.word2idx[word])
                    else:
                        tokens.append(0)  # <UNK> token
            
            # Handle sequence length
            if len(tokens) < self.dataset.sequence_length:
                # Pad with zeros if too short
                tokens = [0] * (self.dataset.sequence_length - len(tokens)) + tokens
            else:
                # Take the last sequence_length tokens if too long
                tokens = tokens[-self.dataset.sequence_length:]
            
            # Convert to tensor
            input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
            
            # Forward pass
            logits, _ = self.model(input_tensor)
            
            # Convert logits to probabilities
            probabilities = F.softmax(logits, dim=-1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities[0], top_k)
            
            # Convert back to words
            predictions = []
            for i in range(top_k):
                word_idx = top_indices[i].item()
                prob = top_probs[i].item()
                
                # Get word from vocabulary
                if word_idx < len(self.dataset.idx2word):
                    word = self.dataset.idx2word[word_idx]
                else:
                    word = "<UNK>"
                
                if return_probabilities:
                    predictions.append((word, prob))
                else:
                    predictions.append((word, None))
            
            return predictions
    
    def predict_next_word_only(self, input_text: str) -> str:
        """
        Predict only the most likely next word.
        
        Args:
            input_text (str): Input sentence/text
        
        Returns:
            str: Most likely next word
        """
        predictions = self.predict_next_words(input_text, top_k=1, return_probabilities=False)
        return predictions[0][0] if predictions else "<UNK>"
    
    def generate_continuation(
        self, 
        input_text: str, 
        max_words: int = 10,
        temperature: float = 1.0
    ) -> str:
        """
        Generate a continuation of the input text.
        
        Args:
            input_text (str): Starting text
            max_words (int): Maximum number of words to generate
            temperature (float): Sampling temperature
        
        Returns:
            str: Generated continuation
        """
        return self.model.generate_text(
            dataset=self.dataset,
            seed_text=input_text,
            max_length=max_words,
            temperature=temperature,
            device=self.device
        )


def load_predictor_with_corpus(
    text_corpus: str,
    model_path: str = 'next_word_lstm.pth',
    sequence_length: int = 8,
    device: str = 'auto'
) -> NextWordPredictor:
    """
    Load a predictor with the original text corpus to ensure vocabulary consistency.
    
    Args:
        text_corpus (str): Original training text corpus
        model_path (str): Path to saved model
        sequence_length (int): Sequence length used during training
        device (str): Device for inference
    
    Returns:
        NextWordPredictor: Configured predictor
    """
    # Set device
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate dataset with original corpus
    dataset = TextDataset(
        text_corpus=text_corpus,
        sequence_length=sequence_length,
        min_word_freq=1,
        vocab_size=None
    )
    
    # Create model
    model = NextWordLSTM(
        vocab_size=dataset.get_vocab_size(),
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout'],
        bidirectional=False
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create custom predictor
    predictor = NextWordPredictor.__new__(NextWordPredictor)
    predictor.model_path = model_path
    predictor.device = device
    predictor.model = model
    predictor.dataset = dataset
    
    print(f"✓ Predictor loaded with original corpus on {device}")
    print(f"✓ Vocabulary size: {dataset.get_vocab_size()}")
    
    return predictor


# Standalone prediction functions
def predict_next_words(
    input_text: str,
    model_path: str = 'next_word_lstm.pth',
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Standalone function to predict next words.
    
    Args:
        input_text (str): Input text
        model_path (str): Path to trained model
        top_k (int): Number of predictions to return
    
    Returns:
        List[Tuple[str, float]]: List of (word, probability) tuples
    """
    predictor = NextWordPredictor(model_path)
    return predictor.predict_next_words(input_text, top_k=top_k)


# Example usage and testing
if __name__ == "__main__":
    print("Next Word Prediction Demo")
    print("=" * 50)
    
    # Check if trained model exists
    model_path = 'next_word_lstm.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found.")
        print("Please run 'python3 train_model.py' first to train a model.")
        exit(1)
    
    try:
        # Method 1: Using the predictor class
        print("Loading trained model...")
        predictor = NextWordPredictor(model_path)
        
        # Test sentences
        test_sentences = [
            "the quick brown",
            "machine learning is",
            "neural networks can",
            "python is a",
            "deep learning has",
            "artificial intelligence will",
            "data science requires"
        ]
        
        print("\n" + "=" * 50)
        print("TOP 3 NEXT WORD PREDICTIONS")
        print("=" * 50)
        
        for sentence in test_sentences:
            try:
                predictions = predictor.predict_next_words(sentence, top_k=3)
                print(f"\nInput: '{sentence}'")
                print("Top 3 predictions:")
                
                for i, (word, prob) in enumerate(predictions, 1):
                    print(f"  {i}. {word:<15} (probability: {prob:.4f})")
                    
            except Exception as e:
                print(f"Error predicting for '{sentence}': {e}")
        
        # Test single word prediction
        print("\n" + "=" * 50)
        print("SINGLE WORD PREDICTIONS")
        print("=" * 50)
        
        for sentence in test_sentences[:3]:  # Just test first 3
            try:
                next_word = predictor.predict_next_word_only(sentence)
                print(f"'{sentence}' → '{next_word}'")
            except Exception as e:
                print(f"Error: {e}")
        
        # Test text generation
        print("\n" + "=" * 50)
        print("TEXT GENERATION")
        print("=" * 50)
        
        for sentence in test_sentences[:2]:  # Just test first 2
            try:
                continuation = predictor.generate_continuation(
                    sentence, 
                    max_words=15, 
                    temperature=0.8
                )
                print(f"Seed: '{sentence}'")
                print(f"Generated: {continuation}")
                print()
            except Exception as e:
                print(f"Error generating for '{sentence}': {e}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nThis might happen because the vocabulary wasn't saved with the model.")
        print("For better results, use load_predictor_with_corpus() with the original training text.")
    
    print("=" * 50)
    print("Demo completed!")
