import torch
from torch.utils.data import Dataset
import re
from collections import Counter
from typing import List, Tuple, Dict, Optional


class TextDataset(Dataset):
    """
    A PyTorch Dataset class for next-word prediction from text corpus.
    
    This dataset tokenizes text into word-level sequences and prepares 
    input-output pairs where the model learns to predict the next word
    given a sequence of previous words.
    """
    
    def __init__(
        self, 
        text_corpus: str, 
        sequence_length: int = 10,
        min_word_freq: int = 2,
        vocab_size: Optional[int] = None
    ):
        """
        Initialize the TextDataset.
        
        Args:
            text_corpus (str): The input text corpus as a string
            sequence_length (int): Length of input sequences for prediction
            min_word_freq (int): Minimum frequency for words to be included in vocabulary
            vocab_size (int, optional): Maximum vocabulary size. If None, uses all words above min_freq
        """
        self.sequence_length = sequence_length
        self.min_word_freq = min_word_freq
        self.vocab_size = vocab_size
        
        # Tokenize and build vocabulary
        self.tokens = self._tokenize(text_corpus)
        self.vocab = self._build_vocabulary(self.tokens)
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        # Convert tokens to indices
        self.token_indices = self._tokens_to_indices(self.tokens)
        
        # Create input-output pairs
        self.sequences = self._create_sequences()
        
        print(f"Dataset created with {len(self.sequences)} sequences")
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Total tokens: {len(self.tokens)}")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into word-level tokens.
        
        Args:
            text (str): Input text corpus
            
        Returns:
            List[str]: List of tokenized words
        """
        # Convert to lowercase and split by whitespace and punctuation
        text = text.lower()
        # Keep letters, numbers, and basic punctuation as separate tokens
        tokens = re.findall(r'\w+|[.!?;,]', text)
        return tokens
    
    def _build_vocabulary(self, tokens: List[str]) -> List[str]:
        """
        Build vocabulary from tokens based on frequency.
        
        Args:
            tokens (List[str]): List of tokenized words
            
        Returns:
            List[str]: Vocabulary list
        """
        # Count word frequencies
        word_freq = Counter(tokens)
        
        # Filter by minimum frequency
        filtered_words = [word for word, freq in word_freq.items() if freq >= self.min_word_freq]
        
        # Sort by frequency (most common first)
        filtered_words.sort(key=lambda x: word_freq[x], reverse=True)
        
        # Limit vocabulary size if specified
        if self.vocab_size and len(filtered_words) > self.vocab_size - 1:  # -1 for <UNK>
            filtered_words = filtered_words[:self.vocab_size - 1]
        
        # Add unknown token
        vocab = ['<UNK>'] + filtered_words
        
        return vocab
    
    def _tokens_to_indices(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to vocabulary indices.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[int]: List of vocabulary indices
        """
        indices = []
        unk_idx = self.word2idx['<UNK>']
        
        for token in tokens:
            indices.append(self.word2idx.get(token, unk_idx))
        
        return indices
    
    def _create_sequences(self) -> List[Tuple[List[int], int]]:
        """
        Create input-output sequence pairs for next-word prediction.
        
        Returns:
            List[Tuple[List[int], int]]: List of (input_sequence, target_word) pairs
        """
        sequences = []
        
        # Create sliding window sequences
        for i in range(len(self.token_indices) - self.sequence_length):
            input_seq = self.token_indices[i:i + self.sequence_length]
            target = self.token_indices[i + self.sequence_length]
            sequences.append((input_seq, target))
        
        return sequences
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sequence pair.
        
        Args:
            idx (int): Index of the sequence
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (input_sequence, target_word) as tensors
        """
        input_seq, target = self.sequences[idx]
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)
    
    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self.vocab)
    
    def decode_sequence(self, indices: List[int]) -> str:
        """
        Decode a sequence of indices back to words.
        
        Args:
            indices (List[int]): List of vocabulary indices
            
        Returns:
            str: Decoded text
        """
        words = [self.idx2word[idx] for idx in indices]
        return ' '.join(words)
    
    def encode_text(self, text: str) -> List[int]:
        """
        Encode new text using the existing vocabulary.
        
        Args:
            text (str): Text to encode
            
        Returns:
            List[int]: List of vocabulary indices
        """
        tokens = self._tokenize(text)
        return self._tokens_to_indices(tokens)


# Example usage and testing
if __name__ == "__main__":
    # Sample text corpus
    sample_text = """
    The quick brown fox jumps over the lazy dog. The dog was sleeping under the tree.
    The fox was very clever and quick. It jumped over the dog again and again.
    The lazy dog finally woke up and chased the fox. The fox ran into the forest.
    The dog returned to sleep under the tree. The sun was shining bright.
    """
    
    # Create dataset
    dataset = TextDataset(
        text_corpus=sample_text,
        sequence_length=5,
        min_word_freq=1,
        vocab_size=50
    )
    
    # Display some information
    print(f"\nVocabulary: {dataset.vocab[:20]}...")  # First 20 words
    print(f"Dataset length: {len(dataset)}")
    
    # Show a few examples
    print("\nFirst 5 training examples:")
    for i in range(5):
        input_seq, target = dataset[i]
        input_words = dataset.decode_sequence(input_seq.tolist())
        target_word = dataset.idx2word[target.item()]
        print(f"Input: '{input_words}' -> Target: '{target_word}'")
    
    # Test with PyTorch DataLoader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    print(f"\nBatch example:")
    for batch_inputs, batch_targets in dataloader:
        print(f"Batch input shape: {batch_inputs.shape}")
        print(f"Batch target shape: {batch_targets.shape}")
        break
