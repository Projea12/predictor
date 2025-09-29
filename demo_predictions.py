import torch
from predict import load_predictor_with_corpus
import os


def demo_next_word_predictions():
    """
    Demonstrate the next word prediction functionality using the original corpus.
    """
    # Original training corpus (same as used in train_model.py)
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
    
    print("Next Word Prediction Demo with Original Corpus")
    print("=" * 60)
    
    # Check if model exists
    model_path = 'next_word_lstm.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found.")
        print("Please run 'python3 train_model.py' first to train a model.")
        return
    
    try:
        # Load predictor with original corpus
        print("Loading trained model with original training corpus...")
        predictor = load_predictor_with_corpus(
            text_corpus=sample_text,
            model_path=model_path,
            sequence_length=8
        )
        
        # Test sentences that should work well with our training data
        test_sentences = [
            "the quick brown",
            "machine learning is",
            "neural networks can", 
            "python is a",
            "the dog was",
            "deep learning has",
            "natural language processing",
            "the fox jumped"
        ]
        
        print("\n" + "=" * 60)
        print("TOP 3 NEXT WORD PREDICTIONS")
        print("=" * 60)
        
        for sentence in test_sentences:
            try:
                predictions = predictor.predict_next_words(sentence, top_k=3)
                print(f"\nInput: '{sentence}'")
                print("Top 3 predictions:")
                
                for i, (word, prob) in enumerate(predictions, 1):
                    print(f"  {i}. '{word:<15}' (probability: {prob:.4f})")
                    
            except Exception as e:
                print(f"Error predicting for '{sentence}': {e}")
        
        # Test single word predictions
        print("\n" + "=" * 60)
        print("SINGLE MOST LIKELY NEXT WORD")
        print("=" * 60)
        
        for sentence in test_sentences:
            try:
                next_word = predictor.predict_next_word_only(sentence)
                print(f"'{sentence}' → '{next_word}'")
            except Exception as e:
                print(f"Error: {e}")
        
        # Test text generation
        print("\n" + "=" * 60)
        print("TEXT GENERATION EXAMPLES")  
        print("=" * 60)
        
        generation_seeds = [
            "the quick brown",
            "machine learning is",
            "neural networks can",
            "python is a"
        ]
        
        for seed in generation_seeds:
            try:
                # Generate with different temperatures
                print(f"\nSeed: '{seed}'")
                
                for temp in [0.5, 1.0, 1.5]:
                    continuation = predictor.generate_continuation(
                        seed, 
                        max_words=12, 
                        temperature=temp
                    )
                    print(f"  T={temp}: {continuation}")
                    
            except Exception as e:
                print(f"Error generating for '{seed}': {e}")
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE!")
        print("=" * 60)
        print("Key Functions Available:")
        print("• predict_next_words(text, top_k=3) - Get top-k predictions with probabilities")
        print("• predict_next_word_only(text) - Get single most likely word")
        print("• generate_continuation(text, max_words, temperature) - Generate text")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_next_word_predictions()
