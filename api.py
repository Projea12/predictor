from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import os
import logging
from predict import load_predictor_with_corpus
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Next Word Prediction API",
    description="LSTM-based next word prediction service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

predictor = None

class PredictionRequest(BaseModel):
    text: str
    top_k: Optional[int] = 3

class WordPrediction(BaseModel):
    word: str
    probability: float

class PredictionResponse(BaseModel):
    input_text: str
    predictions: List[WordPrediction]
    success: bool
    message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vocab_size: Optional[int] = None

# Training corpus will be loaded when needed
TRAINING_CORPUS = None

def load_model():
    """Load the trained model on startup."""
    global predictor, TRAINING_CORPUS
    
    model_path = 'next_word_lstm.pth'
    
    if not os.path.exists(model_path):
        logger.error(f"Model file '{model_path}' not found.")
        logger.info("Please run 'python3 train_model.py' first to train a model.")
        return False
    
    try:
        # Use the original simple corpus that was used for training (compatible vocabulary)
        logger.info("Using original training corpus for vocabulary compatibility...")
        TRAINING_CORPUS = """The quick brown fox jumps over the lazy dog. The dog was sleeping under the tree.
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
Cross-validation helps assess model performance and generalization capability."""
        
        logger.info(f"✓ Training corpus loaded: {len(TRAINING_CORPUS):,} characters")
        
        # Load model with compatible corpus
        logger.info("Loading trained model...")
        predictor = load_predictor_with_corpus(
            text_corpus=TRAINING_CORPUS,
            model_path=model_path,
            sequence_length=8
        )
        logger.info("✓ Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        traceback.print_exc()
        return False

@app.on_event("startup")
async def startup_event():
    """Load model when the API starts."""
    success = load_model()
    if not success:
        logger.warning("API started without a loaded model. /predict endpoint will not work.")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Next Word Prediction API",
        "description": "LSTM-based next word prediction service",
        "endpoints": {
            "/predict": "POST - Predict next words for input text",
            "/health": "GET - Check API and model status",
            "/docs": "GET - Interactive API documentation"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global predictor
    
    model_loaded = predictor is not None
    vocab_size = None
    
    if model_loaded:
        try:
            vocab_size = predictor.dataset.get_vocab_size()
        except:
            vocab_size = None
    
    return HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        vocab_size=vocab_size
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_next_words(request: PredictionRequest):
    """
    Predict the top-k most likely next words for the input text.
    
    Args:
        request: PredictionRequest containing text and optional top_k parameter
    
    Returns:
        PredictionResponse with predictions and metadata
    """
    global predictor
    
    # Check if model is loaded
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the trained model file exists and restart the API."
        )
    
    # Validate input
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty."
        )
    
    if request.top_k < 1 or request.top_k > 10:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 10."
        )
    
    try:
        # Make prediction
        predictions = predictor.predict_next_words(
            input_text=request.text,
            top_k=request.top_k,
            return_probabilities=True
        )
        
        # Convert to response format
        word_predictions = [
            WordPrediction(word=word, probability=prob)
            for word, prob in predictions
        ]
        
        return PredictionResponse(
            input_text=request.text,
            predictions=word_predictions,
            success=True,
            message=f"Successfully predicted top {len(word_predictions)} next words."
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/single", response_model=Dict[str, str])
async def predict_single_word(request: Dict[str, str]):
    """
    Predict only the most likely next word.
    
    Args:
        request: Dictionary with 'text' key
    
    Returns:
        Dictionary with input text and predicted word
    """
    global predictor
    
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )
    
    text = request.get('text', '').strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty."
        )
    
    try:
        next_word = predictor.predict_next_word_only(text)
        
        return {
            "input_text": text,
            "predicted_word": next_word,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Single prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/generate", response_model=Dict[str, str])
async def generate_text(request: Dict[str, Any]):
    """
    Generate a continuation of the input text.
    
    Args:
        request: Dictionary with 'text', optional 'max_words', and 'temperature'
    
    Returns:
        Dictionary with input and generated text
    """
    global predictor
    
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )
    
    text = request.get('text', '').strip()
    max_words = request.get('max_words', 15)
    temperature = request.get('temperature', 1.0)
    
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty."
        )
    
    if max_words < 1 or max_words > 50:
        raise HTTPException(
            status_code=400,
            detail="max_words must be between 1 and 50."
        )
    
    if temperature < 0.1 or temperature > 2.0:
        raise HTTPException(
            status_code=400,
            detail="temperature must be between 0.1 and 2.0."
        )
    
    try:
        generated = predictor.generate_continuation(
            input_text=text,
            max_words=max_words,
            temperature=temperature
        )
        
        return {
            "input_text": text,
            "generated_text": generated,
            "max_words": max_words,
            "temperature": temperature,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Text generation failed: {str(e)}"
        )

# Exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    traceback.print_exc()
    
    return HTTPException(
        status_code=500,
        detail="An unexpected error occurred."
    )

if __name__ == "__main__":
    # Run the API server
    print("Starting Next Word Prediction API...")
    print("API Documentation will be available at: http://localhost:8000/docs")
    print("Health check available at: http://localhost:8000/health")
    print("Prediction endpoint: POST http://localhost:8000/predict")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
