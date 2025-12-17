from typing import Dict

import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV

from lab11_lib.logger import get_configured_logger
from lab11_lib.settings import CLASSIFIER_OUTPUT_MAPPER

logger = get_configured_logger()


def load_joblib_model(path: str = "large_models/model/classifier.joblib") -> LogisticRegressionCV | None:
    """Load a joblib model from the specified path."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found at path: {path}")
        return None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def load_sentence_transformer_model(
    path: str = "large_models/model/sentence_transformer.model", device: str = "cpu"
) -> SentenceTransformer | None:
    """Load a SentenceTransformer model from the specified path."""
    # can be done smarter, to do not repeat code
    try:
        model = SentenceTransformer(path, device=device)
        return model
    except FileNotFoundError:
        logger.error(f"SentenceTransformer model file not found at path: {path}")
        return None
    except Exception as e:
        logger.error(f"Error loading SentenceTransformer model: {e}")
        return None


def predict_sentiment(sentence_transformer: SentenceTransformer, classifier: LogisticRegressionCV, sentence: Dict[str, str]) -> str:
    """Predict the sentiment of a given sentence."""
    embeddings = sentence_transformer.encode([sentence["text"]])

    prediction = classifier.predict(embeddings)
    logger.info(f"Prediction: {prediction}")
    return CLASSIFIER_OUTPUT_MAPPER[prediction[0]]
