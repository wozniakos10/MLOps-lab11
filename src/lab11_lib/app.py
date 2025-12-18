from typing import Dict

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from mangum import Mangum
from tokenizers import Tokenizer

from lab11_lib.api.models import PredictRequest, PredictResponse
from lab11_lib.logger import get_configured_logger
from lab11_lib.settings import CLASSIFIER_OUTPUT_MAPPER, Settings

logger = get_configured_logger(__name__)

settings = Settings()
app = FastAPI()
# loading models and tokenizer
logger.info(f"Loading tokenizer from {settings.LOCAL_TOKENIZER_JSON_FILE_PATH}...")
tokenizer = Tokenizer.from_file(settings.LOCAL_TOKENIZER_JSON_FILE_PATH)
logger.info(f"Loading ONNX sentence transformer model from {settings.LOCAL_ONNX_SENTENCE_TRANSFORMER_PATH}...")
ort_session = ort.InferenceSession(settings.LOCAL_ONNX_SENTENCE_TRANSFORMER_PATH)
logger.info(f"Loading ONNX classifier model from {settings.LOCAL_ONNX_CLASSIFIER_PATH}...")
ort_classifier = ort.InferenceSession(settings.LOCAL_ONNX_CLASSIFIER_PATH)


# Different approaches for loading models in FastAPI. Right now following Simple Approach
##https://apxml.com/courses/fastapi-ml-deployment/chapter-3-integrating-ml-models/loading-models-fastapi

# Implementing that function directly here, so I would be able to use `deprecated_app.py` for testing previous configuration
# without any dependencies error. That app file do not required any heavy libraries like torch or transformers


def predict_sentiment_onnx(
    ort_session: ort.InferenceSession, ort_classifier: ort.InferenceSession, tokenizer: Tokenizer, sentence: Dict[str, str]
) -> str:
    # tokenize input
    encoded = tokenizer.encode(sentence["text"])

    # prepare numpy arrays for ONNX
    input_ids = np.array([encoded.ids])
    attention_mask = np.array([encoded.attention_mask])

    # run embedding inference
    embedding_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    embeddings = ort_session.run(None, embedding_inputs)[0]

    # run classifier inference
    classifier_input_name = ort_classifier.get_inputs()[0].name
    classifier_inputs = {classifier_input_name: embeddings.astype(np.float32)}
    prediction = ort_classifier.run(None, classifier_inputs)[0]

    label = CLASSIFIER_OUTPUT_MAPPER.get(prediction[0], "unknown")  # return this label as response
    logger.info(f"Prediction: {label}")
    logger.info("Sucessfuly run inference with onnx based models!")
    return label


@app.get("/")
def welcome_root():
    return {"message": "Welcome to the ML API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if tokenizer is None or ort_session is None or ort_classifier is None:
        raise HTTPException(status_code=503, detail="Model is not available")

    prediction = predict_sentiment_onnx(ort_session, ort_classifier, tokenizer, request.model_dump())
    return PredictResponse(prediction=prediction)


handler = Mangum(app)


if __name__ == "__main__":
    # it's for local/integration testing
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
