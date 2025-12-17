# export_classifier_to_onnx.py

import os

import joblib
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from lab11_lib.logger import get_configured_logger
from lab11_lib.settings import Settings

logger = get_configured_logger(__name__)


def export_classifier_to_onnx(settings: Settings):
    logger.info(f"Loading classifier from {settings.LOCAL_CLASSIFIER_PATH}...")
    classifier = joblib.load(settings.LOCAL_CLASSIFIER_PATH)

    # define input shape: (batch_size, embedding_dim)
    initial_type = [("float_input", FloatTensorType([None, settings.CLASSIFIER_EMBEDDING_DIM]))]

    logger.info("Converting to ONNX...")
    onnx_model = convert_sklearn(classifier, initial_types=initial_type)  # TODO: complete conversion here

    logger.info(f"Saving ONNX model to {settings.LOCAL_ONNX_CLASSIFIER_PATH}...")
    # TODO: save the onnx_model to settings.LOCAL_ONNX_CLASSIFIER_PATH
    os.makedirs(os.path.dirname(settings.LOCAL_ONNX_CLASSIFIER_PATH), exist_ok=True)
    onnx.save_model(onnx_model, settings.LOCAL_ONNX_CLASSIFIER_PATH)


if __name__ == "__main__":
    settings = Settings()
    export_classifier_to_onnx(settings)
    logger.info("Export completed.")
