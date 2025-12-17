from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # AWS_REGION: str
    # ECR_REPOSITORY: str
    S3_BUCKET: str = "mlops-lab11-dw"
    S3_CLASSIFIER_OBJECT_KEY: str = "models/models_lab_11/classifier.joblib"
    CLASSIFIER_EMBEDDING_DIM: int = 384
    S3_SENTENCE_TRANSFORMER_KEY: str = "models/models_lab_11/sentence_transformer.model/"
    LOCAL_CLASSIFIER_PATH: str = "models/model/classifier.joblib"
    LOCAL_SENTENCE_TRANSFORMER_PATH: str = "models/model/sentence_transformer.model"
    LOCAL_ONNX_CLASSIFIER_PATH: str = "models/model_onnx/classifier/classifier.onnx"
    LOCAL_ONNX_SENTENCE_TRANSFORMER_PATH: str = "models/model_onnx/sentence_transformer/sentence_transformer.onnx"
    LOCAL_TOKENIZER_PATH: str = "models/tokenizer/"
    LOCAL_TOKENIZER_JSON_FILE_PATH: str = LOCAL_TOKENIZER_PATH + "tokenizer.json"


CLASSIFIER_OUTPUT_MAPPER = {0: "negative", 1: "neutral", 2: "positive"}
