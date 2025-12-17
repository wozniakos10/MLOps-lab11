from lab11_lib.aws_utils import download_file_from_s3, download_folder_from_s3
from lab11_lib.settings import Settings

if __name__ == "__main__":
    settings = Settings()

    # Download classifier
    download_file_from_s3(
        bucket=settings.S3_BUCKET,
        object_name=settings.S3_CLASSIFIER_OBJECT_KEY,
        file_name=settings.LOCAL_CLASSIFIER_PATH,
    )

    # Download sentence transformer model files
    download_folder_from_s3(
        bucket=settings.S3_BUCKET,
        prefix=settings.S3_SENTENCE_TRANSFORMER_KEY,
        local_dir=settings.LOCAL_SENTENCE_TRANSFORMER_PATH,
    )
