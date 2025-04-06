from chest_classifier import logger
from chest_classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> Stage: {STAGE_NAME} completed <<<<<<< \n\nx===============x")
except Exception as e:
    logger.exception(e)
    raise e
