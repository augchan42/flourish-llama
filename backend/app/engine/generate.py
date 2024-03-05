import logging

from dotenv import load_dotenv

from app.engine.constants import STORAGE_DIR, SUMMARY_STORAGE_DIR
from app.engine.context import create_service_context
from app.engine.loader import get_documents

load_dotenv()

from llama_index import (    
    VectorStoreIndex,
    SummaryIndex
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def generate_datasource(service_context):
    logger.info("Creating VectorStoreIndex")
    # load the documents and create the index
    documents = get_documents()
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    # store it for later
    index.storage_context.persist(STORAGE_DIR)    

    logger.info(f"Finished creating new VectorStoreIndex. Stored in {STORAGE_DIR}")

    logger.info("Creating SummaryIndex")
    summaryIndex = SummaryIndex.from_documents(documents, service_context=service_context)
    summaryIndex.storage_context.persist(SUMMARY_STORAGE_DIR)

    logger.info(f"Finished creating summaryIndex. Stored in {SUMMARY_STORAGE_DIR}")

    


if __name__ == "__main__":
    service_context = create_service_context()
    generate_datasource(service_context)
