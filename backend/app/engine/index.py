import logging
import os

from app.engine.context import create_service_context
from llama_index import (
    StorageContext,
    load_index_from_storage,
)


def get_index(directory):
    service_context = create_service_context()
    # check if storage already exists
    if not os.path.exists(directory):
        raise Exception(
            f"Storage directory {directory} is empty - call 'python app/engine/generate.py' to generate the storage first"
        )
    logger = logging.getLogger("uvicorn")
    # load the existing index
    logger.info(f"Loading index from {directory}...")
    storage_context = StorageContext.from_defaults(persist_dir=directory)
    index = load_index_from_storage(storage_context, service_context=service_context)
    logger.info(f"Finished loading index from {directory}")
    return index
