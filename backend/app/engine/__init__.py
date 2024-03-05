from app.engine.index import get_index
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.react.base import ReActAgent
from llama_index.llms.openai import OpenAI
from app.engine.constants import STORAGE_DIR, SUMMARY_STORAGE_DIR

# def get_chat_engine():
#     return get_index(STORAGE_DIR).as_chat_engine(
#         similarity_top_k=3, chat_mode="condense_plus_context"
#     )

def get_tools():
    chat_chunk = get_index(STORAGE_DIR).as_query_engine(similarity_top_k=3)
    chat_summary = get_index(SUMMARY_STORAGE_DIR).as_query_engine(
        response_mode="tree_summarize",
        verbose=True)
    query_engine_tools = [
        QueryEngineTool(
            query_engine=chat_chunk,
            metadata=ToolMetadata(
                name="Flourish broken into Chunks",
                description=(
                    "Ask questions about the book Flourish, broken up into 1024 byte chunks. "
                    "Supports Dense Retrieval and Semantic search"
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=chat_summary,
            metadata=ToolMetadata(
                name="Flourish Summarized",
                description=(
                    "Contains summary information about the book Flourish"
                    "Good at answering broad questions about the book that requires synthesis across multiple chapters"
                ),
            ),
        ),
    ]

    return query_engine_tools

def get_agent():
    llm = OpenAI(model="gpt-3.5-turbo-0613", chat_mode="condense_plus_context")

    agent = ReActAgent.from_tools(
        get_tools(),
        llm=llm,
        verbose=True,
        # context=context
    )

    return agent