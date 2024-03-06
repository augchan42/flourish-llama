from app.engine.index import get_index
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.react.base import ReActAgent
from llama_index.llms.openai import OpenAI
from app.engine.constants import STORAGE_DIR, SUMMARY_STORAGE_DIR


from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.selectors.pydantic_selectors import PydanticSingleSelector
# from llama_index.tools.query_engine import QueryEngineTool

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
                name="FlourishChunks",
                description=(
                    "Ask questions about the book Flourish, broken up into 1024 byte chunks. "
                    "Supports Dense Retrieval and Semantic search"
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=chat_summary,
            metadata=ToolMetadata(
                name="FlourishSummarized",
                description=(
                    "Contains summary information about the book Flourish"
                    "Good at answering broad questions about the book that requires synthesis across multiple chapters"
                ),
            ),
        ),
    ]

    return query_engine_tools

def get_list_tool():
    list_query_engine = get_index(SUMMARY_STORAGE_DIR).as_query_engine(
        response_mode="tree_summarize",
        verbose=True,
        use_async=True,
    )
    list_tool = QueryEngineTool.from_defaults(
        query_engine=list_query_engine,
        description=(
            "Useful for summarization questions related to the book Flourish by Harjeet Virdee."
        ),
    )
    return list_tool

def get_vector_tool():
    vector_query_engine = get_index(STORAGE_DIR).as_query_engine(similarity_top_k=3)
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for retrieving specific context related to the book Flourish by Harjeet Virdee."
        ),
    )
    return vector_tool

def get_agent():
    llm = OpenAI(model="gpt-3.5-turbo-0613", chat_mode="react")

    agent = ReActAgent.from_tools(
        get_tools(),
        llm=llm,
        verbose=True,
        # context=context
    )

    return agent

def get_router():
    query_engine = RouterQueryEngine(
        selector=PydanticSingleSelector.from_defaults(),
        query_engine_tools=[
            get_list_tool,
            get_vector_tool,
        ],    
    )

    return query_engine