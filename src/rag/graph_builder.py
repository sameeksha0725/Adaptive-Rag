"""
Graph builder module for the adaptive RAG system.
"""

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import START, END
from langgraph.graph.state import StateGraph

from src.rag.reAct_agent import agent_executor
from src.rag.retriever_setup import get_retriever, search_faiss_documents, search_bm25_documents
from src.rag.reranker import CrossEncoderReranker
from src.config.settings import Config
from src.llms.openai import llm
from src.models.grade import Grade
from src.models.route_identifier import RouteIdentifier
from src.models.state import State
from src.tools.graph_tools import routing_tool, doc_tool

config = Config()
reranker = CrossEncoderReranker(config)


# Node implementations
def query_classifier(state: State):
    """
    Classify the query to determine if it's related to indexed documents.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated state with route and latest_query.
    """
    question = state["messages"][-1].content
    retriever = get_retriever()
    context = retriever.invoke(question)
    print("docs received from Qdrant")
    print(context)

    llm_with_structured_output = llm.with_structured_output(RouteIdentifier)
    classify_prompt = PromptTemplate(
        template=config.prompt("classify_prompt"),
        input_variables=["question", "context"]
    )
    chain = classify_prompt | llm_with_structured_output
    result = chain.invoke({"question": question, "context": context})
    print("result received is in query classifier")
    print(result.route)

    return {"messages": state["messages"], "route": result.route, "latest_query": question}


def general_llm(state: State):
    """
    Fetch general common knowledge result from the LLM.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated messages from LLM.
    """
    result = llm.invoke(state["messages"])
    print("inside general llm")
    print(result)
    return {"messages": result}


def retriever_node(state: State):
    """
    Retrieve results from vector stores using the reAct agent.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated messages with tool calls.
    """
    messages = state["latest_query"]
    result = agent_executor.invoke({"input": messages})

    # Extract tool calls
    intermediate_steps = result.get("intermediate_steps", [])
    tool_calls = []
    if intermediate_steps:
        for action, tool_result in intermediate_steps:
            tool_calls.append({
                "tool": action.tool,
                "input": action.tool_input,
            })

    new_message = AIMessage(
        content=result["output"],
        additional_kwargs={"tool_calls": tool_calls},
    )

    return {
        "messages": [new_message]
    }


def vector_search(state: State):
    """Perform semantic vector search (FAISS) and attach results to state.

    Returns a list of documents under `vector_results` (ordered).
    """
    question = state.get("latest_query")
    docs = search_faiss_documents(question, k=config.reranker_pool_size())

    results = []
    for d in docs:
        text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        results.append({"text": text, "meta": getattr(d, "metadata", {})})

    return {"vector_results": results}


def keyword_search(state: State):
    """Perform keyword BM25 search and attach results to state as `keyword_results`."""
    question = state.get("latest_query")
    docs = search_bm25_documents(question, k=config.reranker_pool_size())

    results = []
    for d in docs:
        text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        results.append({"text": text, "meta": getattr(d, "metadata", {})})

    return {"keyword_results": results}


def hybrid_fusion(state: State):
    """Fuse vector and keyword search results using Reciprocal Rank Fusion (RRF).

    Produces a list of fused candidate documents under `hybrid_candidates`.
    """
    vector_results = state.get("vector_results") or []
    keyword_results = state.get("keyword_results") or []

    # If one side is missing, try to populate it using the search helpers
    if not vector_results and state.get("latest_query"):
        vector_docs = search_faiss_documents(state.get("latest_query"), k=config.reranker_pool_size())
        for d in vector_docs:
            text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
            vector_results.append({"text": text, "meta": getattr(d, "metadata", {})})

    if not keyword_results and state.get("latest_query"):
        keyword_docs = search_bm25_documents(state.get("latest_query"), k=config.reranker_pool_size())
        for d in keyword_docs:
            text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
            keyword_results.append({"text": text, "meta": getattr(d, "metadata", {})})

    k_rrf = 60
    scores = {}

    def score_list(lst):
        for rank, item in enumerate(lst, start=1):
            key = item["text"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_rrf + rank)

    score_list(vector_results)
    score_list(keyword_results)

    ranked_texts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_k = config.reranker_pool_size()
    hybrid_candidates = []

    # Build a mapping from text -> meta from the original result lists
    meta_by_text = {}
    for item in vector_results + keyword_results:
        text = item["text"]
        if text not in meta_by_text:
            meta_by_text[text] = dict(item.get("meta", {}) or {})

    for text, _score in ranked_texts[:top_k]:
        hybrid_candidates.append({"text": text, "meta": meta_by_text.get(text, {})})

    return {"hybrid_candidates": hybrid_candidates}


def reranker_node(state: State):
    """Rerank combined hybrid candidates with a cross-encoder and emit fused context."""
    query = state.get("latest_query")
    candidates = state.get("hybrid_candidates") or []
    reranked = reranker.rerank_documents(query, candidates)

    # Build fused context text and sources list (deduplicated)
    top_texts = [candidate["text"] for candidate in reranked]
    fused_context = "\n\n".join(top_texts)

    sources = []
    seen = set()
    for cand in reranked:
        meta = cand.get("meta", {}) or {}
        filename = meta.get("filename") or meta.get("source")
        page = meta.get("page")
        chunk_id = meta.get("chunk_id")
        if filename:
            if page is not None:
                label = f"{filename} (Page {page})"
            elif chunk_id is not None:
                label = f"{filename} (Chunk {chunk_id})"
            else:
                label = f"{filename}"

            if label not in seen:
                seen.add(label)
                sources.append(label)

    return {"messages": [AIMessage(content=fused_context)], "sources": sources}


def grade(state: State):
    """
    Grade the results retrieved from vector stores.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated state with binary_score.
    """
    grading_prompt = PromptTemplate(
        template=config.prompt("grading_prompt"),
        input_variables=["question", "context"]
    )
    context = state["messages"][-1].content
    question = state["latest_query"]

    llm_with_grade = llm.with_structured_output(Grade)

    chain_graded = grading_prompt | llm_with_grade
    result = chain_graded.invoke({"question": question, "context": context})

    print(result)
    return {"messages": state["messages"], "binary_score": result.binary_score}


def rewrite_query(state: State):
    """
    Rewrite the query to get better retrieval results.

    Args:
        state (State): State of the question.

    Returns:
        dict: Updated latest_query and retry_count.
    """
    query = state["latest_query"]
    rewrite_prompt = PromptTemplate(
        template=config.prompt("rewrite_prompt"),
        input_variables=["query"]
    )
    chain = rewrite_prompt | llm
    result = chain.invoke({"query": query})
    print("Rewritten query:", result.content)

    retry_count = state.get("retry_count") or 0
    retry_count += 1

    return {
        "latest_query": result.content,
        "retry_count": retry_count
    }


def generate(state: State):
    """
    Generate the final answer for the user.

    Args:
        state (State): State of the question.

    Returns:
        dict: Generated response.
    """
    context = state["messages"][-1].content

    generate_prompt = PromptTemplate(
        template=config.prompt("generate_prompt"),
        input_variables=["context"]
    )

    generate_chain = generate_prompt | llm
    result = generate_chain.invoke({"context": context})

    # Append Sources section if provided in state
    sources = state.get("sources") or []
    if sources:
        sources_text = "\n\nSources:\n" + "\n".join(f"• {s}" for s in sources)
        final_content = f"{result.content}{sources_text}"
    else:
        final_content = result.content

    return {"messages": [{"role": "assistant", "content": final_content}]}


def web_search(state: State):
    """
    Search the web for the rewritten query.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Search results as messages.
    """
    # Initialize the Tavily tool
    search_tool = TavilySearchResults()

    # Search a query
    result = search_tool.invoke(state["latest_query"])

    contents = [item["content"] for item in result if "content" in item]
    print(contents)

    return {
        "messages": [{"role": "assistant", "content": "\n\n".join(contents)}]
    }


# Build the graph
graph = StateGraph(State)

graph.add_node("query_analysis", query_classifier)
graph.add_node("retriever", retriever_node)
graph.add_node("vector_search", vector_search)
graph.add_node("keyword_search", keyword_search)
graph.add_node("hybrid_fusion", hybrid_fusion)
graph.add_node("reranker", reranker_node)
graph.add_node("grade", grade)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite_query)
graph.add_node("web_search", web_search)
graph.add_node("general_llm", general_llm)

graph.add_edge(START, "query_analysis")
graph.add_edge("web_search", "generate")
# Branch to both vector and keyword searches from the retriever node
graph.add_edge("retriever", "vector_search")
graph.add_edge("retriever", "keyword_search")

# Fuse the results, rerank, then continue to grading
graph.add_edge("vector_search", "hybrid_fusion")
graph.add_edge("keyword_search", "hybrid_fusion")
graph.add_edge("hybrid_fusion", "reranker")
graph.add_edge("reranker", "grade")
graph.add_edge("rewrite", "retriever")
graph.add_conditional_edges("query_analysis", routing_tool)
graph.add_conditional_edges("grade", doc_tool)
graph.add_edge("generate", END)
graph.add_edge("general_llm", END)

builder = graph.compile()

