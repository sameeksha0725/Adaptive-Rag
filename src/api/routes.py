"""
API routes for RAG operations.
"""

from typing import Dict
from uuid import uuid4

from fastapi import APIRouter, Body, File, Header, UploadFile
from langchain_core.messages import AIMessage, HumanMessage

from src.memory.chat_history_mongo import ChatHistory
from src.models.query_request import QueryRequest
from src.rag.document_upload import documents
from src.rag.graph_builder import builder

router = APIRouter()

USER_STORE: Dict[str, dict] = {}


@router.post("/init")
async def init_api():
    """Return a lightweight API token for the Streamlit UI."""
    return {"api_token": "adaptive-rag-demo-token"}


@router.post("/create_user")
async def create_user(payload: dict = Body(...), api_token: str = Header(default=None, alias="X-API-TOKEN")):
    """Create a simple in-memory user for local/demo use."""
    username = payload.get("username")
    password = payload.get("password")

    if not username or not password:
        return {"status": "error", "message": "Username and password are required"}

    USER_STORE[username] = {"username": username, "password": password}
    return {"status": "ok", "message": "User created"}


@router.post("/login")
async def login_user(payload: dict = Body(...), api_token: str = Header(default=None, alias="X-API-TOKEN")):
    """Authenticate a user and return a simple JWT-style token."""
    username = payload.get("username")
    password = payload.get("password")

    user = USER_STORE.get(username)
    if not user or user.get("password") != password:
        return {"status": "error", "message": "Invalid username or password"}

    jwt_token = f"jwt-{username}-{uuid4().hex}"
    return {"jwt": jwt_token}


@router.post("/rag/query")
async def rag_query(req: QueryRequest):
    """
    Process a RAG query and return the result.

    Args:
        req: The query request containing query text and session_id.

    Returns:
        The generated response from the RAG pipeline.
    """
    #chat_history=ChatInMemoryHistory.get_session_history(req.token)
    chat_history = ChatHistory.get_session_history(req.session_id)
    await chat_history.add_message(HumanMessage(content=req.query))

    # Fetch full history
    messages = await chat_history.get_messages()
    result = builder.invoke({
        "messages": messages
    })
    output_text = result["messages"][-1].content
    confidence = result.get("confidence")
    sources = result.get("sources") or []

    # Save assistant message (store only content in chat history)
    await chat_history.add_message(AIMessage(content=output_text))

    # Return structured result while preserving the top-level `result` key
    return {"result": {"content": output_text, "confidence": confidence, "sources": sources, "retrieval_analytics": result.get("retrieval_analytics")}}


@router.post("/rag/documents/upload")
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description")
):
    """
    Upload a document for RAG processing.

    Args:
        file: The file to upload (PDF or TXT).
        description: Document description provided via header.

    Returns:
        Upload status.
    """
    status_upload = documents(description, file)
    return {"status": status_upload}

