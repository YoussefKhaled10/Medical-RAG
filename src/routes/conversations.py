from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from src.dependencies.auth import get_current_user
from src.helpers.config import settings
from src.models import AssetModel, ConversationModel, get_db_session
from src.models.db_schemes.medical_rag import User
from src.services.ConversationService import ConversationService
from src.services.rag_factory import create_rag_service
from src.stores.llm.GenerationExceptions import GenerationProviderError


conversations_router = APIRouter(
    prefix="/api/v1/conversations",
    tags=["Conversations"],
)


class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = Field(default=None, max_length=160)


class ConversationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=160)

    @field_validator("title")
    @classmethod
    def clean_title(cls, value: str) -> str:
        clean = " ".join(value.split()).strip()
        if not clean:
            raise ValueError("title must not be empty")
        return clean


class ConversationMessageCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    client_request_id: str = Field(min_length=8, max_length=64)
    retrieval_limit: int = Field(default=5, ge=1, le=10)
    generation_provider: Literal["gemini", "groq", "manus", "glm"] | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1200, ge=64, le=8192)
    search_scope: Literal["system", "private", "combined", "selected_file"] = "system"
    asset_id: int | None = Field(default=None, gt=0)

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        clean = " ".join(value.split()).strip()
        if not clean:
            raise ValueError("question must not be empty")
        return clean


@conversations_router.get("")
async def list_conversations(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    items = await ConversationModel.list_owned(session, current_user.id, limit, offset)
    return {
        "items": [ConversationService.serialize_conversation(item) for item in items],
        "limit": limit,
        "offset": offset,
        "has_more": len(items) == limit,
    }


@conversations_router.post("", status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ConversationCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    title = " ".join((payload.title or "New conversation").split()).strip()[:160]
    item = await ConversationModel.create(session, current_user.id, title)
    if payload.title:
        item.title_source = "manual"
    await session.commit()
    await session.refresh(item)
    return ConversationService.serialize_conversation(item)


@conversations_router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: int,
    message_limit: int = Query(default=30, ge=1, le=100),
    message_offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    item = await ConversationService.require_owned(session, conversation_id, current_user.id)
    messages = await ConversationModel.list_messages(
        session, conversation_id, current_user.id, message_limit, message_offset
    )
    result = ConversationService.serialize_conversation(item)
    result["messages"] = [ConversationService.serialize_message(message) for message in messages]
    result["message_limit"] = message_limit
    result["message_offset"] = message_offset
    return result


@conversations_router.get("/{conversation_id}/messages")
async def get_messages(
    conversation_id: int,
    limit: int = Query(default=30, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    await ConversationService.require_owned(session, conversation_id, current_user.id)
    messages = await ConversationModel.list_messages(
        session, conversation_id, current_user.id, limit, offset
    )
    return {
        "items": [ConversationService.serialize_message(item) for item in messages],
        "limit": limit,
        "offset": offset,
        "has_more": len(messages) == limit,
    }


@conversations_router.patch("/{conversation_id}")
async def rename_conversation(
    conversation_id: int,
    payload: ConversationUpdate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    item = await ConversationService.require_owned(session, conversation_id, current_user.id)
    item.title = payload.title
    item.title_source = "manual"
    await session.commit()
    await session.refresh(item)
    return ConversationService.serialize_conversation(item)


@conversations_router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> None:
    await ConversationService.require_owned(session, conversation_id, current_user.id)
    await ConversationModel.delete_owned(session, conversation_id, current_user.id)
    await session.commit()


@conversations_router.post("/{conversation_id}/messages")
async def send_message(
    conversation_id: int,
    payload: ConversationMessageCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    conversation = await ConversationService.require_owned(
        session, conversation_id, current_user.id
    )
    existing = await ConversationModel.find_by_request_id(
        session, current_user.id, payload.client_request_id
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This message request was already accepted.",
        )

    # Persist the user's message first, then release the transaction before RAG.
    user_message = await ConversationModel.add_message(
        session,
        conversation_id=conversation.id,
        user_id=current_user.id,
        role="user",
        content=payload.question,
        status="accepted",
        request_id=payload.client_request_id,
    )
    await ConversationModel.refresh_summary(session, conversation, payload.question)
    await session.commit()

    # Only history from this owned conversation is trusted. Client history is ignored.
    conversation_history = await ConversationModel.recent_history(
        session, conversation.id, current_user.id, limit=8
    )
    if conversation_history and conversation_history[-1].get("role") == "user":
        conversation_history = conversation_history[:-1]

    global_id = int(settings.GLOBAL_PROJECT_ID)
    private_id = current_user.private_project_id
    effective_asset_id: int | None = None

    if payload.search_scope == "system":
        if payload.asset_id is not None:
            raise HTTPException(status_code=422, detail="asset_id is only valid with selected_file scope.")
        effective_projects = [global_id]
    elif payload.search_scope == "private":
        if not private_id:
            raise HTTPException(status_code=400, detail="No private project is assigned to this user.")
        if payload.asset_id is not None:
            raise HTTPException(status_code=422, detail="asset_id is only valid with selected_file scope.")
        effective_projects = [private_id]
    elif payload.search_scope == "combined":
        if not private_id:
            raise HTTPException(status_code=400, detail="No private project is assigned to this user.")
        if payload.asset_id is not None:
            raise HTTPException(status_code=422, detail="asset_id is only valid with selected_file scope.")
        effective_projects = [global_id] if private_id == global_id else [global_id, private_id]
    else:
        if not private_id:
            raise HTTPException(status_code=400, detail="No private project is assigned to this user.")
        if payload.asset_id is None:
            raise HTTPException(status_code=422, detail="Select a file before searching.")
        asset = await AssetModel.get_by_id(session=session, asset_id=payload.asset_id)
        if asset is None or asset.project_id != private_id:
            raise HTTPException(status_code=404, detail="Selected file was not found.")
        effective_projects = [private_id]
        effective_asset_id = asset.id

    try:
        rag_service = create_rag_service(payload.generation_provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        output = await rag_service.ask(
            session=session,
            question=payload.question,
            project_id=effective_projects,
            asset_id=effective_asset_id,
            retrieval_limit=payload.retrieval_limit,
            temperature=payload.temperature,
            max_output_tokens=payload.max_output_tokens,
            conversation_history=conversation_history,
        )
        await ConversationModel.update_message_status(
            session,
            user_message,
            status="completed",
        )
        await session.commit()
    except GenerationProviderError as exc:
        await session.rollback()
        owned_user_message = await ConversationModel.find_by_request_id(
            session,
            current_user.id,
            payload.client_request_id,
        )
        if owned_user_message is not None:
            await ConversationModel.update_message_status(
                session,
                owned_user_message,
                status="failed",
                metadata={
                    "error_type": type(exc).__name__,
                    "provider": exc.provider,
                    "retryable": exc.retryable,
                    "retry_after_seconds": exc.retry_after_seconds,
                },
            )
            await session.commit()
        detail: dict[str, Any] = {
            "message": str(exc),
            "provider": exc.provider,
            "retryable": exc.retryable,
        }
        if exc.retry_after_seconds is not None:
            detail["retry_after_seconds"] = exc.retry_after_seconds
        raise HTTPException(status_code=exc.status_code, detail=detail) from exc
    except Exception as exc:
        await session.rollback()
        owned_user_message = await ConversationModel.find_by_request_id(
            session,
            current_user.id,
            payload.client_request_id,
        )
        if owned_user_message is not None:
            await ConversationModel.update_message_status(
                session,
                owned_user_message,
                status="failed",
                metadata={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                },
            )
            await session.commit()
        raise
    finally:
        await rag_service.close()
    metadata = {
        "answer_language": output.get("answer_language"),
        "grounded": output.get("grounded", False),
        "refused": output.get("refused", False),
        "safety_flagged": output.get("safety_flagged", False),
        "provider": output.get("provider"),
        "model": output.get("model"),
        "rag_request_id": output.get("request_id"),
        "relevance": output.get("relevance", {}),
        "evidence_strength": output.get("evidence_strength", {}),
        "citation_evaluation": output.get("citation_evaluation", {}),
        "claim_validation": output.get("claim_validation", {}),
    }
    assistant = await ConversationModel.add_message(
        session,
        conversation_id=conversation.id,
        user_id=current_user.id,
        role="assistant",
        content=str(output.get("answer") or ""),
        status="refused" if output.get("refused") else "completed",
        sources=list(output.get("sources") or []),
        evidence=list(output.get("evidence") or []),
        metadata=metadata,
    )
    conversation = await ConversationService.require_owned(
        session, conversation.id, current_user.id
    )
    await ConversationModel.refresh_summary(session, conversation, assistant.content)
    await session.commit()
    await session.refresh(assistant)
    await session.refresh(conversation)

    return {
        "conversation": ConversationService.serialize_conversation(conversation),
        "message": ConversationService.serialize_message(assistant),
        "rag": output,
    }
