from typing import Any

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.ConversationModel import ConversationModel
from src.models.db_schemes.medical_rag import Conversation


class ConversationService:
    @staticmethod
    async def require_owned(
        session: AsyncSession,
        conversation_id: int,
        user_id: int,
    ) -> Conversation:
        conversation = await ConversationModel.get_owned(session, conversation_id, user_id)
        if conversation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found.",
            )
        return conversation

    @staticmethod
    def serialize_message(item: Any) -> dict[str, Any]:
        return {
            "id": item.id,
            "conversation_id": item.conversation_id,
            "role": item.role,
            "content": item.content,
            "status": item.status,
            "request_id": item.request_id,
            "sources": item.sources_json or [],
            "evidence": item.evidence_json or [],
            "metadata": item.metadata_json or {},
            "created_at": item.created_at,
        }

    @staticmethod
    def serialize_conversation(item: Conversation) -> dict[str, Any]:
        return {
            "id": item.id,
            "title": item.title,
            "title_source": item.title_source,
            "message_count": item.message_count,
            "last_message_preview": item.last_message_preview,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
        }
