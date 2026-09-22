from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.db_schemes.medical_rag import ChatMessage, Conversation


class ConversationModel:
    @staticmethod
    async def create(session: AsyncSession, user_id: int, title: str = "New conversation") -> Conversation:
        item = Conversation(user_id=user_id, title=title, title_source="default")
        session.add(item)
        await session.flush()
        return item

    @staticmethod
    async def get_owned(
        session: AsyncSession,
        conversation_id: int,
        user_id: int,
    ) -> Conversation | None:
        result = await session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_owned(
        session: AsyncSession,
        user_id: int,
        limit: int = 20,
        offset: int = 0,
    ) -> Sequence[Conversation]:
        result = await session.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    @staticmethod
    async def add_message(
        session: AsyncSession,
        *,
        conversation_id: int,
        user_id: int,
        role: str,
        content: str,
        status: str = "completed",
        request_id: str | None = None,
        sources: list[dict[str, Any]] | None = None,
        evidence: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            conversation_id=conversation_id,
            user_id=user_id,
            role=role,
            content=content,
            status=status,
            request_id=request_id,
            sources_json=sources or [],
            evidence_json=evidence or [],
            metadata_json=metadata or {},
        )
        session.add(message)
        await session.flush()
        return message

    @staticmethod
    async def update_message_status(
        session: AsyncSession,
        message: ChatMessage,
        *,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        message.status = status
        if metadata is not None:
            merged = dict(message.metadata_json or {})
            merged.update(metadata)
            message.metadata_json = merged
        await session.flush()
        return message

    @staticmethod
    async def list_messages(
        session: AsyncSession,
        conversation_id: int,
        user_id: int,
        limit: int = 30,
        offset: int = 0,
    ) -> Sequence[ChatMessage]:
        result = await session.execute(
            select(ChatMessage)
            .where(
                ChatMessage.conversation_id == conversation_id,
                ChatMessage.user_id == user_id,
            )
            .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return tuple(reversed(result.scalars().all()))

    @staticmethod
    async def recent_history(
        session: AsyncSession,
        conversation_id: int,
        user_id: int,
        limit: int = 8,
    ) -> list[dict[str, str]]:
        messages = await ConversationModel.list_messages(
            session, conversation_id, user_id, limit=limit, offset=0
        )
        return [
            {"role": item.role, "content": item.content}
            for item in messages
            if item.role in {"user", "assistant"}
            and item.status in {"completed", "refused"}
        ]

    @staticmethod
    async def find_by_request_id(
        session: AsyncSession,
        user_id: int,
        request_id: str,
    ) -> ChatMessage | None:
        result = await session.execute(
            select(ChatMessage).where(
                ChatMessage.user_id == user_id,
                ChatMessage.request_id == request_id,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def refresh_summary(
        session: AsyncSession,
        conversation: Conversation,
        latest_content: str,
    ) -> None:
        count = await session.scalar(
            select(func.count(ChatMessage.id)).where(
                ChatMessage.conversation_id == conversation.id
            )
        )
        conversation.message_count = int(count or 0)
        conversation.last_message_preview = " ".join(latest_content.split())[:280]
        if conversation.title_source == "default":
            first_user = await session.scalar(
                select(ChatMessage.content)
                .where(
                    ChatMessage.conversation_id == conversation.id,
                    ChatMessage.role == "user",
                )
                .order_by(ChatMessage.id.asc())
                .limit(1)
            )
            if first_user:
                clean = " ".join(first_user.split()).strip()
                conversation.title = clean[:60] + ("..." if len(clean) > 60 else "")
                conversation.title_source = "automatic"
        conversation.updated_at = func.now()
        await session.flush()

    @staticmethod
    async def delete_owned(session: AsyncSession, conversation_id: int, user_id: int) -> bool:
        result = await session.execute(
            delete(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
        return bool(result.rowcount)
