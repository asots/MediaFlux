"""在 SDK 分发给工作线程前，统一新建私聊话题首条消息的路由。"""

from __future__ import annotations

import telebot

from app import config
from app.modules.telegram_topic_routing import record_private_topic_service_message


class TopicRoutingBot(telebot.TeleBot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 仅关联创建事件紧邻的首条消息，不保存“最近活跃话题”。
        self._topic_openers: dict[tuple[int, int], tuple[int, int, int]] = {}

    def process_new_messages(self, new_messages):
        for message in new_messages:
            self._route_private_topic(message)
        super().process_new_messages(new_messages)

    def _route_private_topic(self, message):
        if (
            message.chat.type != "private"
            or str(message.chat.id) != config.get("TG_CHAT_ID", "").strip()
        ):
            return

        actor = message.from_user
        pending = self._topic_openers.pop((message.chat.id, message.message_id), None)
        if (
            pending is not None
            and message.message_thread_id is None
            and message.reply_to_message is None
            and actor is not None
            and not actor.is_bot
            and actor.id == pending[0]
            and 0 <= message.date - pending[2] <= 3
        ):
            message.message_thread_id = pending[1]
            message.is_topic_message = True

        if message.content_type not in {
            "forum_topic_created", "forum_topic_edited",
            "forum_topic_closed", "forum_topic_reopened",
        }:
            return

        from app.bot.agent_adapter import telegram_agent_owner

        # 同步落盘，避免首条文本已进入 Agent 而创建事件仍在线程队列中。
        record_private_topic_service_message(
            telegram_agent_owner(message.chat.id, message.chat.id), message,
        )
        created = message.forum_topic_created
        if (
            created is not None
            and getattr(created, "is_name_implicit", None) is True
            and message.message_thread_id is not None
            and actor is not None
            and not actor.is_bot
        ):
            self._topic_openers[(message.chat.id, message.message_id + 1)] = (
                actor.id, message.message_thread_id, message.date,
            )
            if len(self._topic_openers) > 128:
                self._topic_openers.pop(next(iter(self._topic_openers)))
