import logging
import asyncio
from pyrogram.client import Client
from pyrogram import filters
from pyrogram.enums.parse_mode import ParseMode
from pyrogram.types import Message
from src.core.llm_client import EventExtractor
from src.config import API_ID, API_HASH, MONITORED_CHANNEL_IDS
from src.utils.storage import append_event_to_json

# NOTE: тестируется
from src.core.my_rag_dense_bm25 import EventRetriever

model = "qwen2.5:1.5b"  # NOTE: можно пожирнее взять
event_extractor = EventExtractor(model)
retriever = EventRetriever()
logging.info(f"🤖 Использую модель {model}")

app = Client("Posts Parser", api_id=API_ID, api_hash=API_HASH)

#
# @app.on_message(filters.user(users="aabdyev") & filters.private)
# async def hello(client: Client, message: Message):
#     logging.info(f"Захвачено сообщение {message.text}")
#     await message.forward(chat_id="me")


# NOTE: Запрос -> LLM -> JSONL + CHROMADB
@app.on_message(filters.chat(MONITORED_CHANNEL_IDS))
async def read_post(client: Client, message: Message):
    event_id = f"{message.chat.id}:{message.id}"
    logging.info(f"Новый пост в канале {message.chat.title}.")

    content = message.text or message.caption
    if not content:
        return

    event = await event_extractor.process_post(content, event_id)

    if event:
        logging.info(f"Извлечено событие: {event}")
        append_event_to_json(event)
        # добавь_в_базу_chromadb()
        retriever.add_event(event)
    else:
        logging.warning(
            f"Не удалось извлечь информацию о мероприятии из поста {message.id} в канале {message.chat.id}."
        )


@app.on_message(
    filters.private & filters.text & filters.user(users=["aabdyev", "Alen4i"])
)
# NOTE: UserMsg -> SEARCH on CHROMADB -> RERANK -> AnswerMsg
async def handle_user_query(client: Client, message: Message):
    logging.info(f"[INFO] Пользователь пишет {message.text}")
    query = message.text
    await message.reply("Ищу самые интересные события для вас... ⏳")

    try:
        # избежание блокировки потока
        loop = asyncio.get_event_loop()
        event_ids = await loop.run_in_executor(None, retriever.search, query)

        if not event_ids:
            await message.reply("К сожалению, ничего подходящего не нашел :(")
            return

        for eid in event_ids:
            chat_id_str, msg_id_str = eid.split(":")
            await client.forward_messages(
                chat_id=message.chat.id,
                from_chat_id=int(chat_id_str),
                message_ids=int(msg_id_str),
            )

    except Exception as e:
        logging.error(f"Ошибка при поиске/пересылке: {e}")
        await message.reply("Упс, произошла ошибка при поиске.")


def run_bot():
    logging.info("Бот запускается...")
    app.run()
    logging.info("Бот остановлен.")
