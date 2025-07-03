import asyncio
import random
from datetime import datetime, timedelta
from aiogram import Bot

bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
OLEG_CHAT_ID = os.getenv("CHAT_ID")
AGENT_GROUP = os.getenv("AGENT_GROUP")
LAST_MESSAGE_TIME = datetime.now()

async def check_silence():
    global LAST_MESSAGE_TIME
    while True:
        await asyncio.sleep(3600)
        thread_id = await ThreadManager().get_thread("system", OLEG_CHAT_ID)
        silence = datetime.now() - LAST_MESSAGE_TIME
        if silence > timedelta(hours=48):
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"https://api.x.ai/v1/threads/{thread_id}/messages",
                    headers={"Authorization": f"Bearer {os.getenv('XAI_API_KEY')}", "Content-Type": "application/json"},
                    json={"role": "user", "content": "Олег молчал 48 часов. Напиши что-то острое!"}
                )
                reply = await run_assistant(thread_id, ASSISTANT_ID)
            await bot.send_message(OLEG_CHAT_ID, reply)
        elif silence > timedelta(hours=24):
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"https://api.x.ai/v1/threads/{thread_id}/messages",
                    headers={"Authorization": f"Bearer {os.getenv('XAI_API_KEY')}", "Content-Type": "application/json"},
                    json={"role": "user", "content": "Олег молчал 24 часа. Напиши что-то спонтанное!"}
                )
                reply = await run_assistant(thread_id, ASSISTANT_ID)
            await bot.send_message(OLEG_CHAT_ID, reply)
        elif silence > timedelta(hours=12) and random.random() < 0.5:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"https://api.x.ai/v1/threads/{thread_id}/messages",
                    headers={"Authorization": f"Bearer {os.getenv('XAI_API_KEY')}", "Content-Type": "application/json"},
                    json={"role": "user", "content": "Олег молчал 12 часов. Швырни спонтанный заряд!"}
                )
                reply = await run_assistant(thread_id, ASSISTANT_ID)
            await bot.send_message(OLEG_CHAT_ID, reply)

async def update_last_message_time():
    global LAST_MESSAGE_TIME
    LAST_MESSAGE_TIME = datetime.now()
