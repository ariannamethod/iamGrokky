import os
import asyncio
import tempfile
from openai import OpenAI
from pydub import AudioSegment
from utils.telegram_utils import send_voice_message, download_file, get_file_url

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class VoiceHandler:
    def __init__(self):
        self.voice_enabled = {}  # Словарь для отслеживания включенного голоса по chat_id
        
    def is_voice_enabled(self, chat_id: str) -> bool:
        """Проверяет, включен ли голосовой режим для чата"""
        return self.voice_enabled.get(chat_id, False)
    
    def enable_voice(self, chat_id: str):
        """Включает голосовой режим для чата"""
        self.voice_enabled[chat_id] = True
        
    def disable_voice(self, chat_id: str):
        """Выключает голосовой режим для чата"""
        self.voice_enabled[chat_id] = False
    
    async def transcribe_voice(self, file_id: str) -> str:
        """Транскрибирует голосовое сообщение через Whisper"""
        try:
            # Получаем URL файла
            file_url = await get_file_url(file_id)
            if not file_url:
                return "Ошибка получения файла"
            
            # Скачиваем файл
            audio_data = await download_file(file_url)
            if not audio_data:
                return "Ошибка скачивания аудио"
            
            # Создаем временный файл
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Конвертируем в wav если нужно
                audio = AudioSegment.from_file(temp_file_path)
                wav_path = temp_file_path.replace(".ogg", ".wav")
                audio.export(wav_path, format="wav")
                
                # Транскрибируем через Whisper
                with open(wav_path, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="ru"  # Можно сделать автоопределение
                    )
                
                return transcript.text
                
            finally:
                # Удаляем временные файлы
                try:
                    os.unlink(temp_file_path)
                    if os.path.exists(wav_path):
                        os.unlink(wav_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"Ошибка транскрипции: {e}")
            return f"Грокки не смог разобрать голос: {e}"
    
    async def text_to_speech(self, text: str, voice: str = "alloy") -> bytes:
        """Преобразует текст в речь через OpenAI TTS"""
        try:
            # Ограничиваем длину текста
            if len(text) > 4000:
                text = text[:4000] + "..."
            
            response = client.audio.speech.create(
                model="tts-1",
                voice=voice,  # alloy, echo, fable, onyx, nova, shimmer
                input=text,
                response_format="opus"  # Для Telegram лучше opus
            )
            
            return response.content
            
        except Exception as e:
            print(f"Ошибка TTS: {e}")
            return None
    
    async def process_voice_message(self, file_id: str, chat_id: str) -> str:
        """Обрабатывает входящее голосовое сообщение"""
        if not self.is_voice_enabled(chat_id):
            return "Голосовой режим выключен. Используйте /voiceon для включения."
        
        # Транскрибируем голосовое сообщение
        transcription = await self.transcribe_voice(file_id)
        
        if transcription.startswith("Ошибка") or transcription.startswith("Грокки не смог"):
            return transcription
        
        return transcription
    
    async def send_voice_response(self, text: str, chat_id: str, voice: str = "alloy"):
        """Отправляет голосовой ответ"""
        if not self.is_voice_enabled(chat_id):
            return False
        
        # Генерируем аудио
        audio_data = await self.text_to_speech(text, voice)
        if not audio_data:
            return False
        
        # Отправляем голосовое сообщение
        return await send_voice_message(chat_id, audio_data)

# Глобальный экземпляр обработчика голоса
voice_handler = VoiceHandler()
