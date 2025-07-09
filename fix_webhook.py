#!/usr/bin/env python3
"""
Скрипт для исправления webhook URL в Telegram Bot API
Устанавливает правильный webhook URL вместо заглушки
"""

import os
import requests
import sys
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

def fix_webhook():
    """Устанавливает правильный webhook URL"""
    
    # Получаем токен из переменной окружения
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN не установлен!")
        print("💡 Убедитесь, что в .env файле указан реальный токен бота")
        return False
    
    # Проверяем, что токен не является заглушкой
    test_tokens = [
        "your_telegram_bot_token_here", 
        "<your-telegram-bot-token>", 
        "your-telegram-bot-token",
        "test_token"
    ]
    
    if token in test_tokens:
        print("❌ TELEGRAM_BOT_TOKEN содержит заглушку! Установите реальный токен.")
        print(f"   Текущее значение: {token}")
        print("💡 Получите токен у @BotFather в Telegram")
        return False
    
    # Получаем URL приложения (Railway устанавливает разные переменные)
    app_url = (
        os.getenv("RAILWAY_STATIC_URL") or 
        os.getenv("RAILWAY_PUBLIC_DOMAIN") or
        os.getenv("WEBHOOK_URL") or
        os.getenv("PUBLIC_URL")
    )
    
    if not app_url:
        print("❌ Не найден URL приложения!")
        print("💡 Возможные переменные: RAILWAY_STATIC_URL, WEBHOOK_URL, PUBLIC_URL")
        print("💡 Для Railway обычно: https://your-app-name.railway.app")
        
        # Попробуем угадать URL на основе имени проекта
        railway_service = os.getenv("RAILWAY_SERVICE_NAME")
        if railway_service:
            suggested_url = f"https://{railway_service}.railway.app"
            print(f"💡 Возможный URL: {suggested_url}")
            
            user_input = input(f"Использовать {suggested_url}? (y/n): ").lower().strip()
            if user_input == 'y':
                app_url = suggested_url
            else:
                manual_url = input("Введите URL вашего приложения: ").strip()
                if manual_url:
                    app_url = manual_url
        
        if not app_url:
            return False
    
    # Убираем trailing slash если есть
    app_url = app_url.rstrip('/')
    
    # Формируем правильный webhook URL
    webhook_url = f"{app_url}/webhook"
    
    print(f"🔧 Устанавливаем webhook URL: {webhook_url}")
    print(f"🔑 Используем токен: {token[:10]}...")
    
    # Сначала удаляем старый webhook
    delete_url = f"https://api.telegram.org/bot{token}/deleteWebhook"
    try:
        response = requests.post(delete_url)
        if response.status_code == 200:
            print("✅ Старый webhook удален")
        else:
            print(f"⚠️ Ошибка удаления старого webhook: {response.text}")
    except Exception as e:
        print(f"⚠️ Ошибка при удалении webhook: {e}")
    
    # Устанавливаем новый webhook
    set_url = f"https://api.telegram.org/bot{token}/setWebhook"
    data = {
        "url": webhook_url,
        "drop_pending_updates": True  # Очищаем очередь старых сообщений
    }
    
    try:
        response = requests.post(set_url, json=data)
        result = response.json()
        
        if response.status_code == 200 and result.get("ok"):
            print("✅ Webhook успешно установлен!")
            print(f"📝 Описание: {result.get('description', 'N/A')}")
            return True
        else:
            print(f"❌ Ошибка установки webhook: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при установке webhook: {e}")
        return False

def check_webhook():
    """Проверяет текущий webhook"""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN не установлен!")
        return
    
    url = f"https://api.telegram.org/bot{token}/getWebhookInfo"
    
    try:
        response = requests.get(url)
        result = response.json()
        
        if response.status_code == 200 and result.get("ok"):
            webhook_info = result.get("result", {})
            current_url = webhook_info.get("url", "")
            
            print("📋 Текущая информация о webhook:")
            print(f"   URL: {current_url}")
            print(f"   Pending updates: {webhook_info.get('pending_update_count', 0)}")
            print(f"   Last error: {webhook_info.get('last_error_message', 'Нет ошибок')}")
            
            # Проверяем на заглушку
            if "<your-telegram-bot-token>" in current_url or "%3Cyour-telegram-bot-token%3E" in current_url:
                print("❌ НАЙДЕНА ЗАГЛУШКА В WEBHOOK URL!")
                return False
            elif current_url:
                print("✅ Webhook URL выглядит корректно")
                return True
            else:
                print("⚠️ Webhook не установлен")
                return False
        else:
            print(f"❌ Ошибка получения информации о webhook: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при проверке webhook: {e}")
        return False

def show_env_info():
    """Показывает информацию о переменных окружения"""
    print("📋 Информация о переменных окружения:")
    
    token = os.getenv("TELEGRAM_BOT_TOKEN", "НЕ УСТАНОВЛЕН")
    if token and token not in ["НЕ УСТАНОВЛЕН", "test_token"]:
        token_display = f"{token[:10]}...{token[-4:]}" if len(token) > 14 else "КОРОТКИЙ_ТОКЕН"
    else:
        token_display = token
    
    openai_key = os.getenv("OPENAI_API_KEY", "НЕ УСТАНОВЛЕН")
    if openai_key and openai_key != "НЕ УСТАНОВЛЕН" and openai_key != "test_key":
        openai_display = f"{openai_key[:10]}...{openai_key[-4:]}" if len(openai_key) > 14 else "КОРОТКИЙ_КЛЮЧ"
    else:
        openai_display = openai_key
    
    print(f"   TELEGRAM_BOT_TOKEN: {token_display}")
    print(f"   OPENAI_API_KEY: {openai_display}")
    print(f"   CHAT_ID: {os.getenv('CHAT_ID', 'НЕ УСТАНОВЛЕН')}")
    print(f"   RAILWAY_STATIC_URL: {os.getenv('RAILWAY_STATIC_URL', 'НЕ УСТАНОВЛЕН')}")
    print(f"   WEBHOOK_URL: {os.getenv('WEBHOOK_URL', 'НЕ УСТАНОВЛЕН')}")
    print(f"   PORT: {os.getenv('PORT', '8000')}")

if __name__ == "__main__":
    print("🤖 Grokky Webhook Fixer")
    print("=" * 50)
    
    # Показываем информацию о переменных окружения
    show_env_info()
    print()
    
    # Проверяем текущий webhook
    print("1️⃣ Проверяем текущий webhook...")
    webhook_ok = check_webhook()
    
    if not webhook_ok:
        print("\n2️⃣ Исправляем webhook...")
        if fix_webhook():
            print("\n3️⃣ Проверяем результат...")
            check_webhook()
        else:
            print("❌ Не удалось исправить webhook!")
            print("\n📖 Смотрите WEBHOOK_FIX_INSTRUCTIONS.md для ручного исправления")
            sys.exit(1)
    else:
        print("✅ Webhook уже настроен правильно!")
    
    print("\n🎉 Готово! Теперь бот должен получать сообщения.")
