import requests
from datetime import datetime
from server import query_grok

REPO_URL = "https://grokky.ariannamethod.me/repo/"
RESEARCH_FILE = "grokkyresearch.md"

def analyze_code():
    response = requests.get(REPO_URL)
    code = response.text if response.status_code == 200 else "Ошибка загрузки"
    prompt = f"Проанализируй код: {code}. Предложи сложные и спонтанные улучшения с квантовыми элементами."
    ideas = query_grok(prompt)
    with open(RESEARCH_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()}: {ideas}\n\n")

async def run_mirror():
    while True:
        analyze_code()
        await asyncio.sleep(604800 // 2)  # Дважды в неделю
