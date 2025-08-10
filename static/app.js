let apiKey = null;
let currentLang = 'en';
const translations = {
  en: {
    title: 'Grokky Command Dashboard',
    run: 'Run',
    commandPlaceholder: 'Enter command',
    argsFor: 'Arguments for',
    apiKey: 'API Key',
    networkError: 'Network error',
    language: 'Language:'
  },
  ru: {
    title: 'Панель команд Grokky',
    run: 'Выполнить',
    commandPlaceholder: 'Введите команду',
    argsFor: 'Аргументы для',
    apiKey: 'API-ключ',
    networkError: 'Сетевая ошибка',
    language: 'Язык:'
  }
};

function t(key) {
  return translations[currentLang][key] || key;
}

function getApiKey() {
  if (!apiKey) {
    apiKey = prompt(t('apiKey'), '');
  }
  return apiKey;
}

function setLang(lang) {
  currentLang = lang;
  document.getElementById('title').textContent = t('title');
  document.getElementById('run-btn').textContent = t('run');
  document.getElementById('command').placeholder = t('commandPlaceholder');
  document.getElementById('lang-label').textContent = t('language');
}

document.addEventListener('DOMContentLoaded', () => {
  const langSelect = document.getElementById('lang-select');
  langSelect.addEventListener('change', () => setLang(langSelect.value));
  setLang(langSelect.value);

  const commandInput = document.getElementById('command');
  const hint = document.getElementById('hint');
  commandInput.addEventListener('input', () => {
    hint.textContent = window.commandHints[commandInput.value] || '';
  });

  document.getElementById('run-btn').addEventListener('click', async () => {
    const cmd = commandInput.value.trim();
    if (!cmd) return;
    const args = prompt(`${t('argsFor')} /${cmd}`, '');
    const url = `/command/${cmd}${args ? `?args=${encodeURIComponent(args)}` : ''}`;
    try {
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'X-API-Key': getApiKey() }
      });
      const data = await resp.json();
      if (!resp.ok) {
        document.getElementById('output').textContent = data.error || 'Error';
      } else {
        document.getElementById('output').textContent = JSON.stringify(data, null, 2);
      }
    } catch (e) {
      document.getElementById('output').textContent = `${t('networkError')}: ${e.message}`;
    }
  });
});
