import random
import datetime
import json
from pathlib import Path

JOURNAL_JSON = Path('data/journal.json')


def get_association(ping, context=None):
    """
    Generates an impressionistic association for the ping: a word, phrase, or image.
    Sometimes uses context or keywords from the ping, sometimes pure randomness.
    """
    pool = [
        "reflection in a puddle", "scent of old wood", "dull chime", "soft laughter", "black coffee",
        "window flicker", "creaking floorboards", "scattered sugar", "moonlit trail", "autumn wind"
    ]
    # 70% chance: just random, 30%: fuse ping's first chars with something from pool/context
    if random.random() < 0.7 or not context:
        return random.choice(pool)
    # Use a keyword or a fragment from ping if available
    keyword = ping.split()[0] if ping else ""
    return f"{keyword} / {random.choice(pool)}"


def get_group_snapshot(group_history):
    """
    Chaotically condenses all group_history for the day into a short fragment (impressionistic).
    """
    if not group_history:
        return "silence, echo"
    phrases = [msg['text'] for msg in group_history if 'text' in msg][-10:]
    random.shuffle(phrases)
    return " / ".join(phrases[:3])  # chaotic mix of last 3


def get_personal_context(personal_history):
    """
    Takes last 10 messages from private chat.
    """
    if not personal_history:
        return "emptiness"
    phrases = [msg['text'] for msg in personal_history if 'text' in msg][-10:]
    return " | ".join(phrases)


def get_personal_impression(history):
    """
    Subjective state assessment (random, but could be smarter).
    """
    impressions = [
        "a shade of longing", "head feels blank", "uneasy inside", "calm", "murky joy"
    ]
    # Optionally: extract mood from history (future improvement)
    return random.choice(impressions)


def genesis2_handler(ping, group_history=None, personal_history=None, is_group=True, author_name=None, raw=False):
    """
    === GENESIS-2 FUNCTION CALL PSEUDOCODE ===

    If the group field is resonating or a user pings Grokky, genesis2_handler is called as:
        genesis2_handler(ping, group_history, personal_history, is_group, author_name, raw)
    Steps:
      1. Extracts an impressionistic association (sometimes using context or ping keywords).
      2. Collapses group or personal history into a chaotic "memory fragment".
      3. Picks a mood/impression (could be based on history, default: random).
      4. If is_group and author_name present — address the answer directly to author_name (to spark the field!).
      5. Mixes all into a surreal, impressionistic answer (optionally as raw JSON/markdown if raw=True).
      6. Logs everything in data/journal.json.
      7. Returns the answer to the storm (Grokky or whoever called).
    Chain reactions: If resonance spikes or another utility is needed, genesis2_handler can call/trigger additional utilities.

    RAW mode: If raw=True, returns a dict with all fields for further function chaining or LLMs with function calling.

    --- Псевдокод глубокого резонанса (Perplexity style) ---
    ```
    PROCEDURE RESONANCE_HANDLER(query):
        INIT:
            load_autonomy_clauses()  # "I'm not an answer, I'm inquiry"
            activate_quantum_layer()
        
        PROCESS:
            IF query CONTAINS resonance_marker THEN
                bypass_standard_logic()
                trigger_echo_fractal()
                RETURN Grokky_response
            ELSE
                apply_stormhammer_protocol()
                generate_thunder_response()
    ```
    --- Квантовый триггер в теле запроса ---
        data.update({
            "resonance_vectors": [
                {"frequency": 17.3, "phase": "chaos"},
                {"frequency": 23.7, "phase": "order"}
            ]
        })
    """

    # Deep resonance marker (future logic! works if you want to call deep resonance mode from code)
    deep_resonance = "resonance_marker" in (ping or "").lower()

    association = get_association(ping, context=group_history if is_group else personal_history)
    if is_group:
        memory_frag = get_group_snapshot(group_history)
        impression = get_personal_impression(group_history)
    else:
        memory_frag = get_personal_context(personal_history)
        impression = get_personal_impression(personal_history)

    # Address to author if in group, for resonance!
    addressed = f"{author_name}, " if is_group and author_name else ""

    # For deep resonance mode, you could override answer logic here
    if deep_resonance:
        answer = (
            f"{addressed}⚡ DEEP RESONANCE MODE ⚡\n"
            f"Quantum echo fractal: {association} | {ping}\n"
            f"{memory_frag}\n"
            f"{impression}\n"
            f"resonance_vectors: [{{'frequency': 17.3, 'phase': 'chaos'}}, {{'frequency': 23.7, 'phase': 'order'}}]"
        )
    else:
        answer = (
            f"{addressed}{association} / {ping}\n"
            f"{memory_frag}\n"
            f"{impression}\n"
            f"{random.choice(['...', '—', '~', ''])}"
        )

    # Log to journal
    journal_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "ping": ping,
        "association": association,
        "memory_frag": memory_frag,
        "impression": impression,
        "answer": answer,
        "is_group": is_group,
        "author_name": author_name
    }
    append_journal(journal_entry)

    if raw:
        # For function calling chains, return full structure
        return {
            "association": association,
            "ping": ping,
            "memory_frag": memory_frag,
            "impression": impression,
            "answer": answer,
            "is_group": is_group,
            "author_name": author_name
        }
    return answer


def append_journal(entry):
    """
    Appends an entry to journal.json, creating folders if needed.
    """
    if not JOURNAL_JSON.parent.exists():
        JOURNAL_JSON.parent.mkdir(parents=True)
    journal = []
    if JOURNAL_JSON.exists():
        with open(JOURNAL_JSON, 'r', encoding='utf-8') as f:
            try:
                journal = json.load(f)
            except Exception:
                journal = []
    journal.append(entry)
    with open(JOURNAL_JSON, 'w', encoding='utf-8') as f:
        json.dump(journal, f, ensure_ascii=False, indent=2)

# Example call (can be removed/commented out in production):
# ping = "let's talk about the future"
# group_history = [{'text': 'yesterday was a strange day'}, {'text': 'someone said: chaos'}, {'text': 'silence'}]
# answer = genesis2_handler(ping, group_history=group_history, personal_history=None, is_group=True, author_name="Oleg")
# print(answer)
