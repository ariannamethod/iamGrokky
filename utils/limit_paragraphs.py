import re

def limit_paragraphs(text, max_paragraphs=4):
    """
    Trims the text to a maximum of N paragraphs.
    A paragraph is considered a block separated by empty lines or line breaks.
    """
    # Split by double newlines, bullets, or at least by single newlines if everything is stuck together.
    paragraphs = re.split(r'(?:\n\s*\n|\r\n\s*\r\n|(?<=\n)-\s|\r\s*\r)', text)
    if len(paragraphs) == 1:
        paragraphs = text.split('\n')
    limited = [p.strip() for p in paragraphs if p.strip()][:max_paragraphs]
    if not limited:
        return "[Empty response. Even Arianna cannot extract anything from this.]"
    return '\n\n'.join(limited)
