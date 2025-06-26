def split_message(text, max_length=4000):
    """
    Splits a long message into parts to avoid truncation by Telegram.
    Arianna does not beautify the text—she simply splits where it is possible.
    """
    result = []
    while len(text) > max_length:
        idx = text.rfind('\n', 0, max_length)
        if idx == -1:
            idx = max_length
        result.append(text[:idx])
        text = text[idx:].lstrip('\n')
    if text:
        result.append(text)
    if not result:
        return ["Even emptiness can be divided."]
    return result
