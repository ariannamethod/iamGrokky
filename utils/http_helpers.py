import logging


def check_httpx_response(response):
    """Validate httpx response status and log details on error."""
    if not (200 <= response.status_code < 300):
        url = getattr(response.request, "url", "unknown")
        text = response.text
        logging.error("HTTPX error %s for %s: %s", response.status_code, url, text)
        if response.status_code == 404:
            raise RuntimeError(f"URL not found: {url}")


def check_openai_response(response):
    """Validate OpenAI response status if available and log on error."""
    status = getattr(response, "http_status", None)
    http_resp = getattr(response, "http_response", None) or getattr(response, "response", None)
    if status is None and http_resp is not None:
        status = getattr(http_resp, "status_code", None)
    url = getattr(getattr(http_resp, "request", http_resp), "url", "openai") if http_resp else "openai"
    text = getattr(http_resp, "text", "") if http_resp else ""
    if status is not None and not (200 <= status < 300):
        logging.error("OpenAI error %s for %s: %s", status, url, text)
        if status == 404:
            raise RuntimeError(f"OpenAI endpoint not found: {url}")


def log_openai_exception(exc):
    """Log exception from OpenAI client with status if available."""
    status = getattr(exc, "status_code", None)
    resp = getattr(exc, "response", None)
    url = getattr(getattr(resp, "request", resp), "url", "openai") if resp else "openai"
    text = getattr(resp, "text", "") if resp else ""
    logging.error("OpenAI exception %s for %s: %s", status, url, text)
    if status == 404:
        raise RuntimeError(f"OpenAI endpoint not found: {url}") from exc
