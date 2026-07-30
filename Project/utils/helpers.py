"""
Generic helper functions used across the RAG and agent layers.
"""

import re
import time
from functools import wraps
from typing import Callable, Any


def clean_text(text: str) -> str:
    """
    Normalize whitespace and strip unwanted control characters from text.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text with normalized whitespace.
    """
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_file_extension(file_name: str) -> str:
    """
    Return the lowercase file extension (without the dot) of a file name.

    Args:
        file_name: Name or path of the file.

    Returns:
        Lowercase extension string, e.g. "pdf".
    """
    return file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""


def timed(func: Callable) -> Callable:
    """
    Decorator that measures and attaches execution time (in seconds) to
    a function's return value when the return value is a dict.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = round(time.perf_counter() - start, 3)
        if isinstance(result, dict):
            result["execution_time"] = elapsed
        return result

    return wrapper


def format_sources(sources: list) -> str:
    """
    Format a list of source metadata dicts into a human-readable string.

    Args:
        sources: List of dicts each containing at least a "source" key.

    Returns:
        A formatted, de-duplicated string listing the source documents.
    """
    if not sources:
        return "No sources used."
    seen = []
    for item in sources:
        name = item.get("source", "unknown")
        if name not in seen:
            seen.append(name)
    return ", ".join(seen)
