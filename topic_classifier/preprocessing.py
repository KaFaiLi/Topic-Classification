"""
Phase 1: Email Preprocessing and Noise Removal
"""
import re
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup


def parse_email(file_path: str) -> str:
    """
    Parse raw email file and return combined plain text.
    """
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    text = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                text += part.get_content()
            elif part.get_content_type() == "text/html":
                soup = BeautifulSoup(part.get_content(), 'html.parser')
                text += soup.get_text(separator="\n")
    else:
        if msg.get_content_type() == "text/plain":
            text = msg.get_content()
        elif msg.get_content_type() == "text/html":
            soup = BeautifulSoup(msg.get_content(), 'html.parser')
            text = soup.get_text(separator="\n")
    return text


def remove_signature(text: str) -> str:
    """
    Remove common signatures and disclaimers.
    """
    patterns = [
        r"(?m)^--\s*$.*",      # signature delimiter
        r"(?mi)^regards,.*$",  # Regards,
        r"(?mi)^sincerely,.*$"  # Sincerely,
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.MULTILINE|re.IGNORECASE|re.DOTALL)
    return text


def remove_forwarded_text(text: str) -> str:
    """Remove forwarded message blocks."""
    patterns = [
        r"(?m)^[-]{2,} forwarded message [-]{2,}$",
        r"(?m)^_{2,}$"
    ]
    for pat in patterns:
        text = re.sub(pat + r"[\s\S]*", "", text)
    return text


def remove_quoted_text(text: str) -> str:
    """Remove quoted reply text."""
    # remove 'On ... wrote:' blocks
    text = re.sub(r"(?s)On .+?wrote:.*", "", text)
    # remove lines starting with '>'
    text = re.sub(r"(?m)^>.*$", "", text)
    return text


def remove_short_lines(text: str, min_len: int = 20) -> str:
    """Remove lines shorter than min_len characters (likely boilerplate)."""
    lines = [line for line in text.splitlines() if len(line.strip()) >= min_len]
    return "\n".join(lines)


def clean_text(text: str) -> str:
    """
    Full cleaning: remove signatures, boilerplate, extra whitespace.
    """
    text = remove_signature(text)
    text = remove_forwarded_text(text)
    text = remove_quoted_text(text)
    # collapse multiple blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    # remove short boilerplate lines
    text = remove_short_lines(text)
    # normalize whitespace within lines
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(lines).strip()
