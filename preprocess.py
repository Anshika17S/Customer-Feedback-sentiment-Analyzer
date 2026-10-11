import re

def clean_text(text):
    # Convert to lowercase
    text = str(text).lower()
    # Replace anything that is not a letter or space with a space
    text = re.sub(r"[^a-z\s]", " ", text)
    # Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
