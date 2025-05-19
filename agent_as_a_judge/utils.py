import tiktoken
import logging

def truncate_string(text, model="gpt-3.5-turbo", max_tokens=1000, drop_mode="tail"):
    # ... existing code ...
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        # 不再打印warning，直接fallback
        enc = tiktoken.get_encoding("cl100k_base")
    # ... existing code ... 