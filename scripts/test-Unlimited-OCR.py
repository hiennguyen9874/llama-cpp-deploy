import base64
import mimetypes
from pathlib import Path

import requests

image_path = Path("scripts/test.png")
mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
data = base64.b64encode(image_path.read_bytes()).decode()

response = requests.post(
    "http://127.0.0.1:8080/v1/chat/completions",
    json={
        "model": "Unlimited-OCR",
        "temperature": 0,
        "max_tokens": 8192,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Free OCR."},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{data}"
                }},
            ],
        }],
    },
    timeout=1200,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])