import base64
import json
import mimetypes
import sys
import urllib.request
from pathlib import Path

PROMPT = """Extract all readable content from the image in natural human reading order and output the result as a single Markdown document. For charts or images, represent them using an HTML image tag: <img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as HTML: <table>...</table>. Transcribe all other text as standard Markdown. Preserve the original text without translation or paraphrasing."""

path = Path(sys.argv[1])
mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
data_url = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()

payload = {
    "model": "ovisocr2",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": PROMPT},
        ],
    }],
    "temperature": 0,
    "max_tokens": 16384,
    "stream": False,
}

request = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=3600) as response:
    result = json.load(response)

markdown = result["choices"][0]["message"]["content"].strip()
print(markdown)