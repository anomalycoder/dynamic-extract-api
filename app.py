import os
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from dateutil.parser import parse

from google import genai

# ----------------------------------------------------
# Load Gemini API Key
# ----------------------------------------------------
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# ----------------------------------------------------
# FastAPI
# ----------------------------------------------------
app = FastAPI(
    title="Dynamic Extract API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Request Model
# ----------------------------------------------------
class RequestModel(BaseModel):
    text: str
    schema: dict[str, str]

# ----------------------------------------------------
# Type Conversion
# ----------------------------------------------------
def convert(value, dtype):

    if value is None:
        return None

    try:

        if dtype == "string":
            return str(value)

        elif dtype == "integer":
            return int(value)

        elif dtype == "float":
            return float(value)

        elif dtype == "boolean":

            if isinstance(value, bool):
                return value

            return str(value).lower() in [
                "true",
                "yes",
                "1"
            ]

        elif dtype == "date":
            return parse(str(value)).date().isoformat()

        elif dtype == "array[string]":

            if isinstance(value, list):
                return [str(x) for x in value]

            return [str(value)]

        elif dtype == "array[integer]":

            if isinstance(value, list):
                return [int(x) for x in value]

            return [int(value)]

    except:
        return None

    return None

# ----------------------------------------------------
# Root
# ----------------------------------------------------
@app.get("/")
def home():
    return {
        "status": "running"
    }

# ----------------------------------------------------
# Dynamic Extraction Endpoint
# ----------------------------------------------------
@app.post("/dynamic-extract")
def dynamic_extract(req: RequestModel):

    prompt = f"""
You are an information extraction engine.

Extract information from the text.

Return ONLY valid JSON.

IMPORTANT:

- Return EXACTLY these keys:
{json.dumps(req.schema, indent=2)}

Rules:

1. No extra keys.
2. No missing keys.
3. Missing values -> null.
4. date -> YYYY-MM-DD
5. integer -> JSON integer
6. float -> JSON number
7. boolean -> true/false
8. array[string] -> JSON array of strings
9. array[integer] -> JSON array of integers

TEXT:

{req.text}
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        text = response.text.strip()

        # Remove markdown if Gemini returns it
        if text.startswith("```"):
            text = text.replace("```json", "")
            text = text.replace("```", "")
            text = text.strip()

        data = json.loads(text)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini Error: {e}"
        )

    # Build final output using ONLY schema keys
    output = {}

    for field, dtype in req.schema.items():

        value = data.get(field)

        output[field] = convert(value, dtype)

    return output

# ----------------------------------------------------
# Run
# ----------------------------------------------------
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
