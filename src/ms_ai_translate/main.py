from fastapi import FastAPI
from transformers import pipeline, AutoTokenizer
from pydantic import BaseModel
from langdetect import detect

app = FastAPI()

model_name = "facebook/m2m100_418M"
translator = pipeline("translation", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def translate_text(text, target_language):
    source_language = detect(text)
    forced_bos_token_id = tokenizer.lang_code_to_id[target_language]
    result = translator(text, forced_bos_token_id=forced_bos_token_id, max_length=255)
    return result[0]['translation_text']

class Request(BaseModel):
    text: str
    target_language: str

@app.get("/")
async def root():
    return {"message": "System running"}

@app.post("/ai")
async def ai_translate(request: Request):
    try:
        translated_text = translate_text(request.text, request.target_language)
        return {"translated_text": translated_text}
    except Exception as error:
        return {"error": str(error)}

