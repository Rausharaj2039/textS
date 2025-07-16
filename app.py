from flask import Flask, request, render_template, jsonify
import requests
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch

app = Flask(__name__)

# Load local mT5 model and tokenizer for multilingual summarization
MODEL_NAME = "google/mt5-base"
tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Translation API details (still uses HuggingFace for translation)
API_TOKEN = "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"

LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "mr": "Marathi",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "ru": "Russian",
    "zh": "Chinese",
    # ... aur bhi add kar sakte hain (see model card)
}

def translate_text(text, src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    payload = {"inputs": text}
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0 and "translation_text" in result[0]:
            return result[0]["translation_text"]
        elif isinstance(result, dict) and "error" in result:
            if src_lang == "en" and tgt_lang == "gu":
                mid_model1 = "Helsinki-NLP/opus-mt-en-hi"
                mid_api1 = f"https://api-inference.huggingface.co/models/{mid_model1}"
                mid_resp1 = requests.post(mid_api1, headers=headers, json=payload)
                if mid_resp1.status_code == 200:
                    mid_result1 = mid_resp1.json()
                    if isinstance(mid_result1, list) and len(mid_result1) > 0 and "translation_text" in mid_result1[0]:
                        hindi_text = mid_result1[0]["translation_text"]
                        mid_model2 = "Helsinki-NLP/opus-mt-hi-gu"
                        mid_api2 = f"https://api-inference.huggingface.co/models/{mid_model2}"
                        mid_payload2 = {"inputs": hindi_text}
                        mid_resp2 = requests.post(mid_api2, headers=headers, json=mid_payload2)
                        if mid_resp2.status_code == 200:
                            mid_result2 = mid_resp2.json()
                            if isinstance(mid_result2, list) and len(mid_result2) > 0 and "translation_text" in mid_result2[0]:
                                return mid_result2[0]["translation_text"]
                            else:
                                return "Translation API Error: Could not complete multi-hop translation (Hindi to Gujarati)."
                        else:
                            return "Translation API Error: Could not complete multi-hop translation (Hindi to Gujarati)."
                    else:
                        return "Translation API Error: Could not complete multi-hop translation (English to Hindi)."
                else:
                    return "Translation API Error: Could not complete multi-hop translation (English to Hindi)."
            return f"Translation API Error: {result['error']}"
        else:
            return str(result)
    else:
        return f"Translation API Error: {response.status_code} - {response.text}"

def summarize_text(text, min_length=30, max_length=150, language="en"):
    # mT5 model supports multilingual summarization
    input_text = "summarize: " + text.strip()
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def language_flag(code):
    flags = {
        'en': '🇬🇧', 'hi': '🇮🇳', 'bn': '🇧🇩', 'gu': '🇮🇳', 'mr': '🇮🇳', 'pa': '🇮🇳',
        'ta': '🇮🇳', 'te': '🇮🇳', 'ur': '🇵🇰', 'fr': '🇫🇷', 'es': '🇪🇸', 'de': '🇩🇪',
        'ru': '🇷🇺', 'zh': '🇨🇳',
        # Add more as needed
    }
    return flags.get(code, '')

app.jinja_env.globals.update(language_flag=language_flag)

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = None
    text = ''
    min_length = 30
    max_length = 150
    selected_language = "en"
    selected_input_language = "en"
    if request.method == 'POST':
        text = request.form['text']
        min_length = int(request.form.get('min_length', 30))
        max_length = int(request.form.get('max_length', 150))
        selected_language = request.form.get('language', 'en')
        selected_input_language = request.form.get('input_language', 'en')
        if selected_input_language != selected_language:
            translated_text = translate_text(text, selected_input_language, selected_language)
            summary = summarize_text(translated_text, min_length=min_length, max_length=max_length, language=selected_language)
        else:
            summary = summarize_text(text, min_length=min_length, max_length=max_length, language=selected_language)
    return render_template('index.html', summary=summary, text=text, languages=LANGUAGES, selected_language=selected_language, selected_input_language=selected_input_language)

@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    data = request.get_json()
    text = data.get('text', '')
    min_length = int(data.get('min_length', 30))
    max_length = int(data.get('max_length', 150))
    language = data.get('language', 'en')
    input_language = data.get('input_language', 'en')
    if input_language != language:
        translated_text = translate_text(text, input_language, language)
        summary = summarize_text(translated_text, min_length=min_length, max_length=max_length, language=language)
    else:
        summary = summarize_text(text, min_length=min_length, max_length=max_length, language=language)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    # Local development ke liye: python app.py
    # Browser me: http://127.0.0.1:5000/
    # Agar aapko port ya host change karna hai (e.g. cloud/Render/Heroku):
    # app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(debug=True) 