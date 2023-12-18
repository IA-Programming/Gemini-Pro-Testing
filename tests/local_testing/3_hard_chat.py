from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel('gemini-pro')

messages = []

while True:
    message = input("You: ")
    messages.append({
        "role": "user",
        "parts": [message],
    })

    response = model.generate_content(messages)

    messages.append({
        "role": "model",
        "parts": [response.text],
    })

    print("Gemini: " + response.text)
