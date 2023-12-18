from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel('gemini-pro')

response = model.generate_content("Hello")

print(response.text)