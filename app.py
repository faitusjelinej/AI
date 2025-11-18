from flask import Flask, render_template, request, jsonify
from openai import AzureOpenAI
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
    api_version="2024-02-01"
)


def get_weather(city):
    weather_data = requests.get(
        f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={os.getenv('OPENWEATHER_API_KEY')}&units=metric"
    ).json()
    return {
        "city": weather_data["name"],
        "temp": weather_data["main"]["temp"],
        "description": weather_data["weather"][0]["description"],
        "humidity": weather_data["main"]["humidity"]
    }

functions = [{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"]
    }
}]    



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    
    response = client.chat.completions.create(
        model="gpt-5-chat",
        messages=[{"role": "user", "content": user_message}],
        functions=functions,
        function_call="auto"
    )
    
    message = response.choices[0].message
    
    if message.function_call:
        func_args = json.loads(message.function_call.arguments)
        weather_data = get_weather(func_args["city"])
        
        final_response = client.chat.completions.create(
            model="gpt-5-chat",
            messages=[
                {"role": "user", "content": user_message},
                {"role": "function", "name": "get_weather", "content": json.dumps(weather_data)}
            ]
        )
        return jsonify({"response": final_response.choices[0].message.content})
    else:
        return jsonify({"response": message.content})

if __name__ == '__main__':
    app.run(debug=True, port=5000)