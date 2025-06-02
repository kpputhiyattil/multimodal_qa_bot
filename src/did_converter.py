import requests

api_key = "YOUR_DID_API_KEY"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "script": {
        "type": "text",
        "input": "Hello, I am an AI avatar created using D-ID.",
        "provider": {"type": "microsoft", "voice_id": "en-US-JennyNeural"}
    },
    "source_url": "https://create-images-results.d-id.com/YourAvatarImage.jpg",
    "config": {"fluent": True}
}

response = requests.post("https://api.d-id.com/talks", json=data, headers=headers)
print(response.json())
