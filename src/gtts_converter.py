from gtts import gTTS
import os

# Your input text
text = "Hello! This is a test of the Google Text-to-Speech library using Python."

# Set language (e.g., 'en' for English, 'hi' for Hindi)
language = 'en'

# Convert text to speech
tts = gTTS(text=text, lang=language, slow=False)  # slow=True for slower speech

# Save the audio file
tts.save("output.mp3")

# Play the saved audio (works on Windows/macOS/Linux depending on default player)
# os.system("start output.mp3")   # Windows
# os.system("afplay output.mp3") # macOS
# os.system("mpg123 output.mp3") # Linux (requires mpg123 installed)
