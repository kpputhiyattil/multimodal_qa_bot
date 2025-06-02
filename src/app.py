from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from openai import OpenAI
import pyttsx3
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import moviepy.editor as mpe
import textwrap
from concurrent.futures import ThreadPoolExecutor

from langchain_implementation import langchain_query

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Configure your OpenAI API key
api_key = "Enter your API key"

# Global variable to store conversation context
conversation_context = {}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def extract_text_from_pdf(pdf_path):
    text = ""
    conversation_context["file"] = pdf_path
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def generate_llm_response(prompt, context=None, temperature=0.7):
    messages = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": prompt})

    client = OpenAI(api_key=api_key)  # Or set OPENAI_API_KEY as env variable

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    print(response)
    print(response.choices[0].message.content)

    return response.choices[0].message.content


def text_to_speech(text, filename):
    """Fallback using pyttsx3 when FFmpeg isn't available"""
    if not filename.endswith('.wav'):
        filename += '.wav'
    
    try:
        engine = pyttsx3.init()
        engine.save_to_file(text, filename)
        engine.runAndWait()
        return filename
    except Exception as e:
        raise RuntimeError(f"TTS failed: {str(e)}")


def generate_speaking_avatar(
    text, 
    audio_path,
    output_video,  # Must use .mp4 extension
    avatar_closed="avatar_male_closed.png", 
    avatar_open="avatar_male_open.png",
    max_workers=4
):
    # Load audio - convert to compatible format if needed
    audio = mpe.AudioFileClip(audio_path)
    
    duration = audio.duration
    
    # Preload avatar images
    closed_avatar = Image.open(avatar_closed)
    open_avatar = Image.open(avatar_open)
    closed_avatar.thumbnail((200, 200))
    open_avatar.thumbnail((200, 200))
    
    # Prepare font
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    wrapped_text = textwrap.fill(text, width=50)
    fps = 24
    mouth_change_rate = 0.2
    
    # Create frames in memory without saving to disk
    def create_frame(t):
        avatar_img = open_avatar if (t % mouth_change_rate) < mouth_change_rate/2 else closed_avatar
        frame = Image.new('RGB', (800, 600), (73, 109, 137))
        frame.paste(avatar_img, (50, 50))
        draw = ImageDraw.Draw(frame)
        draw.text((50, 300), wrapped_text, fill=(255, 255, 255), font=font)
        return np.array(frame)  # Convert directly to numpy array
    
    # Parallel frame creation
    times = np.arange(0, duration, 1/fps)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        frames = list(executor.map(create_frame, times))
    
    # Create video clip from frames
    video = mpe.ImageSequenceClip(frames, fps=fps)
    video = video.set_audio(audio)
    
    # Write video with proper codec settings
    video.write_videofile(
        output_video,
        fps=fps,
        codec='libx264',  # Proper codec for MP4
        audio_codec='aac',  # Proper audio codec for MP4
        threads=4,
        preset='fast',
        ffmpeg_params=[
            '-crf', '23',  # Quality setting
            '-pix_fmt', 'yuv420p'  # Ensures compatibility with most players
        ]
    )
    
    return output_video


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract text from PDF
            pdf_text = extract_text_from_pdf(filepath)

            # Generate initial summary
            summary_prompt = """Please provide a comprehensive summary of the following document.
            Highlight key concepts, main arguments, and important details.
            Also explain any complex concepts in simple terms:"""

            summary = generate_llm_response(f"{summary_prompt}\n\n{pdf_text}")

            # Store context for follow-up questions
            conversation_context['original_text'] = pdf_text
            conversation_context['summary'] = summary

            # Generate audio and video
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'summary_audio.wav')
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'summary_video.mp4')
            print("Text to speech conversion")
            text_to_speech(summary, audio_path)
            # create_avatar_video(summary, video_path)
            # create_avatar_video(summary, audio_path, video_path)
            print("Speaking avatar generation")
            generate_speaking_avatar(summary, audio_path, video_path)

            # create_video_from_text(summary, audio_path, video_path)

            # Prepare response
            response = {
                'summary': summary,
                'audio_url': f"/uploads/{os.path.basename(audio_path)}",
                'video_url': f"/uploads/{os.path.basename(video_path)}"
            }

            return jsonify(response)
    except Exception as e:
        print(e)

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    if 'original_text' not in conversation_context:
        return jsonify({'error': 'No document context available'}), 400
    file = conversation_context["file"]
    result = langchain_query(file, api_key, question)
    # Generate answer using LLM with context
    answer = generate_llm_response(
        f"for the question: {question}, the answer generated is {result}. Format the answer",
        context=f"Document text: {conversation_context['original_text']}\n\nSummary: {conversation_context['summary']} Answer:{result}"
    )
    return jsonify({'answer': answer})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
