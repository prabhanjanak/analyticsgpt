from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
import speech_recognition as sr
import plotly.express as px
from elevenlabs import text_to_speech, save

# Load environment variables
load_dotenv()

# Configure ElevenLabs API Key
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
if not elevenlabs_api_key:
    raise ValueError("ElevenLabs API key not found. Please set ELEVENLABS_API_KEY in the .env file.")

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the GOOGLE_API_KEY in the .env file.")

genai.configure(api_key=api_key)

# Function to load Gemini Pro model
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Streamlit UI Layout
st.set_page_config(page_title="Voice-Controlled Chart Assistant", page_icon="🎙️", layout="wide")

# ---- LOGO & TITLE ----
col1, col2 = st.columns([0.2, 0.8])  # Adjust the width for logo & title

with col1:
    st.image("logo.png", width=150)  # Replace with actual logo filename

with col2:
    st.title("🎙️ Voice-Controlled Chart Assistant (Gemini)")

# Embed JavaScript to request microphone and speaker permissions
st.components.v1.html("""
    <script>
        async function requestPermissions() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                alert('Microphone permission granted!');
                stream.getTracks().forEach(track => track.stop());
            } catch (err) {
                alert('Microphone permission denied!');
            }
        }

        document.addEventListener('DOMContentLoaded', (event) => {
            requestPermissions();
        });
    </script>
    """, height=0)

# Function to Capture Audio and Convert to Text
def record_audio():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.write("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        st.write("🔄 Processing...")

        try:
            text = recognizer.recognize_google(audio)  # Uses Google Speech API for STT
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "API unavailable"

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Start Listening and Process Audio
if st.button("🎙️ Start Listening"):
    user_input = record_audio()

    if user_input:
        st.success(f"🗣️ You said: {user_input}")

        # Process with Google Gemini
        response = get_gemini_response(user_input)
        full_response = ""  # Ensure full_response is always initialized

        st.subheader("🤖 AI Response:")
        for chunk in response:
            st.write(chunk.text)
            full_response += chunk.text + " "  # Constructing full_response correctly
            st.session_state['chat_history'].append(("Bot", chunk.text))

        # Ensure full_response is valid before using it in text_to_speech
        if full_response.strip():  # Prevents empty response errors
            audio = text_to_speech(full_response, voice="Bella")  # Removed `api_key` (Fix for TypeError)
            save(audio, "response.mp3")  # Saves the audio file

            # Provide a download link for the audio file
            with open("response.mp3", "rb") as file:
                btn = st.download_button(
                    label="Download Audio Response",
                    data=file,
                    file_name="response.mp3",
                    mime="audio/mp3"
                )
        else:
            st.warning("No valid response generated for TTS.")

        # Example Chart (Can be modified based on Gemini's response)
        df = px.data.iris()
        fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
        st.plotly_chart(fig)

# Display Chat History
st.subheader("📝 Chat History:")
for role, text in st.session_state['chat_history']:
    st.write(f"**{role}:** {text}")
