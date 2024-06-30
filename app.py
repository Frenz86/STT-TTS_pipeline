import streamlit as st
from sklearn.pipeline import Pipeline
import speech_recognition as sr
from gtts import gTTS
import io
from dotenv import load_dotenv
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder
import os

load_dotenv()

class STTTransformer:
    def transform(self, audio_bytes):
        recognizer = sr.Recognizer()
        with io.BytesIO(audio_bytes) as audio_file:
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="it-IT")
            return text
        except sr.UnknownValueError:
            return "Speech Recognition non ha compreso l'audio"
        except sr.RequestError as e:
            return f"Errore nel servizio Speech Recognition; {e}"

class LLMTransformer:
    def __init__(self, messages):
        self.messages = messages

    def transform(self, text):
        try:
            self.messages.append({"role": "user", "content": text})
            completion = st.session_state.client.chat.completions.create(
                model=st.session_state.MODEL,
                messages=self.messages
            )
            response = completion.choices[0].message.content
            self.messages.append({"role": "assistant", "content": response})
            return response
        except Exception as e:
            return f"Errore nell'elaborazione LLM: {str(e)}"

class TTSTransformer:
    def transform(self, text):
        tts = gTTS(text=text, lang='it')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp

def main():
    # --- Page Config ---
    st.set_page_config(
        page_title="Chatbot Vocale",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # --- Header ---
    st.title("Chatbot Vocale")

    # --- Side Bar ---
    with st.sidebar:
        cols_keys = st.columns(2)
        with cols_keys[0]:
            default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else ""
            with st.markdown("🔐 OpenAI"):
                openai_api_key = st.text_input("Introduce your OpenAI API Key (https://platform.openai.com/)", value=default_openai_api_key, type="password")
    
    # --- Main Content ---
    if (openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key):
        st.write("#")
        st.warning("⬅️ Please Inserisci API di OpenAI") 
    else:
        st.session_state.client = OpenAI(api_key=openai_api_key)
        st.session_state.MODEL = "gpt-3.5-turbo"

    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "Sei un chatbot conversazionale, cerca di rispondere all'utente in modo naturale e coinvolgente massimo in 40 parole."}
        ]

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    if 'pipeline' not in st.session_state and 'client' in st.session_state and 'MODEL' in st.session_state:
        st.session_state.pipeline = Pipeline([
            ('stt', STTTransformer()),
            ('llm', LLMTransformer(st.session_state.messages)),
            ('tts', TTSTransformer())
        ])

    st.write("Clicca il pulsante per iniziare la registrazione. Clicca di nuovo per fermarla.")
    audio_bytes = audio_recorder()

    if audio_bytes and 'client' in st.session_state and 'MODEL' in st.session_state:
        st.audio(audio_bytes, format="audio/wav")

        with st.spinner("Elaborazione in corso..."):
            try:
                stt_result = st.session_state.pipeline.named_steps['stt'].transform(audio_bytes)
                llm_result = st.session_state.pipeline.named_steps['llm'].transform(stt_result)
                tts_result = st.session_state.pipeline.named_steps['tts'].transform(llm_result)

                st.session_state.conversation.append(("Utente", stt_result))
                st.session_state.conversation.append(("Assistente", llm_result))

                st.audio(tts_result, format='audio/mp3')
            
            except Exception as e:
                st.error(f"Si è verificato un errore: {str(e)}")

    # Visualizza la conversazione
    for speaker, message in st.session_state.conversation:
        if speaker == "Utente":
            st.write(f"👤 **Utente**: {message}")
        else:
            st.write(f"🤖 **Assistente**: {message}")

if __name__=="__main__":
    main()