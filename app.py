import streamlit as st
from sklearn.pipeline import Pipeline
import speech_recognition as sr
from gtts import gTTS
import io
from dotenv import load_dotenv
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder

load_dotenv()

MODEL = "gpt-3.5-turbo"
client = OpenAI()

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
            completion = client.chat.completions.create(
                model=MODEL,
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

st.title("Chatbot Vocale")

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "Sei un chatbot conversazionale, cerca di rispondere all'utente in modo naturale e coinvolgente massimo in 40 parole."}
    ]

if 'conversation' not in st.session_state:
    st.session_state.conversation = []

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = Pipeline([
        ('stt', STTTransformer()),
        ('llm', LLMTransformer(st.session_state.messages)),
        ('tts', TTSTransformer())
    ])

st.write("Clicca il pulsante per iniziare la registrazione. Clicca di nuovo per fermarla.")
audio_bytes = audio_recorder()

if audio_bytes:
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

st.info("Nota: Assicurati di avere un file .env con la tua API key di OpenAI e un microfono collegato.")