import streamlit as st
import requests
import base64
import os
import llama_index
from transformers import pipeline
from audio_recorder_streamlit import audio_recorder
from openai import OpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from scipy.io.wavfile import write
os.environ['OPENAI_API_KEY'] = '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'
API_KEY = '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'
def RAG(text):
    documents = SimpleDirectoryReader("db3").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query(text)
    
    # Extract the text from the response
    response_text = response.response if hasattr(response, 'response') else str(response)

    return response_text
def linkRAGhindi(text):
    new_prompt="निम्नलिखित प्रश्न के लिए सबसे उपयुक्त वेबसाइट लिंक दें"+text
    documents = SimpleDirectoryReader("db1").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query(new_prompt)
    
    # Extract the text from the response
    response_text = response.response if hasattr(response, 'response') else str(response)

    return response_text
def rechindi(text):
    new_prompt="निम्नलिखित प्रश्न के लिए सबसे उपयुक्त वेबसाइट लिंक दें"+text
    documents = SimpleDirectoryReader("db2").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query(new_prompt)
    
    # Extract the text from the response
    response_text = response.response if hasattr(response, 'response') else str(response)
    return response_text
def linkRAGenglish(text):
    new_prompt="Give the most appropiate website link for the following question "+text
    documents = SimpleDirectoryReader("db1").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query(new_prompt)
    # Extract the text from the response
    response_text = response.response if hasattr(response, 'response') else str(response)
    return response_text
def recenglish(text):
    new_prompt="Give the most intresting other website link for the following question "+text
    documents = SimpleDirectoryReader("db2").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query(new_prompt)
    # Extract the text from the response
    response_text = response.response if hasattr(response, 'response') else str(response)
    return response_text
def transcribe_text_to_voice_english(audio_location):
    client = OpenAI(api_key=API_KEY)
    audio_file = open(audio_location, "rb")
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return transcript.text

def transcribe_text_to_voice_hindi(audio_location):
    url = "https://api.runpod.ai/v2/faster-whisper/runsync"
    
    with open(audio_location, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

    payload = {
        "input": {
            "audio_base64": audio_base64,
            "model": "small",
            "transcription": "plain_text",
            "translate": True,
            "language": "hi",
            "temperature": 0,
            "best_of": 5,
            "beam_size": 5,
            "patience": 1,
            "suppress_tokens": "-1",
            "condition_on_previous_text": False,
            "temperature_increment_on_fallback": 0.2,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1,
            "no_speech_threshold": 0.6,
            "word_timestamps": False
        },
        "enable_vad": False
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "X01PG949AHTOVRYHLQZKSRIWN82UHBUU5JYLNAHM"
    }

    response = requests.post(url, json=payload, headers=headers)
    response_json = response.json()
    transcription = response_json["output"]["transcription"]
    translation = response_json["output"]["translation"].strip().split('\n')[-1].strip()
    return transcription, translation


def recommendation(text):
    client = OpenAI(api_key=API_KEY)
    messages = [{"role": "user", "content": text}]
    response = client.chat.completions.create(model="gpt-3.5-turbo-1106", messages=messages)
    return response.choices[0].message.content
def text_to_speech_ai(speech_file_path, api_response):
    client = OpenAI(api_key=API_KEY)
    response = client.audio.speech.create(model="tts-1",voice="nova",input=api_response)
    response.stream_to_file(speech_file_path)





st.title("🚀 SHRESHTH 💬 Bhuvan Assistant")

# Radio wheel for language selection
language = st.radio("Language/भाषा",["English", "हिंदी"])
# Displaying description based on selected language
if language == "English":
    mode = st.radio("Select Mode Of Input", ["Voice","Text"])
    st.write("Smart - Helpful - Robust - Effortless - System for Text-to-speech and Human-like Assistance")
    if mode == "Voice" or mode == "आवाज":
        st.write("Click on the voice recorder and let me know how I can help you today with your Queries Regarding Bhuvan!")
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
        )

        if audio_bytes:
            # Save the Recorded File
            audio_location = "audio_file.wav"
            with open(audio_location, "wb") as f:
                f.write(audio_bytes)
    
            if language == "English":
               text=transcribe_text_to_voice_english(audio_location)
               st.write(text)
            else:
               text,trans=transcribe_text_to_voice_hindi(audio_location)
               st.write(text)


            link_response = linkRAGenglish(text)
            st.write("SHRESHTH:", link_response)
            api_response = RAG(text)
            st.write("SHRESHTH:", api_response)
            speech_file_path = 'audio_response.mp3'
            text_to_speech_ai(speech_file_path, api_response)
            st.audio(speech_file_path)
            recctext="recommend top three other websites that could interest the user depending on this link and answer : " + link_response + api_response
            recc=linkRAGenglish(recctext)  
            st.write("SHRESHTH:", recc)
    else: 
        # Text input option
        text_input = st.text_area("Enter your text here and press Enter", "")
        if st.button("Submit"):
            # Process the entered text
            link_response = linkRAGenglish(text_input)
            st.write("SHRESHTH:", link_response)
            api_response = RAG(text_input)
            st.write("SHRESHTH:", api_response)
            # Read out the text response using tts
            speech_file_path = 'audio_response.mp3'
            text_to_speech_ai(speech_file_path, api_response)
            st.audio(speech_file_path)
            recctext="recommend top three other websites that could interest the user depending on this link and answer : " + link_response + api_response
            recc=linkRAGenglish(recctext)  
            st.write("SHRESHTH:", recc)
else:
    mode = st.radio("इनपुट मोड का चयन करें", ["आवाज", "टेक्स्ट"])
    st.write("स्मार्ट - सहायक - मजबूत - प्रयासहीन - पाठ-से-बोल के लिए एक सिस्टम और मानव जैसी सहायता")

    if mode == "Voice" or mode == "आवाज" or mode == "ভয়েস":
        st.write("आवाज रेकॉर्डर पर क्लिक करें और मुझसे यह बताएं कि आज आपकी भुवन से संबंधित सवालों में मैं आपकी कैसे मदद कर सकता हूँ!")
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
        )

        if audio_bytes:
            # Save the Recorded File
            audio_location = "audio_file.wav"
            with open(audio_location, "wb") as f:
                f.write(audio_bytes)

            if language == "English":
               text=transcribe_text_to_voice_english(audio_location)
               st.write(text)
            else:
               text,trans=transcribe_text_to_voice_hindi(audio_location)
               st.write(text)
    
            link_response = linkRAGhindi(text)
            st.write("श्रेष्ठ:", link_response)
            api_response = RAG(text)
            st.write("श्रेष्ठ:", api_response)      
            # Read out the text response using tts
            speech_file_path = 'audio_response.mp3'
            text_to_speech_ai(speech_file_path, api_response)
            st.audio(speech_file_path)
            recctext="recommend top three other websites that could interest the user depending on this link and answer : " + link_response + api_response
            recc=rechindi(recctext)  
            st.write("श्रेष्ठ:", recc)
            
    else: 
        # Text input option
        text_input = st.text_area("आप यहाँ अपना टेक्स्ट दर्ज करें और एंटर दबाएं", "")
        if st.button("एंटर"):
            # Process the entered text
            link_response = linkRAGhindi(text_input)
            st.write("श्रेष्ठ:", link_response)
            api_response = RAG(text_input)
            st.write("श्रेष्ठ:", api_response)
    
            # Read out the text response using tts
            speech_file_path = 'audio_response.mp3'
            text_to_speech_ai(speech_file_path, api_response)
            st.audio(speech_file_path)
            recctext="recommend top three other websites that could interest the user depending on this link and answer : " + link_response + api_response
            recc=rechindi(recctext)  
            st.write("श्रेष्ठ:", recc)
