import streamlit as st
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
import os
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from moviepy.editor import VideoFileClip
from openai import OpenAI

# ----------------------- API KEYS -----------------------
DG_KEY = "317676cd90279f24e3abec7f20762c0ee225a58b"
OPENAI_API_KEY = "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"  # 🔹 Replace with your OpenAI API key
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------- FILE PATHS -----------------------
AUDIO_FILE_PATH = "output_audio.webm"
TRANSCRIPT_FILE = "transcript.json"

# ----------------------- FUNCTIONS -----------------------
def extract_audio_from_video(video_file_path, audio_file_path):
    try:
        video = VideoFileClip(video_file_path)
        audio = video.audio
        audio.write_audiofile(audio_file_path, codec='libvorbis')
        return audio_file_path
    except Exception as e:
        st.error(f"Exception in extracting audio: {e}")
        return None


def transcribe_audio(audio_file_path):
    try:
        deepgram = DeepgramClient(DG_KEY)
        with open(audio_file_path, "rb") as audio_file:
            buffer_data = audio_file.read()

        payload: FileSource = {"buffer": buffer_data}

        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
        )

        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

        with open(TRANSCRIPT_FILE, "w") as transcript_file:
            transcript_file.write(response.to_json(indent=4))

        return response['results']['channels'][0]['alternatives'][0]['transcript']

    except Exception as e:
        st.error(f"Exception in transcribing audio: {e}")
        return None


def generate_wordcloud(transcript):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(transcript)
    wordcloud_path = "wordcloud.png"
    wordcloud.to_file(wordcloud_path)
    return wordcloud_path


def save_transcript_to_txt(transcript):
    transcript_file_path = "transcript.txt"
    with open(transcript_file_path, "w") as out_file:
        out_file.write(transcript)
    return transcript_file_path


def file_download_link(file_path, file_label):
    with open(file_path, "rb") as file:
        contents = file.read()
        b64 = base64.b64encode(contents).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="{file_path}">{file_label}</a>'
    return href


def analyze_sentiment(transcript):
    try:
        # 🔹 Using GPT-4.1-nano for sentiment analysis
        prompt = (
            "Analyze the sentiment of the following text and give output "
            "in a single word: Positive, Neutral, or Negative.\n\n"
            f"{transcript}"
        )

        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Exception in analyzing sentiment: {e}")
        return None

# ----------------------- STREAMLIT APP -----------------------
st.title("📺 Video Analyzer with GPT-4.1-nano 🎥")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mkv", "avi"])

if uploaded_file is not None:
    video_file_path = "uploaded_video." + uploaded_file.name.split('.')[-1]
    with open(video_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(uploaded_file)

    if st.button("Analyze"):
        st.write("🎵 Extracting audio from video...")
        audio_file_path = extract_audio_from_video(video_file_path, AUDIO_FILE_PATH)

        if audio_file_path:
            st.write("🗣️ Transcribing audio...")
            transcript = transcribe_audio(audio_file_path)

            if transcript:
                st.subheader("📝 Transcript")
                st.text_area("Transcript", transcript, height=200)

                transcript_txt_path = save_transcript_to_txt(transcript)
                st.markdown(file_download_link(transcript_txt_path, "Download Transcript as TXT"), unsafe_allow_html=True)

                st.write("☁️ Generating word cloud...")
                wordcloud_path = generate_wordcloud(transcript)
                st.image(wordcloud_path, caption='Generated Word Cloud', use_column_width=True)
                st.markdown(file_download_link(wordcloud_path, "Download Word Cloud as PNG"), unsafe_allow_html=True)

                st.write("🧠 Analyzing sentiment using GPT-4.1-nano...")
                sentiment = analyze_sentiment(transcript)
                st.subheader("💬 Sentiment Analysis Result")
                st.text_area("Sentiment Analysis", sentiment, height=100)
else:
    st.info("Please upload a video file.")
