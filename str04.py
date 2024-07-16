import streamlit as st
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
import os
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from moviepy.editor import VideoFileClip
import google.generativeai as genai

# Deepgram API key
DG_KEY = "bd9aae03709c9036ed4751d88bf308f64a5a9df1"

# Google API key
GOOGLE_API_KEY = "AIzaSyA7zsIwxGufvhIHfc-ckxD6dIW5rL7ZNMM"
genai.configure(api_key=GOOGLE_API_KEY)

# Path to save the extracted audio file and transcript JSON file
AUDIO_FILE_PATH = "output_audio.webm"
TRANSCRIPT_FILE = "transcript.json"


def extract_audio_from_video(video_file_path, audio_file_path):
    try:
        # Load the video file
        video = VideoFileClip(video_file_path)

        # Extract the audio
        audio = video.audio

        # Save the audio in WEBA format
        audio.write_audiofile(audio_file_path, codec='libvorbis')

        return audio_file_path

    except Exception as e:
        st.error(f"Exception in extracting audio: {e}")
        return None


def transcribe_audio(audio_file_path):
    try:
        # Create a Deepgram client using the API key
        deepgram = DeepgramClient(DG_KEY)

        # Read the audio file
        with open(audio_file_path, "rb") as audio_file:
            buffer_data = audio_file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        # Configure Deepgram options for audio analysis
        options = PrerecordedOptions(
            model="whisper-large",
            # language='hi',
            smart_format=True,
        )

        # Call the transcribe_file method with the text payload and options
        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

        # Write the response JSON to a file
        with open(TRANSCRIPT_FILE, "w") as transcript_file:
            transcript_file.write(response.to_json(indent=4))

        return response['results']['channels'][0]['alternatives'][0]['transcript']

    except Exception as e:
        st.error(f"Exception in transcribing audio: {e}")
        return None


def generate_wordcloud(transcript):
    # Create a word cloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(transcript)

    # Save the wordcloud as a PNG image
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
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(f"Analyze the sentiment of the following text and give output in single word one of from Positive or Neutral or Negative:\n\n{transcript}")
        return response.text

    except Exception as e:
        st.error(f"Exception in analyzing sentiment: {e}")
        return None


st.title("📺Video Analyzer📽️")

# Upload video file
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mkv", "avi"])

if uploaded_file is not None:
    video_file_path = "uploaded_video." + uploaded_file.name.split('.')[-1]
    with open(video_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(uploaded_file)

    if st.button("Analyze"):
        st.write("Extracting audio from video...")
        audio_file_path = extract_audio_from_video(video_file_path, AUDIO_FILE_PATH)

        if audio_file_path:
            st.write("Transcribing audio...")
            transcript = transcribe_audio(audio_file_path)

            if transcript:
                st.subheader("Transcript")
                st.text_area("Transcript", transcript, height=200)

                transcript_txt_path = save_transcript_to_txt(transcript)
                st.markdown(file_download_link(transcript_txt_path, "Download Transcript as TXT"), unsafe_allow_html=True)

                st.write("Generating word cloud...")
                wordcloud_path = generate_wordcloud(transcript)

                st.image(wordcloud_path, caption='Generated Word Cloud', use_column_width=True)
                st.markdown(file_download_link(wordcloud_path, "Download Word Cloud as PNG"), unsafe_allow_html=True)

                st.write("Analyzing sentiment...")
                sentiment = analyze_sentiment(transcript)
                st.subheader("Sentiment Analysis")
                st.text_area("Sentiment Analysis Result", sentiment, height=200)
else:
    st.info("Please upload a video file.")
