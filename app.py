import re
import os
from dotenv import load_dotenv
from typing import Any, Union
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("openai_apikey")
groq_api_key = os.getenv("groq_apikey")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")


def create_summary(transcription: str) -> str:
    """Creates summary of the given text using hugging face model.

    Args:
        transcription (str): string transcription data for summarization

    Returns:
        str: summarized text string
    """
    # text_summary = pipeline(
    #     "summarization",
    #     model="sshleifer/distilbart-cnn-12-6",
    #     torch_dtype=torch.bfloat16,
    # )
    prompt = """
    You are a professional text summarizer, you read the input text given to you
    and then return the summarized version of the text while keeping the 
    meaning relevant and accurate. Please provide the accurate summarization of the context given to you
    <context>
    {transcription}
    """

    summary_template = PromptTemplate(
        input_variables=["transcription"],
        template=prompt,
    )

    chain = LLMChain(llm=llm, prompt=summary_template, verbose=True)
    summary = chain.run(transcription)
    # summary = text_summary(transcription)
    return summary
    # return summary[0]["summary_text"]


def get_youtube_transcription(video_url: str) -> str:
    """Handler method to
    1. Fetch the video id from youtube
    2. Fetch the transcription/captions of the video
    3. Convert the transcription object to str plain text
    4. Create & return summary

    Args:
        video_url (str): URL of the youtube video

    Returns:
        str: Summary of the transcription
    """
    # Step 1 : Fetch the video ID from the URL
    video_id = extract_video_id(video_url)
    if not video_id:
        return "Video ID could not be extracted :| "
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        # [{'text': "in today's video we will create our", 'start': 0.08, 'duration': 3.48}....]
        # format this transcription into raw text format
        formatter = TextFormatter()
        text = formatter.format_transcript(transcript)
        return create_summary(text)
    except Exception as err:
        return f"Error occured: {err}"


def extract_video_id(url: str) -> Union[str, None]:
    """Strips id from the url

    Args:
        url (str): url of the youtube video

    Returns:
        Union[str, None]: return id if its present or None
    """
    # Regex to extract the video ID from various YouTube URL formats
    regex = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None


# def main():
#     url = "https://www.youtube.com/watch?v=r2u4Z9jCC04"
#     print("Inside main")
#     print(get_youtube_transcription(url))

st.title("Youtube summarizer")
st.subheader("Summarize your youtube video in seconds")
url = st.text_input("Enter the youtube URL")
submit_button = st.button("Submit")

if submit_button:
    st.write(get_youtube_transcription(url))
