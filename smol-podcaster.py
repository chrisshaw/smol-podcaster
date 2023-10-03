import argparse
import requests
from dotenv import load_dotenv
import os
from datetime import datetime
import json

import replicate
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")
anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def transcribe_audio(file_path, episode_name):
    with open(file_path, "rb") as f:
        output = replicate.run(
            "thomasmol/whisper-diarization:7e5dafea13d80265ea436e51a310ae5103b9f16e2039f54de4eede3060a61617",
            input={
                "file": f,
                "num_speakers": 2,
                "prompt": "A user interview between an UPchieve edtech product manager and a student, volunteer tutor, or other user"
            }
        )
    
    with open(f"./interviews-raw-transcripts/{episode_name}.json", "w") as f:
        json.dump(output, f)

    return output['segments']

def process_transcript(transcript, episode_name):
    """
    {
        "end": "3251",
        "text": " This was great.  Yeah, this has been really fun.",
        "start": "3249",
        "speaker": "SPEAKER 1"
    }
        
    The transcript argument of this function is an array of these. 
    """
    
    transcript_strings = []
    
    for entry in transcript:
        speaker = entry["speaker"]
        text = entry["text"]

        # Divide "end" value by 60 and convert to hours, minutes and seconds
        seconds = int(entry["end"])
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)

        timestamp = "[{:02d}:{:02d}:{:02d}]".format(hours, minutes, seconds)

        transcript_strings.append(f"**{speaker}**: {text} {timestamp}")
        
    clean_transcript = "\n\n".join(transcript_strings)
    
    with open(f"./interviews-clean-transcripts/{episode_name}.md", "w") as f:    
        f.write(clean_transcript)
        
    return clean_transcript
    
 
def create_chapters(transcript):
    anthropic = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
        
    chapters = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=3000,
        prompt=f"{HUMAN_PROMPT} Here's an interview transcript with timestamps. Generate a list of all major topics covered in the podcast, and the timestamp at which it's mentioned in the podcast. Use this format: - [00:00:00] Topic name. Here's the transcript: \n\n {transcript} {AI_PROMPT}",
    )
    
    print(chapters.completion)
    
    return chapters.completion

def create_show_notes(transcript):
    anthropic = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
        
    chapters = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=3000,
        prompt=f"{HUMAN_PROMPT} I'll give you an interview transcript; help me create a list of every company, person, project, or any other named entitiy that you find in it. Here's the transcript: \n\n {transcript} {AI_PROMPT}",
    )
    
    print(chapters.completion)
    
    return chapters.completion

def create_writeup(transcript):
    anthropic = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
        
    chapters = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=3000,
        prompt=f"{HUMAN_PROMPT} You're the writing assistant of a product manager. For each interview, we do a write up to recap the core ideas of the interview and further explore them. Write a list of key highlights we should further explore, such as core use cases, major customer or user problems, important user journeys or workflows, and key segmentation attributes. Here's the transcript: \n\n {transcript} {AI_PROMPT}",
    )
    
    print(chapters.completion)
    
    return chapters.completion

def title_suggestions(transcript):
    prompt = f"""
    These are some titles of previous podcast episodes we've published:

    1. "[Student] Interview with Justin Membreno and Chris Shaw (2023-09-25)"
    2. "[Tutor] Interview with Kaya Mendez and Bailey Lowenthal (2023-10-01)"
    3. "[School Counselor] Interview with Ken Norton and Grace Stevens (2023-08-11)"


    Here's a transcript of the interview; suggest 8 title options for it:
    
    {transcript}
    """
    
    try:
        result = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k", 
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        gpt_suggestions = result.choices[0].message.content
    except openai.error.InvalidRequestError as e:
        print(f"An error occurred: {e}")
        gpt_suggestions = "Out of context for GPT"
        
    claude_suggestions = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=3000,
        temperature=0.7,
        prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
    )

    claude_suggestions = claude_suggestions.completion

    suggestions = f"GPT-3.5 16k title suggestions:\n\n{gpt_suggestions}\n\nClaude's title suggestions:\n{claude_suggestions}\n"

    print(suggestions)

    return suggestions
    
def tweet_suggestions(transcript):
    prompt = f"""
    Here's a transcript of our latest podcast episode; suggest 8 tweets to share it on social medias.
    It should include a few bullet points of the most interesting topics. Our audience is technical.
    Use a writing style between Hemingway's and Flash Fiction. 
    
    {transcript}
    """
    
    try:
        result = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k", 
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        gpt_suggestions = result.choices[0].message.content
    except openai.error.InvalidRequestError as e:
        print(f"An error occurred: {e}")
        gpt_suggestions = "Out of context for GPT"

    anthropic = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
        
    claude_suggestions = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=3000,
        temperature=0.7,
        prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
    )

    claude_suggestions = claude_suggestions.completion

    suggestions = f"GPT-3.5 16k tweet suggestions:\n{gpt_suggestions}\n\nClaude's tweet suggestions:\n{claude_suggestions}\n"
    
    print(suggestions)
    
    return suggestions
    
def main():
    parser = argparse.ArgumentParser(description="Transcribe the podcast audio from a local file.")
    parser.add_argument("file_path", help="The local file path of the podcast to be processed.")
    parser.add_argument("name", help="The name of the output transcript file without extension.")
    args = parser.parse_args()

    file_path = args.file_path
    name = args.name
    raw_transcript_path = f"./podcasts-raw-transcripts/{name}.json"
    clean_transcript_path = f"./podcasts-clean-transcripts/{name}.md"
    results_file_path = f"./podcasts-results/{name}.md"

    print(f"Running smol-podcaster on {url}")
    
    # These are probably not the most elegant solutions, but they 
    # help with saving time since transcriptions are the same but we
    # might want to tweak the other prompts for better results.
    
    if not os.path.exists(raw_transcript_path):
        transcript = transcribe_audio(url, name)
    else:
        file = open(raw_transcript_path, "r").read()
        transcript = json.loads(file)['segments']
        
    if not os.path.exists(clean_transcript_path):
        transcript = process_transcript(transcript, name)
    else:
        transcript = open(clean_transcript_path, "r").read()
    
    chapters = create_chapters(transcript)
    show_notes = create_show_notes(transcript)
    title_suggestions_str = title_suggestions(transcript)
    tweet_suggestions_str = tweet_suggestions(transcript)

    with open(results_file_path, "w") as f:
        f.write("Chapters:\n")
        f.write(chapters)
        f.write("\n\n")
        f.write("Writeup:\n")
        f.write(create_writeup(transcript))
        f.write("\n\n")
        f.write("Show Notes:\n")
        f.write(show_notes)
        f.write("\n\n")
        f.write("Title Suggestions:\n")
        f.write(title_suggestions_str)
        f.write("\n\n")
        f.write("Tweet Suggestions:\n")
        f.write(tweet_suggestions_str)
        f.write("\n")
    
    print(f"Results written to {results_file_path}")
    

if __name__ == "__main__":
    main()
