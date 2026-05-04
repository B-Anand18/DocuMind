"""
youtube_service.py
------------------
Extract transcripts and metadata from YouTube videos.
"""

import re
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from langsmith import traceable


def extract_video_id(url: str) -> str:
    """
    Extract video ID from various YouTube URL formats.
    
    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    """
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    raise ValueError("Invalid YouTube URL format")


def get_video_metadata(video_id: str) -> dict:
    """
    Fetch video metadata using pytube.
    
    Returns:
        {"title": str, "duration": str, "author": str}
    """
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        duration_seconds = yt.length
        minutes = duration_seconds // 60
        seconds = duration_seconds % 60
        
        return {
            "title": yt.title,
            "duration": f"{minutes}:{seconds:02d}",
            "author": yt.author,
        }
    except Exception as e:
        # Fallback if pytube fails
        return {
            "title": f"YouTube Video {video_id}",
            "duration": "Unknown",
            "author": "Unknown",
        }


def get_transcript(video_id: str) -> list[dict]:
    """
    Fetch transcript using youtube-transcript-api.
    
    Returns:
        List of {"text": str, "start": float, "duration": float}
    """
    try:
        # Fetch transcript (tries English first)
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=['en'])
        return fetched.snippets
    except Exception as e:
        error_msg = str(e)
        
        # Check if no transcripts at all
        if "No transcripts were found" in error_msg or "Subtitles are disabled" in error_msg:
            # Try to get available languages
            try:
                transcript_list = api.list(video_id)
                available_langs = []
                
                for transcript in transcript_list:
                    lang_name = transcript.language
                    if transcript.is_generated:
                        available_langs.append(f"{lang_name} (auto-generated)")
                    else:
                        available_langs.append(lang_name)
                
                if available_langs:
                    langs_str = ", ".join(available_langs)
                    raise ValueError(
                        f"No English subtitles available for this video. "
                        f"Available languages: {langs_str}. "
                        f"Please try a video with English captions."
                    )
                else:
                    raise ValueError(
                        "No subtitles/transcripts available for this video. "
                        "Please try a different video with captions enabled."
                    )
            except ValueError:
                # Re-raise our custom error
                raise
            except:
                # If we can't get the list, show generic error
                raise ValueError(
                    "No subtitles/transcripts available for this video. "
                    "Please try a different video with captions enabled."
                )
        else:
            # Other errors
            raise ValueError(f"Could not retrieve transcript: {error_msg}")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS or HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_transcript_with_timestamps(transcript: list) -> list[dict]:
    """
    Format transcript segments with readable timestamps.
    
    Returns:
        List of {"timestamp": str, "text": str, "start_seconds": float}
    """
    formatted = []
    for segment in transcript:
        formatted.append({
            "timestamp": format_timestamp(segment.start),
            "text": segment.text,
            "start_seconds": segment.start,
        })
    return formatted


@traceable(name="extract_youtube_data")
def extract_youtube_data(url: str) -> dict:
    """
    Main function to extract all data from a YouTube video.
    
    Returns:
        {
            "text": str,              # Full transcript as text
            "metadata": dict,         # Video metadata
            "segments": list[dict],   # Timestamped segments
            "video_id": str,
            "url": str,
        }
    """
    video_id = extract_video_id(url)
    metadata = get_video_metadata(video_id)
    transcript = get_transcript(video_id)
    segments = format_transcript_with_timestamps(transcript)
    
    # Combine all text
    full_text = " ".join(seg["text"] for seg in segments)
    
    return {
        "text": full_text,
        "metadata": metadata,
        "segments": segments,
        "video_id": video_id,
        "url": url,
    }
