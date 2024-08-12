set url=https://youtube.com/watch?v=%1
yt-dlp.exe --cookies-from-browser chrome -f 140 --extract-audio %url% --output audio_files\%%(id)s.%%(ext)s
python transcript_with_senseVoice.py %url%
