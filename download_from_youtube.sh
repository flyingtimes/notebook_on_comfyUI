#!/bin/bash
# Usage:
# download_from_youtube.sh 'URL'

# Notice:
# First,login youtube account in chrome, and then use following command to get the cookie file beforehand.
#yt-dlp --cookies-from-browser chrome --cookies cookie-youtube.txt -F $1
yt-dlp --cookies-from-browser chrome -f 'ba' -x --audio-format mp3 $1 -o 'audio_files/%(id)s.%(ext)s' 
python transcript_with_senseVoice.py $1
