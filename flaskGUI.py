from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
import asyncio
from QuizGenerator import QuizGenerator
from youtube_transcript_api import YouTubeTranscriptApi
import requests
LANG = "en"
VIDEOID = "IGlTScXzpNg"
url = f"http://video.google.com/timedtext?lang={LANG}&v={VIDEOID}"

@app.route('/')
def index():
    return render_template('index1.html')


@app.route('/process', methods=['POST'])
def process():
    response = requests.get(url)
    subtitles = ""
    if response.status_code == 200:
    # Get the content (typically XML for subtitles)
        subtitles = response.text
        # print(subtitles)
    else:
        print(f"Failed to retrieve subtitles. Status code: {response.status_code}")
    data = request.get_json() # retrieve the data sent from JavaScript
   
    # print(YouTubeTranscriptApi.get_transcript("IGlTScXzpNg"))

    
    result = data['value'] 
    quiz_generator = QuizGenerator(subtitles)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(quiz_generator.sendMessage(["0:00", "1:00"], True))
    print(result)
    return jsonify(result=result) # return the result to JavaScript

# async def newFunction(data):
#     quiz = QuizGenerator(data['value'])
#     await quiz.sendMessage(["0:00", "1:00"], True)