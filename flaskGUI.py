from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
import asyncio
from QuizGenerator import QuizGenerator
from main import VideoTranscript

@app.route('/')
def index():
    return render_template('index1.html')


@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()  # retrieve the data sent from JavaScript
    link = data['value']
    segments = VideoTranscript(link).getVideoText(link)
    subtitles = " ".join([seg['text'] for seg in segments])
    quiz_generator = QuizGenerator(subtitles)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(quiz_generator.sendMessage(["0:00", "1:00"], True))
    print(result)
    return jsonify(result=result) # return the result to JavaScript

# async def newFunction(data):
#     quiz = QuizGenerator(data['value'])
#     await quiz.sendMessage(["0:00", "1:00"], True)