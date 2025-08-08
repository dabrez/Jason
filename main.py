import numpy as np
from sklearn.metrics import silhouette_score
import requests
import re
from urllib.parse import urlparse, parse_qs
import nltk
from nltk.tokenize import sent_tokenize
import whisper
from pytube import YouTube
import os
import tempfile


from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from QuizGenerator import QuizGenerator

## get the video ID out of the\
class VideoTranscript:
    def __init__(self, link: str):
        self.link = link

    def check_youtube_link(self, link):
        try:
            # Send a GET request to the YouTube URL
            response = requests.get(link)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                return True  # The link is valid
            else:
                return False  # The link is invalid or inaccessible
        except requests.exceptions.RequestException as e:
            return False  # Handle exceptions like invalid URL or connection issues
        
    def videoSlice(self, link):
        """Returns video ID"""
    # Parse URL components
        parsed_url = urlparse(link)

    # If URL is youtu.be short link
        if parsed_url.netloc == 'youtu.be':
            return parsed_url.path[1:]

    # If URL is standard youtube.com
        if parsed_url.netloc in ('www.youtube.com', 'youtube.com'):
            query_params = parse_qs(parsed_url.query)
            if 'v' in query_params:
                return query_params['v'][0]
            elif parsed_url.path.startswith('/embed/'):
                return parsed_url.path.split('/')[2]
            elif parsed_url.path.startswith('/v/'):
                return parsed_url.path.split('/')[2]

    # Fallback to regex matching
        regex = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
        match = re.search(regex, link)   
        return match.group(1) if match else None


    def getVideoText(self, link):
        """Downloads audio from the YouTube video and transcribes it using
        OpenAI's Whisper model. Returns a list of segments with text,
        start time, and duration."""
        yt = YouTube(link)
        audio_stream = yt.streams.filter(only_audio=True).first()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            audio_path = tmp.name
        audio_stream.download(filename=audio_path)

        model = whisper.load_model("base")
        result = model.transcribe(audio_path)

        os.remove(audio_path)

        transcript = []
        for seg in result.get("segments", []):
            transcript.append({
                "text": seg["text"].strip(),
                "start": seg["start"],
                "duration": seg["end"] - seg["start"],
            })
        return transcript
        
    # def create_chapters(self, link):
    #     """Identifies the chapters/subsections of the video"""
    #     transcript = self.getVideoText(link)
    #     if not transcript:
    #         print("Transcript not available.")
    #         return None

    #     # Now, we'll use the time stamps to identify chapters.
    #     # For simplicity, let's assume each chapter is a new segment in the transcript.
    #     # For more advanced chaptering, you might use NLP techniques to group related segments.

    #     chapters = []
    #     for entry in transcript:
    #         start_time = entry['start']
    #         duration = entry['duration']
    #         text = entry['text']
    #         end_time = start_time + duration
    #         chapters.append({
    #             'start_time': start_time,
    #             'end_time': end_time,
    #             'text': text
    #         })
    #         print("chapters", chapters)

    #     return chapters    
    #     # print(YouTubeTranscriptApi.get_transcript("xzseFskewlE&t=393s"))
    
    def preprocess_text(self):
        """Preprocesses the transcript text."""
        if not self.transcript:
            self.getVideoText()
        self.texts = []
        self.time_stamps = []
        for entry in self.transcript:
            text = entry['text']
            # Clean text: remove special characters, convert to lowercase, etc.
            text = re.sub(r'[\W_]+', ' ', text).lower()
            self.texts.append(text)
            self.time_stamps.append((entry['start'], entry['start'] + entry['duration']))

    def compute_embeddings(self):
        """Computes embeddings for each transcript segment."""
        self.preprocess_text()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = model.encode(self.texts)

    def estimate_optimal_topics(self):
        """Estimates the optimal number of topics using the silhouette score."""
        max_topics = min(10, len(self.embeddings))
        scores = []
        K = range(2, max_topics)
        for k in K:
            clustering = AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='average')
            labels = clustering.fit_predict(self.embeddings)
            score = silhouette_score(self.embeddings, labels, metric='cosine')
            scores.append(score)
        optimal_k = K[np.argmax(scores)]
        return optimal_k

    def segment_topics(self, num_topics=None):
        """Segments the transcript into topics."""
        self.compute_embeddings()
        if num_topics is None:
            num_topics = self.estimate_optimal_topics()
        clustering = AgglomerativeClustering(n_clusters=num_topics, affinity='cosine', linkage='average')
        self.labels = clustering.fit_predict(self.embeddings)

    def create_chapters(self):
        """Creates chapters based on topic segmentation."""
        self.segment_topics()
        chapters = []
        current_chapter = {
            'start_time': self.time_stamps[0][0],
            'end_time': self.time_stamps[0][1],
            'text': self.texts[0],
            'label': self.labels[0]
        }

        for idx in range(1, len(self.labels)):
            if self.labels[idx] == current_chapter['label']:
                # Same topic, extend the chapter
                current_chapter['end_time'] = self.time_stamps[idx][1]
                current_chapter['text'] += ' ' + self.texts[idx]
            else:
                # Different topic, save current chapter and start a new one
                chapters.append(current_chapter)
                current_chapter = {
                    'start_time': self.time_stamps[idx][0],
                    'end_time': self.time_stamps[idx][1],
                    'text': self.texts[idx],
                    'label': self.labels[idx]
                }

        # Append the last chapter
        chapters.append(current_chapter)
        return chapters

    def generate_chapter_titles(self, chapters):
        """Generates titles for each chapter using TF-IDF keyword extraction."""
        titles = []
        for chapter in chapters:
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform([chapter['text']])
            indices = np.argsort(vectorizer.idf_)[::-1]
            features = vectorizer.get_feature_names_out()
            top_n = 3
            top_features = [features[i] for i in indices[:top_n]]
            title = ' '.join(top_features)
            titles.append(title)
        return titles

# Below is for TESTING ONLY. We will implement the real thing on the site.
if __name__ == "__main__":
    link = input("Enter your youtube link here: ")
    myObject = VideoTranscript(link)
    isValidLink = myObject.check_youtube_link(link)
    print(myObject.getVideoText(link))
    # quiz = QuizGenerator(link)
    # quiz.sendMessage(["0:00", "1:00"], True)
