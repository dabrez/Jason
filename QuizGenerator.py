import abc
import google.generativeai.models
import google.generativeai.generative_models as model
import google.generativeai.types.generation_types as types
import google.generativeai.types.content_types as contentTypes
import google.generativeai
# import GeminiCallback
import asyncio
import threading
from dotenv import load_dotenv
import os

class QuizGenerator():

    def __init__(self, videoLink, selectedLanguage = "English", numberOfQuestions = 5):
        self.computerResponse = ""
        self.transcript = videoLink
        self.numberOfQuestions = numberOfQuestions #Default num of questions = 5
        config = types.GenerationConfig(response_mime_type="application/json")
        systemInstruction = f"You are my teacher. Based on this video transcript in {self.transcript}, focus on the part of the video within the timestamps provided on each request and provide a json with possible questions relating to it BASED ON THE EXAMPLE JSON I GIVE YOU. You may be asked to provide an explanation for a question or be asked to generate an entire quiz (more likely). For Multiple choice questions, you can mark as many answers as true. Do NOT generate free response questions. Also, make sure to use the exact same property names, but just change the contents/values of each property based on the context provided. Also make sure that all the information is true and taken purely from the video clip. Use GitHub Flavored Markdown (no HTML markdown or LateX is supported) whenever possible in the questions and answers, but replace all occurences of ``` with <`>. The user would like the quiz and future chat messages to be generated in " + (selectedLanguage)
        self.gemini_model = model.GenerativeModel("gemini-1.5-pro-002", generation_config=config, system_instruction=systemInstruction)
        self.chat = self.gemini_model.start_chat(history=[])

    async def sendMessage(self, timeStamps:list, generateQuiz:bool):
        quizPrompt = f"\n\nUse this JSON schema to generate {self.numberOfQuestions}" + " questions, and make sure to randomize the order of the options such that the correct answer is not always in the same place:\n\n"
        quizPrompt += "{" + "  \"quiz_title\": \"Sample Quiz\"," + "  \"questions\": [" + "    {" + "      \"type\": \"multiple_choice\"," 
        quizPrompt += "      \"question\": \"What is the capital of France?\", " + "      \"options\": ["
        quizPrompt += "        {\"text\": \"Paris\", \"correct\": true}," + "        {\"text\": \"London\",  \"correct\": false}," 
        quizPrompt += "        {\"text\": \"Berlin\", \"correct\": false}," + "        {\"text\": \"Rome\", \"correct\": false}" 
        quizPrompt += "      ]" + "    }," + "    {" 
        quizPrompt += "      \"type\": \"multiple_choice\","
        quizPrompt += "      \"question\": \"Which of the following are gas giants in our solar system?\", " 
        quizPrompt += "      \"options\": ["
        quizPrompt += "        {\"text\": \"Earth\", \"correct\": false},"
        quizPrompt += "        {\"text\": \"Saturn\", \"correct\": true},"
        quizPrompt += "        {\"text\": \"Jupiter\", \"correct\": true},"
        quizPrompt += "        {\"text\": \"Uranus\", \"correct\": false}"
        quizPrompt += "      ]"
        quizPrompt += "    },"
        quizPrompt += "    {"
        quizPrompt += "      \"type\": \"multiple_choice\","
        quizPrompt += "      \"question\": \"Which of the following is a color?\", "
        quizPrompt += "      \"options\": ["
        quizPrompt += "        {\"text\": \"Red\", \"correct\": false},"
        quizPrompt += "        {\"text\": \"Blue\", \"correct\": false},"
        quizPrompt += "        {\"text\": \"Yellow\", \"correct\": false},"
        quizPrompt += "        {\"text\": \"All of the above\", \"correct\": true}"
        quizPrompt += "      ]"
        quizPrompt += "    },"
        quizPrompt += "  ]"
        quizPrompt += "}"
        try:
            msgContent = [("Timestamps: " + str(timeStamps)), quizPrompt]
            result = []
            # thread = threading.Thread(target=lambda result:(result = self.chat.send_message_async), args=(msgContent)
            # thread.start()
            self.computerResponse = await self.chat.send_message_async(msgContent)
            # asyncio.wait_for(timeout=60.0)
           
            response = self.computerResponse
            
            return response
            
        except Exception as e:
            print(e)
            
    def chatMessage(self, messageContent, valueStorage:list):
        pass

# load_dotenv()            
# b = QuizGenerator("https://youtu.be/-MTRxRO5SRA")
# google.generativeai.configure(api_key="AIzaSyCHb1yl_9vM1C3_a9vJGROkfT_0iEcE0LM")
# asyncio.run(b.sendMessage(["0:00", "1:00"], True))
# b.sendMessage(["0:00", "1:00"], True)
        
        

    
