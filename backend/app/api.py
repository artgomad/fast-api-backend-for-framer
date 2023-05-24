from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from utils.wiki_to_csv import recursively_find_all_pages, extract_sections
import openai
from dotenv import load_dotenv
import pandas as pd
import os
import re
import asyncio
import urllib.request
import requests
import pickle
import json
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.agents import AgentExecutor
# from app.agents.mortgage_agent import create_agent
# from app.agents.mortgage_agent_conversational import create_agent
from app.agents.mortgage_agent_conversational_test import create_agent
from app.agents.followup_question_agent import create_followup_question_agent
from app.agents.followup_question_agent_withInsights import create_followup_question_agent_withInsights
from app.agents.insight_gatherer_agent import create_insights_agent
from app.agents.callbacks.custom_callbacks import MyAgentCallback, MyAgentCallback_works
from app.agents.tools.save_pinecone import Pinecone
from app.agents.tools.create_atlas_map_nomic import NomicAtlas

app = FastAPI()

load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')
wolframalpha_app_key = os.environ.get('WOLFRAMALPHA_APP_ID')
serpapi_api_key = os.environ.get('SERPAPI_API_KEY')

origins = [
    "http://localhost:3000",
    "localhost:3000",
    "https://project-zszmhke1xyd6rlfxgg1i.framercanvas.com",  # Framer Canvas
    "https://comprehensive-value-405432.framer.app"  # A framer publised website
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

wolfram = WolframAlphaAPIWrapper()


@app.get("/", tags=["root"])
async def read_root() -> dict:
    return {"message": "Welcome to your todo list."}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # agent_executor = create_agent(websocket)

    while True:
        data = await websocket.receive_text()
        # Process the received data from the client
        chatlog = json.loads(data)['chatlog']
        chatlog_strings = ""

        # Format chatlog to be fed as agent memory
        for item in chatlog:
            chatlog_strings += item['role'] + ': ' + item['content'] + '\n'

        user_question = chatlog[-1]['content']

        # Added this new
        agent_executor = create_agent(
            websocket=websocket, chatlog=chatlog_strings)

        agent_output = await agent_executor.acall({'input': user_question, 'chat_history': chatlog_strings})

        print('agent_output = ')
        print(agent_output['output'])

        await websocket.send_json({
            "data":  agent_output['output'],
            # "intermediate_steps": agent_output['intermediate_steps'],
        })


@app.websocket("/ws-audio")
async def websocket_endpoint_audio(websocket: WebSocket):
    insightID = 0

    await websocket.accept()

    while True:
        # Receive the JSON payload
        payload_str = await websocket.receive_text()
        payload = json.loads(payload_str)

        research_questions = """
        - Do users understand all the steps?
        - What concepts presented on the screen confuse users?
        - Do users perceive they are getting a good deal?
        """
        context = "The user is looking at the coverage screen of a car insurance booking flow."
        generateInsights = payload['getInsights']
        generateFollowUps = not generateInsights

        question_number = payload['question_number']

        print(generateFollowUps)

        # Receive the binary audio data
        audio_data = await websocket.receive_bytes()
        # Process the received data from the client
        if audio_data:
            temporary_dir = os.path.abspath('data/temporary_files')

            with open(os.path.join(temporary_dir, "audio.webm"), "wb") as f:
                f.write(audio_data)
                print("Saved audio file to audio.webm")

            audio_file = open(os.path.join(temporary_dir, "audio.webm"), "rb")
            transcript = openai.Audio.transcribe("whisper-1", audio_file)

            pinecone_obj = Pinecone()

            if generateFollowUps == True:
                # GENERATE FOLLOW-UP QUESTIONS
                # Save first transcript localy
                with open(os.path.join(temporary_dir, "transcript.txt"), "w") as f:
                    f.write(transcript.text)
                    print(f"{transcript.text} saved")

                # Retreive insights from Pinecone that are relevant to the last transcript
                insights = []
                insights = pinecone_obj.retreive_embeddings(
                    query=transcript.text, top_results_num=3)

                llm_chain = create_followup_question_agent_withInsights()

                # Pass insights and transcript to generate follow-up questions
                chain_output = await llm_chain.arun({
                    'n_questions': 3,
                    'research_questions': research_questions,
                    'context': context,
                    'insights': insights,
                    'transcript': transcript.text})

                with open(os.path.join(temporary_dir, "followup_questions.txt"), "w") as f:
                    f.write(chain_output)
                    print('agent_output (follow-up)= ')
                    print(f"{chain_output} saved")

                # Send follow-up questions back to the client
                await websocket.send_json({"data": chain_output, "returningInsights": generateInsights})

            else:
                # SAVE ANSWERS TO FOLLOW-UP QUESTIONS
                with open(os.path.join(temporary_dir, f"answers_to_questions/answer_to_question{question_number}.txt"), "w") as f:
                    f.write(transcript.text)
                    print(f"{transcript.text} saved")

        else:
            print("Received invalid payload")


@app.websocket("/get-insights")
async def websocket_endpoint_insights(websocket: WebSocket):
    insightID = 0

    await websocket.accept()

    while True:
        # Receive the JSON payload
        payload_str = await websocket.receive_text()
        payload = json.loads(payload_str)

        research_questions = """
        - Do users understand all the steps?
        - What concepts presented on the screen confuse users?
        - Do users perceive they are getting a good deal?
        """
        context = "The user is looking at the coverage screen of a car insurance booking flow."
        getInsights = payload['getInsights']

        if getInsights:
            temporary_dir = os.path.abspath('data/temporary_files')

            pinecone_obj = Pinecone()

            # Get previous transcript and follow-up questions
            previous_transcript = ""
            followup_questions = {"question1": "",
                                  "question2": "", "question3": ""}
            answers = {"answer1": "", "answer2": "", "answer3": ""}

            with open(os.path.join(temporary_dir, "transcript.txt"), "r") as f:
                previous_transcript = f.read()

            with open(os.path.join(temporary_dir, "followup_questions.txt"), "r") as f:
                followup_questions_str = f.read()
                followup_questions_array = list(filter(lambda item: item != "",
                                                       followup_questions_str.split("\n")))
                for i, item in enumerate(followup_questions_array):
                    followup_questions[f"question{i+1}"] = item

            # Set the answers object with the answers saved locally
            for filename in os.listdir(os.path.join(temporary_dir, "answers_to_questions")):
                if filename.endswith(".txt"):
                    question_number = int(filename.split(
                        "answer_to_question")[1].split(".")[0])
                    with open(os.path.join(temporary_dir, "answers_to_questions", filename), "r") as f:
                        answers[f"answer{question_number}"] = f.read()

            print("previous_transcript = " + previous_transcript)
            print(followup_questions)
            print(answers)

            # GENERATE INSIGHTS
            llm_chain = create_insights_agent()

            # Generate insights from the given transcript, follow-up questions, and user answers
            chain_output = await llm_chain.arun(
                {'transcript': previous_transcript,
                 'research_questions': research_questions,
                 'context': context,
                 "followup_questions": followup_questions,
                 "user_answers": answers,
                 })

            print('agent_output (insights) = ')
            print(chain_output)

            # get insight count
            with open(os.path.join(temporary_dir, "insight_count.txt"), "r") as f:
                insightID = int(f.read()) + 1

            # Save each insight individually in Pinecone
            chain_output_arr = re.split(r'\n+', chain_output)
            for item in chain_output_arr:
                pinecone_obj.save(
                    data=item, insightID=insightID, context=context)
                insightID += 1

            with open(os.path.join(temporary_dir, "insight_count.txt"), "w") as f:
                f.write(str(insightID))

        else:
            print("Received invalid payload")

        # Send a response back to the client
        await websocket.send_json({"data": chain_output})


@app.websocket("/cluster-insights")
async def websocket_endpoint_cluster(websocket: WebSocket):

    await websocket.accept()

    # Receive the JSON payload
    payload_str = await websocket.receive_text()
    payload = json.loads(payload_str)

    clusters = int(payload['clusters'])
    print("clusters = ", clusters)

    map_obj = NomicAtlas()
    map_obj.visualise_database(n_clusters=clusters)

    # Send a response back to the client
    await websocket.send_json({"data": "Map updated"})


@app.post("/wiki", tags=["openAI"])
async def wiki_search():
    print("Starting wiki search...")
    pages = recursively_find_all_pages(["2020 Summer Olympics"])
    print("Pages length: ", len(pages))

    res = []
    for page in pages:
        res += extract_sections(page.content, page.title)
    df = pd.DataFrame(res, columns=["title", "heading", "content", "tokens"])
    df = df[df.tokens > 40]
    df = df.drop_duplicates(['title', 'heading'])
    df = df.reset_index().drop('index', axis=1)  # reset index
    df.head()

    df.to_csv('../data/olympics_sections.csv', index=False)

    return {
        "data": "Done."
    }


"""
                # PARALLEL EXECUTION
                llm_chain_noInsights = create_followup_question_agent()
                llm_chain_withInsights = create_followup_question_agent_withInsights()

                # Schedule the two API calls concurrently
                fetch_task1 = asyncio.ensure_future(llm_chain_noInsights.arun({
                    'n_questions': 1,
                    'research_questions': research_questions,
                    'context': context,
                    'transcript': transcript.text}))

                fetch_task2 = asyncio.ensure_future(llm_chain_withInsights.arun({
                    'n_questions': 2,
                    'research_questions': research_questions,
                    'context': context,
                    'insights': insights,
                    'transcript': transcript.text}))

                # Gather the results
                chain_output = await asyncio.gather(fetch_task1, fetch_task2)
                output_string = '\n'.join(chain_output)
"""
