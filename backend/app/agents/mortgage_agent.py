from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from typing import List, Union, Any, Dict
from app.agents.callbacks.custom_callbacks import MyAgentCallback
from app.agents.api_wrappers.ask_human import AskHumanWrapper
from app.agents.api_wrappers.ask_human_2 import AskHumanWrapper, BasicAnswer
from fastapi import WebSocket
import re
from dotenv import load_dotenv
import os

load_dotenv()
serpapi_api_key = os.environ.get('SERPAPI_API_KEY')

RED = "\033[1;31m"
GREEN = "\033[0;32m"
BLUE = "\033[34m"
YELLOW = "\033[1;33m"
RESET = "\033[0m"

# Set up the base template
template = """You are a customer-facing chatbot for ING.  
You have access to the following tools:
{tools}

Use the following format:
Chat History: the conversation so far, answer the last user message
Thought: you should always think about what to do, consider the context and the chat history
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Use this context to enrich user's question and allow you to provide a good answer:
- User interest rate: 4%
- Available loan terms for the user:  ['10 years, '15 years', '20 years', '25 years', '30 years']


Let's begin!, continue the conversation:

Chat History: {chat_history}
{agent_scratchpad}"""

last_action_text = ""
last_observation = ""


class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""

        # kwargs["intermediate_steps"] is a list of tupples (action, observation)
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "

        # This is the equivalet to verbose = TRUE, printing in the console every observation
        if intermediate_steps:
            last_action, last_observation = intermediate_steps[-1]
            print(f"\nObservation: {BLUE}{last_observation}{RESET}\n")

        # Set the agent_scratchpad variable to that value
        # agent_scratchpad grows with every thought iteration
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])

        # Add chat history to kwargs
        kwargs["chat_history"] = kwargs.get("chat_history", "")

        print(self.template.format(**kwargs))

        # Add tools, tool_names and agent_scratchpad variables to the prompt template
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        print(f"\nThought: {YELLOW}{llm_output}{RESET}\n")

        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split(
                    "Final Answer:")[-1].strip()},
                log=llm_output,
            )

        # Check if agent asked a follow-up question
        if "Ask user for more info" in llm_output or "llm basic answer" in llm_output:

            return_values = {"output": llm_output.split(
                "Action: Ask user for more info\nAction Input:")[-1]}

            print(f"\nFinal Answer: {GREEN}{return_values}{RESET}\n")
            return AgentFinish(
                return_values=return_values,
                log=llm_output,
            )

        if "Observation:" in llm_output:
            print("Observation found")
            # return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


def create_agent(websocket: WebSocket):

    # Define which tools the agent can use to answer user queries
    wolfram = WolframAlphaAPIWrapper()
    ask_human = AskHumanWrapper()  # websocket
    basic_answer = BasicAnswer()

    tools = [
        Tool(
            name="Wolfram Alpha",
            func=wolfram.run,
            description="when you need to make a calculation to answer user's query"
        ),
        Tool(
            name="Mortgage and loan calculator",
            func=wolfram.run,
            description="when you need to calculate mortgages and loans"
        ),
        Tool(
            name="Ask user for more info",
            func=ask_human.run,
            description="when the user has given incomplete information to be able to provide an accurate response, the chatbot can ask the user for more info"
        ),
        Tool(
            name="llm basic answer",
            func=basic_answer.run,
            description="when the user question can be answered with the context provided"
        )
    ]

    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["chat_history", "intermediate_steps"],
    )

    #llm = OpenAI(temperature=0)
    llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
    )
    output_parser = CustomOutputParser()
    tool_names = [tool.name for tool in tools]

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )

    callback = MyAgentCallback(websocket)

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback,
        verbose=False,
        # return_intermediate_steps=True,
    )

    return agent_executor
