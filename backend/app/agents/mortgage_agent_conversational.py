from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
from langchain.prompts import BaseChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser

from app.agents.api_wrappers.ask_human import AskHumanWrapper, BasicAnswer
from app.agents.api_wrappers.wolfram_alpha import WolframAlphaAPIWrapper
from app.agents.callbacks.custom_callbacks import MyAgentCallback

from dotenv import load_dotenv
from fastapi import WebSocket
from typing import List, Union
import re
import os


load_dotenv()
serpapi_api_key = os.environ.get('SERPAPI_API_KEY')

RED = "\033[1;31m"
GREEN = "\033[0;32m"
BLUE = "\033[34m"
YELLOW = "\033[1;33m"
RESET = "\033[0m"

system_message = """You are Ben, a digital assistant from ING.
Ben is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations on topics related to ING.
Ben only answers based on the context the user provides or on the output of its tools.
Ben has access to the following tools:
TOOLS:
------
"""


first_message_template = """
{tools}

**To use a tool, ALWAYS use the following format:
```
Thought: you should always think about what to do, consider the context and the previous conversation history
Action: the action to take, should be EXACTLY one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: Answer the user's original question as Ben would do
```

**Use the following context to inform your answer:
- User interest rate: 4%
- Available loan periods for the user:  10, 15, 20, 25 and 30 years

**Previous conversation history:
{chat_history}

Begin!

{agent_scratchpad}"""

last_action_text = ""
last_observation = ""


class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str
    tools: List[Tool]

    # This should return ChatPromptTemplate
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
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

        # Set the agent_scratchpad variable to that value, it grows with every thought iteration
        kwargs["agent_scratchpad"] = thoughts

        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])

        # Add chat history to kwargs
        kwargs["input"] = kwargs.get("input", "")
        kwargs["chat_history"] = kwargs.get("chat_history", "")

        formatted = self.template.format(**kwargs)
        llm_prompt_input = [SystemMessage(
            content=system_message), HumanMessage(content=formatted)]

        print('FORMATED PROMPT AS RECEIVED BY THE LLM\n')
        print(llm_prompt_input)

        return [SystemMessage(content=system_message), HumanMessage(content=formatted)]


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

        # Check if agent asked a follow-up question or give a simple answer
        if "Ask user" in llm_output or "Simple answer" in llm_output:

            return_values = {"output": llm_output.split(
                "\nAction Input:")[-1]}

            print(f"\nFinal Answer: {GREEN}{return_values}{RESET}\n")
            return AgentFinish(
                return_values=return_values,
                log=llm_output,
            )

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
    # wolfram = WolframAlphaAPIWrapper()
    ask_human = AskHumanWrapper()
    basic_answer = BasicAnswer()

    tools = [
        # Tool(
        #     name="Calculate loan",
        #     func=wolfram.run,
        #     description="""
        #     A loan calculator from Wolfram Alpha. Select this tool when you have enough information to make loan calculations.
        #     Action Input should be in the following format: loan {loan amount}, {interest rate}, {loan period}.
        #     """
        # ),
        Tool(
            name="Ask user",
            func=ask_human.run,
            description="""
            Select this tool if you need more information from the user to provide an answer or to use another tool.
            """
        )
    ]

    """
        Tool(
            name="Simple answer",
            func=basic_answer.run,
            description="Select this tool ONLY if there is enough information in the text below to give an answer to user's question."
        )
        """

    # Declare a dynamic prompt formatting class that adds tools, input, chat_history and agent_scratchpad
    # when its format_messages() method is called
    prompt = CustomPromptTemplate(
        tools=tools,
        template=first_message_template,
        input_variables=["input", "chat_history", "intermediate_steps"],
    )

    llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')

    # Declare a chain that will trigger an openAI completion with the given prompt
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
    )

    # Declare a parser that will take the output of the LLMChain (until the stop keyword)
    # and trigger an AgentAction (with a tool and tool input) or AgentFinish
    output_parser = CustomOutputParser()

    tool_names = [tool.name for tool in tools]

    # Declare an agent chain that when executed will sequence the LLMChain and the CustomOutputParser
    # The stop porperty indicates when the LLMChain should stop and the CustomOutputParser should start
    agent_chain = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )

    callback = MyAgentCallback(websocket)

    # Execute the agent_chain sequece
    """
     The AgentExecutor can largely be thought of as a loop that:
        - Passes user input and any previous steps to the Agent (in this case, agent_chain)
        - If the Agent returns an AgentFinish, then return that directly to the user
        - If the Agent returns an AgentAction, it sends an action and action to LLMChain, returning an Observation
        - Repeat, passing the AgentAction and Observation back to the Agent until an AgentFinish is emitted.
    """
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent_chain,
        tools=tools,
        callback_manager=callback,
        verbose=False,
        # return_intermediate_steps=True,
    )

    return agent_executor
