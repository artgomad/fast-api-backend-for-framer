from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
from langchain.prompts import BaseChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser

from app.agents.api_wrappers.ask_human_2 import AskHumanWrapper, BasicAnswer
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

system_message = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

TOOLS:
------ 

Assistant has access to the following tools:"""


first_message_template = """
{tools}

To use a tool, please use the following format:
```
Thought: you should always think about what to do, consider the context and the chat history
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
```

And use the following context to inform your answer:
```
- User interest rate: 4%
- Available loan periods for the user:  ['10 years', '15 years', '20 years', '25 years', '30 years']
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""

last_action_text = ""
last_observation = ""


class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str: #This should return ChatPromptTemplate
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
        llm_prompt_input = [SystemMessage(content=system_message),HumanMessage(content=formatted)]

        print('FORMATED PROMPT AS RECEIVED BY THE LLM\n')
        print(llm_prompt_input)
        
        return [SystemMessage(content=system_message),HumanMessage(content=formatted)]



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
        if "Ask the user for more info" in llm_output or "llm basic answer" in llm_output:

            return_values = {"output": llm_output.split(
                "Action: Ask the user for more info\nAction Input:")[-1]}

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
    wolfram = WolframAlphaAPIWrapper()
    ask_human = AskHumanWrapper()
    basic_answer = BasicAnswer()

    tools = [
        Tool(
            name="Wolfram Alpha (Mortgage and loan calculator)",
            func=wolfram.run,
            description="when you need to calculate mortgages and loans"
        ),
        Tool(
            name="Ask the user for more info",
            func=ask_human.run,
            description="when the user has given incomplete information to be able to provide an accurate response, the chatbot can ask the user for more info"
        ),
        Tool(
            name="LLM basic answer",
            func=basic_answer.run,
            description="when the user question can be answered with the context provided"
        )
    ]

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
