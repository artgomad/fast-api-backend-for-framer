from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from typing import List, Union, Any, Dict
from app.agents.callbacks.custom_callbacks import MyAgentCallback
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

# Define which tools the agent can use to answer user queries
search = SerpAPIWrapper()
# wolfram = WolframAlphaAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
    #,
    # Tool(
    #     name="Calculate",
    #     func=wolfram.run,
    #     description="useful for when you need to make mathematical operations"
    # )
]

# Set up the base template
template = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

Question: {input}
{agent_scratchpad}"""

last_action_text = ""
last_observation = ""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "

        #Get the last item of the intermediate steps
        if intermediate_steps:
            last_action, last_observation = intermediate_steps[-1]
            last_action_text = last_action.log
            print(f"\nThought: {YELLOW}{last_action_text}{RESET}\n")
            print(f"\nObservation: {BLUE}{last_observation}{RESET}\n")

        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        # print('agent_scratchpad = ' + kwargs["agent_scratchpad"])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):
    #def parse_with_prompt(self, completion: str, prompt: PromptValue) -> Any:

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split(
                    "Final Answer:")[-1].strip()},
                log=llm_output,
            )

        if "Observation:" in llm_output:
            print("Observation found")
            #return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)

output_parser = CustomOutputParser()

llm = OpenAI(temperature=0)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)
"""
callback = MyAgentCallback()

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=False, callback_manager=callback
)
"""
