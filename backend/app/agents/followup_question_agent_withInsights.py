from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import BaseChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from dotenv import load_dotenv


load_dotenv()

RED = "\033[1;31m"
GREEN = "\033[0;32m"
BLUE = "\033[34m"
YELLOW = "\033[1;33m"
RESET = "\033[0m"

user_message = """
{transcript}
"""

system_message = """
You are a qualitative researcher conducting a user test in the following context:
{context}

As a qualitative user researcher, your goal is to find answers to the research questions by asking pertinent follow-up questions.
The following guidelines may help you ask more effective questions:
- Prioritise the research objectives
- Ask open-ended questions
- Avoid leading questions
- Ask clear and easy to understand questions
- Probe for elaboration

The research questions of the study are:
{research_questions}

These are some relevant insights you have collected from other sessions:
{insights}

Come up with {n_questions} follow-up question(s) to ask the user based on the transcript provided. The questions should address different topics to maximize insight generation.
"""

#POTENTIAL ADDITION:
"""
As a qualitative user researcher, your goal is to find answers to the research questions by asking pertinent follow-up questions.
The following guidelines may help you ask more effective questions:
- Prioritise the research objectives
- Ask open-ended questions
- Avoid leading questions
- Ask clear and easy to understand questions
- Probe for elaboration
"""


class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str

    # This should return ChatPromptTemplate
    def format_messages(self, **kwargs) -> str:
        # Add input_variables to kwargs
        kwargs["n_questions"] = kwargs.get("n_questions", "")
        kwargs["research_questions"] = kwargs.get("research_questions", "")
        kwargs["context"] = kwargs.get("context", "")
        kwargs["insights"] = kwargs.get("insights", "")
        kwargs["transcript"] = kwargs.get("transcript", "")

        formatted_system_message = system_message.format(**kwargs)
        formatted_user_message = user_message.format(**kwargs)

        llm_prompt_input = [SystemMessage(
            content=formatted_system_message), HumanMessage(content=formatted_user_message)]

        print('FORMATED PROMPT AS RECEIVED BY THE LLM\n')
        print(llm_prompt_input)

        return llm_prompt_input


def create_followup_question_agent_withInsights():
    # Declare a dynamic prompt formatting class that adds tools, input, chat_history
    prompt = CustomPromptTemplate(
        template=system_message,
        input_variables=[
            "n_questions",
            "research_questions",
            "context",
            "insights",
            "transcript"],
    )

    llm = ChatOpenAI(temperature=0.7, model='gpt-4')

    # Declare a chain that will trigger an openAI completion with the given prompt
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
    )

    return llm_chain
