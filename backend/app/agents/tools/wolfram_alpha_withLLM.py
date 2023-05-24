"""Util that calls WolframAlpha."""
from typing import Any, Dict, Optional
from pydantic import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain

template = """
Translate the following user question into a sentence Wolfram Alpha can understand. 
Extract the necessary information from the context and conversation history.
Format your answer by just mentioning relevant variables, no additional text (e.g. loan [loan amount]EUR, [interest rate]%, [loan period]years). 
When lacking information imagine some reasonable numbers, defaulting to the minimum.

Examples:
(1)
CONTEXT: 
- User interest rate: 4%
- Available loan periods for the user:  10, 15, or 20 years
CONVERSATION HISTORY:
assistant: Hey, I'm Ben from ING what can I help you with?
user: How much money can I borrow?
assistant: Can you please provide more details about what you're looking for? Are you inquiring about a personal loan, mortgage or something else?
user: I actually want 10000 for buying a car
USER QUESTION: I actually want 10000 for buying a car
WOLFRAM INPUT: loan 10000 EUR, 4%, 10 years

(2)
CONTEXT: 
- User interest rate: 6%
- Available loan periods for the user:  10, 15, or 20 years
CONVERSATION HISTORY:
assistant: Hey, I'm Ben from ING what can I help you with?
user: I want a mortgage for my home, how much can I get?
USER QUESTION: I want a mortgage for my home, how much can I get?
WOLFRAM INPUT: loan 200000 EUR, 6%, 10 years

Begin!

CONTEXT:
{context}
CONVERSATION HISTORY:
{chatlog}
USER QUESTION:{user_input}
WOLFRAM INPUT:
"""


class WolframAlphaAPIWrapper(BaseModel):
    wolfram_client: Any  #: :meta private:
    wolfram_alpha_appid: Optional[str] = None
    context: str
    chatlog: str

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        wolfram_alpha_appid = get_from_dict_or_env(
            values, "wolfram_alpha_appid", "WOLFRAM_ALPHA_APPID"
        )
        values["wolfram_alpha_appid"] = wolfram_alpha_appid

        try:
            import wolframalpha

        except ImportError:
            raise ImportError(
                "wolframalpha is not installed. "
                "Please install it with `pip install wolframalpha`"
            )
        client = wolframalpha.Client(wolfram_alpha_appid)

        values["wolfram_client"] = client

        # TODO: Add error handling if keys are missing
        return values

    def run(self, query: str) -> str:
        """Run query through WolframAlpha and parse result."""
        # Identify necesary variables for a succesful wolframAPI call
        # MAYBE ADD AN LLM CALL HERE TO MAKE SURE THE QUERY IS VALID

        prompt = PromptTemplate(
            input_variables=["user_input", "context", "chatlog"],
            template=template,
        )

        # Declare a chain that will trigger an openAI completion with the given prompt
        llm_chain = LLMChain(
            llm=ChatOpenAI(temperature=0, model='gpt-3.5-turbo'),
            prompt=prompt,
        )

        wolfram_query = llm_chain.run(user_input=query, context=self.context, chatlog=self.chatlog)

        print('wolfram_query: ', wolfram_query)

        res = self.wolfram_client.query(wolfram_query)

        try:
            assumption = next(res.pods).text
            answer = next(res.results).text
        except StopIteration:
            return "Wolfram Alpha wasn't able to answer it"

        if answer is None or answer == "":
            # We don't want to return the assumption alone if answer is empty
            return "No good Wolfram Alpha Result was found"
        else:
            #return f"Assumption: {assumption} \nAnswer: {answer}"
            return f"{assumption} \n{answer}"
