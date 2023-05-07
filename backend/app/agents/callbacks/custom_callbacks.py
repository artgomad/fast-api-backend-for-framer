from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from typing import List, Union, Any, Dict
from fastapi import FastAPI, WebSocket
import asyncio

# Color codes
RED = "\033[1;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RESET = "\033[0m"


class MyAgentCallback(BaseCallbackManager):
    def __init__(self, websocket: WebSocket):
        super().__init__()
        self.websocket = websocket
        self.logs = ""

    async def async_on_agent_action(self, action: AgentAction) -> Any:
        self.logs += f"{action.log} \n\n"
        response_data = {
            'actions': self.logs,
            'data': ''
        }
        print(f"{GREEN}Send data on agent action to websocket{RESET}")
        await self.websocket.send_json(response_data)

    async def async_on_agent_finish(self, finish: AgentFinish) -> Any:
        if 'Final Answer:' in finish.log:
            final_answer = finish.log.split(
                "Final Answer:")[-1].strip()
            thoughts = finish.log.split(
                "Final Answer:")[0].strip()
        else:
            final_answer = finish.log.split(
                "Action Input:")[-1].strip()
            thoughts = finish.log.split(
                "Action Input:")[0].strip()

        response_data = {
            'actions': thoughts,
            'data': final_answer
        }
        print(f"{GREEN}Send data on agent finish to websocket: {RESET}" + final_answer)
        await self.websocket.send_json(response_data)

    def on_agent_action(self, action: AgentAction, **kwargs) -> Any:
        # asyncio is used to squedule an async function to run in the background
        asyncio.create_task(self.async_on_agent_action(action))

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> Any:
        # print('Final observation: ' +finish.log)
        # return {"last_observation": finish.log}
        asyncio.create_task(self.async_on_agent_finish(finish))

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> Any:
        print('tool_start')

    def on_tool_end(self, output: str, **kwargs) -> Any:
        print('on_tool_end')

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> Any:
        print('tool_error')

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> Any:
        print('llm_start')

    def on_llm_new_token(self, token: str, **kwargs) -> Any:
        print('on_llm_new_token')

    def on_llm_end(self, response: LLMResult, **kwargs) -> Any:
        print('on_llm_end')

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> Any:
        pass

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> Any:
        print('CHAIN STARTS')

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> Any:
        print('\nCHAIN ENDS\n')

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> Any:
        print('chain_error')

    def on_text(self, text: str, **kwargs) -> Any:
        print('on_text')

    def add_handler(self, callback: BaseCallbackHandler) -> None:
        pass

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        pass

    def set_handlers(self, handlers: List[BaseCallbackHandler]) -> None:
        pass


class MyAgentCallback_works(BaseCallbackManager):
    def __init__(self):
        self.last_action_text = ""
        self.last_observation = ""

    def on_agent_action(self, action: AgentAction, **kwargs) -> Any:
        # print('\nThought: ' + f"{YELLOW}{action.log}{RESET}")
        self.last_action_text = action.log
        return {"last_action_text": self.last_action_text}

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> Any:
        # print('Final observation: ' +finish.log)
        self.last_observation = finish.log
        return {"last_observation": self.last_observation}

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> Any:
        print('tool_start')

    def on_tool_end(self, output: str, **kwargs) -> Any:
        print('on_tool_end')

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> Any:
        print('tool_error')

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> Any:
        print('llm_start')

    def on_llm_new_token(self, token: str, **kwargs) -> Any:
        print('on_llm_new_token')

    def on_llm_end(self, response: LLMResult, **kwargs) -> Any:
        print('on_llm_end')

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> Any:
        pass

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> Any:
        print('CHAIN STARTS')

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> Any:
        print('\nCHAIN ENDS\n')

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> Any:
        print('chain_error')

    def on_text(self, text: str, **kwargs) -> Any:
        print('on_text')

    def add_handler(self, callback: BaseCallbackHandler) -> None:
        pass

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        pass

    def set_handlers(self, handlers: List[BaseCallbackHandler]) -> None:
        pass
