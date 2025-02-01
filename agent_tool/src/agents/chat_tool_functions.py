from datetime import timedelta
from typing import List
from pydantic import BaseModel
from restack_ai.workflow import workflow, import_functions, log

with import_functions():
    from openai import pydantic_function_tool
    from src.functions.llm_chat import llm_chat, LlmChatInput, Message
    from src.functions.lookup_sales import lookupSales, LookupSalesInput
    from src.functions.recommend_products import recommend_products, RecommendationInput
    # Step 2: Import your new function to the agent
    # from src.functions.new_function import new_function, FunctionInput, FunctionOutput

class MessageEvent(BaseModel):
    content: str

class EndEvent(BaseModel):
    end: bool

@workflow.defn()
class AgentChatToolFunctions:
    def __init__(self) -> None:
        self.end = False
        self.messages = []

    @workflow.event
    async def message(self, message: MessageEvent) -> List[Message]:
        log.info(f"Received message: {message.content}")

        tools = [
            pydantic_function_tool(
                model=LookupSalesInput,
                name=lookupSales.__name__,
                description="Lookup sales for a given category"
            ),
            pydantic_function_tool(
                model=RecommendationInput,
                name=recommend_products.__name__,
                description="Recommend products based on user preferences"
            ),
            # Step 3 Add your new function to the tools list and adjust the system prompt
            
            # pydantic_function_tool(
            #     model=FunctionInput,
            #     name=new_function.__name__,
            #     description="A function to talk to an ERP to get the latest sales data"
            # )
        ]
        
        # Change the system prompt to the agent
        system_content = "You are a helpful assistant that can help with sales data."

        self.messages.append(Message(role="user", content=message.content or ""))
        completion = await workflow.step(
            llm_chat,
            LlmChatInput(messages=self.messages, tools=tools, system_content=system_content),
            start_to_close_timeout=timedelta(seconds=120)
        )

        log.info(f"completion: {completion}")
        
        tool_calls = completion.choices[0].message.tool_calls
        self.messages.append(Message(role="assistant", content=completion.choices[0].message.content or "", tool_calls=tool_calls))

        log.info(f"tool_calls: {tool_calls}")

        # Ensure responses for all tool calls
        tool_call_responses = []

        if tool_calls:
            for tool_call in tool_calls:
                log.info(f"Processing tool_call: {tool_call}")

                name = tool_call.function.name

                match name:
                    case lookupSales.__name__:
                        args = LookupSalesInput.model_validate_json(tool_call.function.arguments)
                        log.info(f"Calling {name} with args: {args}")

                        result = await workflow.step(
                            lookupSales,
                            input=LookupSalesInput(category=args.category),
                            start_to_close_timeout=timedelta(seconds=120)
                        )
                        tool_call_responses.append(
                            Message(role="tool", tool_call_id=tool_call.id, content=str(result))
                        )

                    case recommend_products.__name__:
                        args = RecommendationInput.model_validate_json(tool_call.function.arguments)
                        log.info(f"Calling {name} with args: {args}")

                        result = await workflow.step(
                            recommend_products,
                            input=args,
                            start_to_close_timeout=timedelta(seconds=120)
                        )
                        tool_call_responses.append(
                            Message(role="tool", tool_call_id=tool_call.id, content=str(result))
                        )

            # Append all tool responses before proceeding to the final LLM call
            self.messages.extend(tool_call_responses)

            completion_with_tool_call = await workflow.step(
                llm_chat,
                LlmChatInput(messages=self.messages, system_content=system_content),
                start_to_close_timeout=timedelta(seconds=120)
            )
            self.messages.append(Message(role="assistant", content=completion_with_tool_call.choices[0].message.content or ""))

        return self.messages
    
    @workflow.event
    async def end(self, end: EndEvent) -> EndEvent:
        log.info(f"Received end")
        self.end = True
        return {"end": True}
  
    @workflow.run
    async def run(self, input: dict):
        await workflow.condition(lambda: self.end)
        return
