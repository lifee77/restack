from datetime import timedelta
from typing import List
import json  # Ensure JSON serialization
from pydantic import BaseModel
from restack_ai.workflow import workflow, import_functions, log

with import_functions():
    from openai import pydantic_function_tool
    from src.functions.llm_chat import llm_chat, LlmChatInput, Message
    from src.functions.lookup_sales import lookupSales, LookupSalesInput
    from src.functions.recommend_products import recommend_products, RecommendationInput
    # from src.functions.new_function import new_function, FunctionInput, FunctionOutput

class MessageEvent(BaseModel):
    content: str

class EndEvent(BaseModel):
    end: bool

@workflow.defn()
class AgentChatToolFunctions:
    def __init__(self) -> None:
        self.finished = False
        self.messages: List[Message] = []

    @workflow.event
    async def message(self, message: MessageEvent) -> List[Message]:
        log.info(f"Received message: {message.content}")

        # Define which tools are available
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
            # pydantic_function_tool(
            #     model=FunctionInput,
            #     name=new_function.__name__,
            #     description="A function to talk to an ERP to get the latest sales data"
            # )
        ]

        # Your system instruction:
        system_content = (
            "You are a sales assistant with tool access. "
            "DO NOT answer any questions about prices, sales, or recommendations by yourself. "
            "ALWAYS use the provided tools to get data before responding to the user. "
            "If a user asks what they should buy or what is the cheapest item, call `lookupSales`. "
            "If the user has a budget and preferences, call `recommend_products` first. "
            "If a tool response is available, summarize it and present it to the user."
        )

        # 1) Insert system content FIRST so it's truly a system message
        self.messages.insert(
            0,
            Message(role="system", content=system_content)
        )

        # 2) Then append the user's message
        self.messages.append(
            Message(role="user", content=message.content or "")
        )

        log.info(f"Calling llm_chat with tools: {tools}")

        completion = await workflow.step(
            llm_chat,
            LlmChatInput(messages=self.messages, tools=tools),
            start_to_close_timeout=timedelta(seconds=120),
        )

        log.info(f"Completion response: {completion}")

        tool_calls = completion.choices[0].message.tool_calls
        # Assistant's response without any new tool calls
        self.messages.append(
            Message(
                role="assistant",
                content=completion.choices[0].message.content or "",
                tool_calls=tool_calls
            )
        )

        log.info(f"Tool Calls Received: {tool_calls}")

        # If the LLM decided to call any tool
        if tool_calls:
            responses = []

            for tool_call in tool_calls:
                log.info(f"Processing tool_call: {tool_call}")
                name = tool_call.function.name

                match name:
                    case lookupSales.__name__:
                        try:
                            args = LookupSalesInput.model_validate_json(tool_call.function.arguments)
                            log.info(f"Calling {name} with args: {args}")

                            result = await workflow.step(
                                lookupSales,
                                input=LookupSalesInput(category=args.category),
                                start_to_close_timeout=timedelta(seconds=120)
                            )

                            # Ensure JSON serialization
                            json_result = json.dumps(result.dict(), default=str)
                            log.info(f"lookupSales result: {json_result}")

                            responses.append(
                                Message(role="tool", tool_call_id=tool_call.id, content=json_result)
                            )

                        except Exception as e:
                            log.error(f"Error calling {name}: {e}")

                    case recommend_products.__name__:
                        try:
                            raw_args = tool_call.function.arguments
                            log.info(f"Raw recommend_products arguments: {raw_args}")

                            # Ensure JSON parsing is robust
                            if isinstance(raw_args, dict):
                                args = RecommendationInput.parse_obj(raw_args)
                            else:
                                args = RecommendationInput.model_validate_json(raw_args)

                        except Exception as e:
                            log.error(f"Failed to parse RecommendationInput JSON: {e}")
                            continue

                        log.info(f"Calling {name} with args: {args}")
                        result = await workflow.step(
                            recommend_products,
                            input=args,
                            start_to_close_timeout=timedelta(seconds=120)
                        )

                        json_result = json.dumps(result.dict(), default=str)
                        log.info(f"recommend_products result: {json_result}")

                        responses.append(
                            Message(role="tool", tool_call_id=tool_call.id, content=json_result)
                        )

                    # case new_function.__name__:
                    #     ...

            # Now append all tool responses before calling llm_chat again
            self.messages.extend(responses)

            # Request the LLM to generate a final answer based on tool results
            completion_with_tool_call = await workflow.step(
                llm_chat,
                LlmChatInput(messages=self.messages),  # no need to pass system content again
                start_to_close_timeout=timedelta(seconds=120)
            )

            self.messages.append(
                Message(
                    role="assistant",
                    content=completion_with_tool_call.choices[0].message.content or ""
                )
            )

        return self.messages

    @workflow.event
    async def end(self, end: EndEvent) -> EndEvent:
        log.info("Received end")
        self.finished = True
        return {"end": True}

    @workflow.run
    async def run(self, input: dict):
        await workflow.condition(lambda: self.finished)
        return
