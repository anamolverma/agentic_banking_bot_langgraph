from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import END , START , StateGraph
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import os,json
from typing import TypedDict
from langgraph.types import interrupt
from pydantic import BaseModel
from typing import Literal
from langchain_core.messages import SystemMessage ,HumanMessage,AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

from langgraph.types import Command

if os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

else:
    print("HUGGINGFACEHUB_API_TOKEN not set. Set the token")

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto", 
)


chat_model = ChatHuggingFace(llm=llm)

class TransferMoney(TypedDict):
    amount:int =None
    recipient:str =None
    approved : bool
    all_info : bool
    result: str
    messages:list[add_messages]
    llm_response: str


def transfer_money_tool(state:TransferMoney):
    print("\n💸 Executing Bank Transfer...")
    print(f"Transferred ₹{state['amount']} to {state['recipient']}")

    return {
        "result": f"₹{state['amount']} transferred successfully."
    }

class ApprovalResponse(BaseModel):
    approved: Literal["y", "n"]

def approved_node(state:TransferMoney):

    approved=interrupt({"action":"Wire transfer","value":state["amount"],"recipient":state["recipient"],"action_required": "y/n"})

    validated = ApprovalResponse(approved=approved.lower())

    return {
            "approved": validated.approved=="y"
        }


def execute_transfer(state:TransferMoney):

    if not state["approved"]:
        print("Transfer rejected")

        return {
            "result": "Transfer Cancelled"
        }

    
    return transfer_money_tool(state)


def llm_node(state: TransferMoney):

    response = chat_model.invoke([
        SystemMessage(content=f"""
        You are a banking transfer assistant.

        Required information:
        - amount
        - recipient

        Information already collected:
        amount: {state.get("amount")}
        recipient: {state.get("recipient")}

        Analyze the conversation.

        Extract any information that is already available.

        If information is missing, generate a natural question
        asking the user for the missing information.

        If all information is available, tell the user that
        you have all the required information.

        Never invent amount or recipient.

        Return ONLY valid JSON in this format:

        {{
            "amount": null,
            "recipient": null,
            "response": "your response to the user"
        }}
        """),
        *state["messages"]
    ])

    
    data = json.loads(response.content)
    print(data)
    amount = (
    data.get("amount")
    if data.get("amount") is not None
    else state.get("amount")
)

    recipient=(
    data.get("recipient")
    if data.get("recipient") is not None
    else state.get("recipient")
)
    
    return {
        "amount": amount,
        "recipient": recipient,
        "llm_response": data.get("response")
    }


def is_all_info_present(state):
    all_info=False
    if state["amount"] and state["recipient"]:
        all_info=True
    else:
        all_info=False

    state["all_info"]=all_info

    return {
        "all_info": all_info
    }


def ask_question(state):

    answer = interrupt(state["llm_response"])
    question = state["llm_response"]
    return {
        "messages": [
            AIMessage(content=question),
            HumanMessage(content=answer)
        ]
    }


def route(state):

    if state["all_info"]:
        return "approved_node"

    return "ask_question"

chain=StateGraph(TransferMoney)

chain.add_node(is_all_info_present)
chain.add_node(llm_node)
chain.add_node(execute_transfer)
chain.add_node(approved_node)
chain.add_node(transfer_money_tool)
chain.add_node(ask_question)

chain.add_edge(START,"llm_node")
chain.add_edge("ask_question", "llm_node")
chain.add_edge("llm_node","is_all_info_present")


chain.add_conditional_edges("is_all_info_present",route,{"approved_node":"approved_node","ask_question":"ask_question"})

chain.add_edge("approved_node","execute_transfer")

chain.add_edge("execute_transfer",END)


memory=MemorySaver()

config = {
    "configurable": {
        "thread_id": "bank-demo3x"
    }
}

compile=chain.compile(checkpointer=memory)

result=compile.invoke({
    "messages": [
        HumanMessage(content="Starting of new conversation..")
    ]
},config=config)

while "__interrupt__" in result:

    interrupt_data = result["__interrupt__"][0]

    print("\nAssistant:", interrupt_data.value)

    user_input = input("You: ")

    result = compile.invoke(
        Command(resume=user_input),
        config=config
    )

print(result)
