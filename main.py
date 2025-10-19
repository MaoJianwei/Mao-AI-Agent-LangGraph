import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TypedDict, Annotated

from IPython.core.display import Image
from IPython.core.display_functions import display
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.constants import START, END
from langgraph.func import task
from langgraph.graph import StateGraph, add_messages
from langgraph.runtime import Runtime
from langgraph.types import interrupt, Command

from lib.constant import NEED_ID_USER_CHAT_INPUT

from dotenv import load_dotenv


""" WebUI """
from fastapi import FastAPI, HTTPException
from starlette.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
import uvicorn

import asyncio
""" ========== """


load_dotenv()



class MaoContextSchema(TypedDict):
    llm: ChatOpenAI


class MaoTotalState(TypedDict):

    is_new_session: bool

    messages: Annotated[list[AnyMessage], add_messages]

    web_ui_prompt: str



global_area = {}
@asynccontextmanager
async def app_lifespan(app: FastAPI):

    print("Mao：开始启动")

    # global checkpointer, llm, graph

    # checkpointer
    global_area["checkpointer"] = InMemorySaver()

    # store


    # LLM
    global_area["llm"] = ChatOpenAI(
        model_name="glm-4.5-flash",  # 私有部署的模型名称
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
        openai_api_key=os.getenv("MAO_OPENAI_KEY"),
        # disable_streaming = True
    )

    # llm = init_chat_model(model="openai:glm-4.5-flash",
    #                       base_url="https://open.bigmodel.cn/api/paas/v4/",
    #                       api_key="27943078d7714b0a9e63c37a338f1051.GXJ9LixW6DWRTti6",
    #                       disable_streaming=False)


    graph_builder = StateGraph(MaoTotalState, MaoContextSchema)

    # graph_builder.add_node("session_list_show_and_choose", session_list_show_and_choose)
    # graph_builder.add_node("session_print_and_verify", session_print_and_verify)
    # graph_builder.add_node("simple_chat", simple_chat)
    #
    # graph_builder.add_edge(START, "session_list_show_and_choose")
    # graph_builder.add_edge("session_list_show_and_choose", "session_print_and_verify")
    # graph_builder.add_edge("session_print_and_verify", "simple_chat")
    # graph_builder.add_edge("simple_chat", "simple_chat")


    graph_builder.add_node("simple_webui_chat", simple_webui_chat)

    graph_builder.add_edge(START, "simple_webui_chat")
    graph_builder.add_edge("simple_webui_chat", END)

    global_area["graph"] = graph_builder.compile(checkpointer=global_area["checkpointer"])

    mermaid_code = global_area["graph"].get_graph().draw_mermaid()
    print(mermaid_code)

    yield

    print("Mao：关闭，释放全局变量")




def gen_session_id():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S_") + str(uuid.uuid4())

def session_list_show_and_choose(checkpointer: InMemorySaver):
    return {}

def session_print_and_verify(checkpointer: InMemorySaver):
    return {}




@task
def llm_inference(llm_server: ChatOpenAI, messages):
    # time.sleep(3)

    # response = llm_server.invoke(messages)
    # return response.content

    writer = get_stream_writer()

    ret = ""
    for chunk in llm_server.stream(messages):
        ret += chunk.text()
        writer(chunk.text())
        # yield chunk.text()
        # print(chunk.text(), end="")

    return ret

def simple_webui_chat(state: MaoTotalState, runtime: Runtime[MaoContextSchema]):

    human_msg = HumanMessage(content=state["web_ui_prompt"])

    state["messages"].append(human_msg)

    try:
        ai_content_task = llm_inference(runtime.context["llm"], state["messages"])
        ai_content = ai_content_task.result()
    except Exception as e:
        print(e)

    return {"messages": [human_msg, AIMessage(content=ai_content)], "is_new_session": False}





async def webui_agent_entry(user_message, session_id):

    config = {"configurable": {"thread_id": session_id}}
    context = {"llm": global_area["llm"]}

    run_data = {"is_new_session": True, "web_ui_prompt": user_message}


    stream_event_type = ["updates", "custom"] # "messages"（会依次输出：流式每个token、完整HumanMessage、完整AImessage，不好用，区分不开各种消息。）,
    for stream_mode, chunk in global_area["graph"].stream(run_data, stream_mode=stream_event_type, config=config, context=context): # "messages"
        # print(stream_mode, chunk)
        if stream_mode == "updates":
            print(f"*** stream_mode: {stream_mode}, chunk: {chunk}")
            for i in chunk.get("__interrupt__", ()):
                if i.value["need"] == NEED_ID_USER_CHAT_INPUT:
                    user_word = input(i.value["cmd_prompt"])
                    run_data = Command(resume=user_word)
        elif stream_mode == "messages":
            print(f"*** stream_mode: {stream_mode}, chunk: {chunk}")
            if len(chunk[0].content) > 20:
                pass
            else:
                # yield chunk[0].content
                pass
        elif stream_mode == "custom":
            print(f"*** stream_mode: {stream_mode}, chunk: {chunk}")
            yield {"type": "chat_text", "data": chunk}
        elif stream_mode == "values":
            print(f"*** stream_mode: {stream_mode}, chunk: {chunk}")
        else:
            print(f"*** stream_mode: {stream_mode}, chunk: {chunk}")

    print("Mao Finish")



""" WebUI """

app = FastAPI(title="Mao LLM Chat SSE Backend", lifespan=app_lifespan)


# # 模拟大模型生成（替换成你的真实模型调用逻辑）
# async def fake_llm_stream(prompt: str):
#     # 模拟模型逐步生成 token
#     response = "这是一个模拟的大模型回复，用于演示 SSE 流式传输。"
#     for i in range(len(response)):
#         yield response[i]
#         await asyncio.sleep(0.05)  # 模拟生成延迟

@app.post("/chat")
async def chat(request: dict):

    prompt = request.get("user-message")
    session_id_input = request.get("session-id")


    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")


    async def event_generator():
        session_id = session_id_input
        if not session_id:
            session_id = gen_session_id()
            yield {"data": {"type": "session_id_text", "data": session_id}}

        async for data in webui_agent_entry(prompt, session_id):
            # SSE 要求每条消息是 data: ... \n\n 格式
            yield {"data": data}
        # 可选：发送结束标记
        yield {"data": {"type": "done", "data": ""}}
        print("Mao 请求处理完成")

    return EventSourceResponse(event_generator(), media_type="text/event-stream")




# @app.post("/get-session-history")
# async def get_session_history(request: dict):
#
#     session_id_input = request.get("session-id")
#
#
#     if not prompt:
#         raise HTTPException(status_code=400, detail="Prompt is required")
#
#
#     async def event_generator():
#         session_id = session_id_input
#         if not session_id:
#             session_id = gen_session_id()
#             yield {"data": {"type": "session_id_text", "data": session_id}}
#
#         async for data in webui_agent_entry(prompt, session_id):
#             # SSE 要求每条消息是 data: ... \n\n 格式
#             yield {"data": data}
#         # 可选：发送结束标记
#         yield {"data": {"type": "done", "data": ""}}
#         print("Mao 请求处理完成")
#
#     return EventSourceResponse(event_generator(), media_type="text/event-stream")



app.mount("/", StaticFiles(directory="static", html=True), name="static")

""" ============================== """


# 在 Python 代码中启动 Uvicorn
if __name__ == "__main__":

    # 调用 uvicorn.run() 方法
    uvicorn.run(
        app="main:app",  # 应用入口：模块名:应用实例名（字符串格式）
        host="0.0.0.0",  # 监听地址（0.0.0.0 允许外部访问）
        port=7181,       # 端口
        reload=False,      # 开发模式：代码修改后自动重启（生产环境禁用）
        workers=12
    )
