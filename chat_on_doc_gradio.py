import gradio as gr
import time
from openai import OpenAI
import os,dotenv
import json
# for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
dotenv.load_dotenv()
api_key = os.environ.get("API_KEY")
client = OpenAI(api_key="sk-742ea5c51cc442b79bd478be1b1fa408", base_url="https://api.deepseek.com/beta")

original_messages = {"role": "system", "content": "You are a helpful assistant"}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气信息，用户需要先给出所需要查询的城市信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "要查询天气的城市，如广州、北京、上海等",
                    }
                },
                "required": ["location"]
            },
        }
    },
]

def get_weather(location):
    return f"{location}今天天气炎热"
def chat(message, history):
    llm_history = [original_messages]
    if len(history)>0:
        for item in history:
            ## 如果用户上传了文件，则会包含文件名和用户输入信息两个部分
            if item[1] is not None:
                llm_history.append({"role": "user", "content": item[0]})
                llm_history.append({"role": "assistant", "content": item[1]})
            else:
                llm_history.append({"role": "user", "content": "我上传了一个附件"})
                llm_history.append({"role": "assistant", "content": "您刚才上传了附件"})
    if message is not None:
        llm_history.append({"role": "user", "content": message})
        print(llm_history)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=llm_history,
            max_tokens=8192,
            temperature=0.7,
            tools=tools,
            stream=False
        )
        
        return response.choices[0].message
    return ""

def chat_with_files(message,history):
    global original_messages
    ret_message = ""
    num_files = len(message["files"])
    if num_files>0:
        ret_message += f"系统消息：你上传了 {num_files} 个附件\n"

        for i in range(num_files):
            print(message["files"][i])
            with open(message["files"][i]['path'], "r", encoding="utf-8") as file:
                content = file.read()
            template = f'''
请根据以下附件内容回答用户提出的问题：
## attachment
{content}
'''
            original_messages = {"role": "system", "content": template}
    answer = chat(message["text"],history)
    print(answer)
    if answer.content:
        answer_message = answer.content
        ret_message += answer_message
    if answer.tool_calls:
        tool = answer.tool_calls[0]
        tool_function_name = tool.function.name
        tool_query_string = json.loads(tool.function.arguments)
        if tool_function_name == 'get_weather':
            ret_message+=get_weather(tool_query_string["location"])
        
    return ret_message

demo = gr.ChatInterface(fn=chat_with_files, title="多功能问答助手", multimodal=True)

demo.launch()
