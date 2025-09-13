from __future__ import annotations

import gradio as gr
import json
from dotenv import load_dotenv
from openai import OpenAI

from tools import run_kubectl, run_helm, run_kubectl_impl, run_helm_impl
from instructions import k8s_helper_instructions
from constants import MODEL_NAME

load_dotenv(override=True)
openai = OpenAI()

run_kubectl_json = {
    "name": run_kubectl.name,
    "description": run_kubectl.description,
    "parameters": run_kubectl.params_json_schema,
}

run_helm_json = {
    "name": run_helm.name,
    "description": run_helm.description,
    "parameters": run_helm.params_json_schema,
}

tools = [
    {"type": "function", "function": run_kubectl_json},
    {"type": "function", "function": run_helm_json},
]


def get_tool_by_name(name):
    if name == run_kubectl.name:
        return run_kubectl_impl
    if name == run_helm.name:
        return run_helm_impl
    raise ValueError(f"Tool {name} not found")


def handle_tool_call(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}", flush=True)
        tool = get_tool_by_name(tool_name)
        result = tool(**arguments) if tool else {}
        results.append(
            {
                "role": "tool",
                "content": result.model_dump_json(),
                "tool_call_id": tool_call.id,
            }
        )
    return results


def chat(message, history):
    messages = (
        [{"role": "system", "content": k8s_helper_instructions}]
        + history
        + [{"role": "user", "content": message}]
    )
    done = False

    while not done:
        response = openai.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=tools
        )

        print(response.choices[0].finish_reason, response.choices[0].message)

        if response.choices[0].finish_reason == "tool_calls":
            message = response.choices[0].message
            tool_calls = message.tool_calls
            results = handle_tool_call(tool_calls)
            messages.append(message)
            messages.extend(results)
        else:
            done = True
    return response.choices[0].message.content


def main():
    gr.ChatInterface(chat, type="messages").launch()


if __name__ == "__main__":
    main()
