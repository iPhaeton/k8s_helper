from __future__ import annotations

import gradio as gr
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI

from tools import run_kubectl, run_helm, run_kubectl_impl, run_helm_impl
from instructions import k8s_helper_instructions, early_stop_validator_instructions
from constants import K8S_HELPER_MODEL_NAME, EARLY_STOP_VALIDATOR_MODEL_NAME
from interfaces import EarlyStopEvaluation

load_dotenv(override=True)
openai = AsyncOpenAI()

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


async def should_stop_early(question, tool_call):
    messages = [
        {"role": "system", "content": early_stop_validator_instructions},
        {
            "role": "user",
            "content": f"""
                Here is the user's question: {question}"
                Here is the tool call that was made: {json.dumps(tool_call.model_dump())}.
                Is this enough to answer the user's question?
            """,
        },
    ]

    response = await openai.chat.completions.create(
        model=EARLY_STOP_VALIDATOR_MODEL_NAME, messages=messages, tools=tools
    )

    return EarlyStopEvaluation.model_validate_json(response.choices[0].message.content)


async def chat(message, history):
    messages = (
        [{"role": "system", "content": k8s_helper_instructions}]
        + history
        + [{"role": "user", "content": message}]
    )
    done = False
    early_stop = False

    step = 0

    while not done:
        step += 1

        response = await openai.chat.completions.create(
            model=K8S_HELPER_MODEL_NAME, messages=messages, tools=tools
        )

        if response.choices[0].finish_reason == "tool_calls":
            message = response.choices[0].message
            tool_calls = message.tool_calls

            early_stop_evaluation = EarlyStopEvaluation(
                should_stop=False, reasoning="By default"
            )

            if step == 1 and len(tool_calls) == 1:
                early_stop_evaluation = await should_stop_early(
                    messages[len(messages) - 1]["content"], tool_calls[0]
                )
                print(
                    f"Early stopping: {early_stop_evaluation.should_stop}.\nReason: {early_stop_evaluation.reasoning}",
                    flush=True,
                )
                if early_stop_evaluation.should_stop:
                    early_stop = True

            results = handle_tool_call(tool_calls)

            if early_stop:
                stdout_content = json.loads(results[0]["content"])["stdout"]
                return f"```\n{stdout_content}\n```"
            else:
                messages.append(message)
                messages.extend(results)
        else:
            done = True

    return response.choices[0].message.content


def main():
    gr.ChatInterface(chat, type="messages").launch()


if __name__ == "__main__":
    main()
