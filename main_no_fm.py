from __future__ import annotations

import gradio as gr
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
import asyncio
import os

from tools import run_kubectl, run_helm, run_kubectl_impl, run_helm_impl
from instructions import k8s_helper_instructions, early_stop_validator_instructions
from interfaces import EarlyStopEvaluation

load_dotenv(override=True)
openai = AsyncOpenAI()

K8S_HELPER_MODEL_NAME = os.getenv("K8S_HELPER_MODEL_NAME")
EARLY_STOP_VALIDATOR_MODEL_NAME = os.getenv("EARLY_STOP_VALIDATOR_MODEL_NAME")
CONTEXT = os.getenv("CONTEXT")

DEFAULT_EARLY_STOPPING_REASONING = "By default"

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


async def handle_tool_call(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}", flush=True)
        tool = get_tool_by_name(tool_name)
        result = await tool(**arguments) if tool else {}
        results.append(
            {
                "role": "tool",
                "content": result.model_dump_json(),
                "tool_call_id": tool_call.id,
            }
        )
    return results


async def should_stop_early(question, tool_call):
    print("Checking for early stopping...", flush=True)
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


async def default_early_stop_evaluation():
    return EarlyStopEvaluation(should_stop=False, reasoning=DEFAULT_EARLY_STOPPING_REASONING)


async def chat_with_early_stop(message, history):
    """Main chat function that returns both response and early stop info"""
    messages = (
        [{"role": "system", "content": k8s_helper_instructions}]
        + history
        + [{"role": "user", "content": message}]
    )
    done = False
    step = 0
    early_stop_info = "**Early Stop Status**\n\nNo evaluation yet."

    while not done:
        step += 1

        response = await openai.chat.completions.create(
            model=K8S_HELPER_MODEL_NAME, messages=messages, tools=tools
        )

        if response.choices[0].finish_reason == "tool_calls":
            message_obj = response.choices[0].message
            tool_calls = message_obj.tool_calls

            early_stop_evaluator = default_early_stop_evaluation()

            if step == 1 and len(tool_calls) == 1:
                early_stop_evaluator = should_stop_early(
                    messages[len(messages) - 1]["content"], tool_calls[0]
                )

            results, early_stop_evaluation = await asyncio.gather(
                handle_tool_call(tool_calls), early_stop_evaluator
            )

            print(
                f"Early stopping: {early_stop_evaluation.should_stop}.\n"
                f"Reason: {early_stop_evaluation.reasoning}",
                flush=True,
            )

            # Update early stop information
            if early_stop_evaluation.reasoning != DEFAULT_EARLY_STOPPING_REASONING:
                early_stop_info = (
                    f"**Early Stop Status**\n\n"
                    f"**Should Stop:** {early_stop_evaluation.should_stop}\n\n"
                    f"**Reasoning:** {early_stop_evaluation.reasoning}"
                )

            if early_stop_evaluation.should_stop:
                stdout_content = json.loads(results[0]["content"])["stdout"]
                return f"```\n{stdout_content}\n```", early_stop_info
            else:
                messages.append(message_obj)
                messages.extend(results)
        else:
            done = True

    return response.choices[0].message.content, early_stop_info


async def chat_with_early_stop_streaming(message, history):
    """Streaming version that yields intermediate early stop updates"""
    messages = (
        [{"role": "system", "content": k8s_helper_instructions}]
        + history
        + [{"role": "user", "content": message}]
    )
    done = False
    step = 0
    early_stop_info = "**Early Stop Status**\n\nNo evaluation yet."
    processing_info = "Processing..."

    while not done:
        step += 1

        response = await openai.chat.completions.create(
            model=K8S_HELPER_MODEL_NAME, messages=messages, tools=tools
        )

        if response.choices[0].finish_reason == "tool_calls":
            message_obj = response.choices[0].message
            tool_calls = message_obj.tool_calls

            early_stop_evaluator = default_early_stop_evaluation()

            if step == 1 and len(tool_calls) == 1:
                early_stop_evaluator = should_stop_early(
                    messages[len(messages) - 1]["content"], tool_calls[0]
                )

            # Start tool execution
            tool_task = asyncio.create_task(handle_tool_call(tool_calls))

            # Wait for early stop evaluation and yield intermediate update
            processing_info = (
                f"Calling tools... {[tool.function.name for tool in tool_calls]}"
            )
            yield processing_info, early_stop_info
            results, early_stop_evaluation = await asyncio.gather(
                tool_task, early_stop_evaluator
            )

            print(
                f"Early stopping: {early_stop_evaluation.should_stop}.\n"
                f"Reason: {early_stop_evaluation.reasoning}",
                flush=True,
            )

            # Update early stop information and yield intermediate update
            if early_stop_evaluation.reasoning != DEFAULT_EARLY_STOPPING_REASONING:
                early_stop_info = (
                    f"**Early Stop Status**\n\n"
                    f"**Should Stop:** {early_stop_evaluation.should_stop}\n\n"
                    f"**Reasoning:** {early_stop_evaluation.reasoning}"
                )

            # Yield intermediate update (no response yet, just early stop info)
            yield processing_info, early_stop_info

            if early_stop_evaluation.should_stop:
                stdout_content = json.loads(results[0]["content"])["stdout"]
                final_response = f"```\n{stdout_content}\n```"
                yield final_response, early_stop_info
                return  # Exit the generator without a value
            else:
                messages.append(message_obj)
                messages.extend(results)
        else:
            done = True

    # Final response
    yield response.choices[0].message.content, early_stop_info


async def chat(message, history, early_stop_widget):
    """Wrapper for backward compatibility - not used in new interface"""
    response, early_stop_info = await chat_with_early_stop(message, history)
    yield response, gr.update(value=early_stop_info)


async def chat_wrapper(message, history, early_stop_widget):
    """Wrapper function to handle async chat with proper yielding"""
    async for chat_response, early_stop_update in chat(
        message, history, early_stop_widget
    ):
        yield chat_response, early_stop_update


def main():
    with gr.Blocks(title="K8s Helper") as interface:
        gr.Markdown("# Kubernetes Helper")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", height=500)
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Ask me about your Kubernetes cluster...",
                )
                clear = gr.Button("Clear")

            with gr.Column(scale=1):
                early_stop_status = gr.Markdown(
                    "**Early Stop Status**\n\nNo evaluation yet.",
                    label="Early Stop Evaluation",
                )

        async def respond(message, history):
            """Handle chat responses and update early stop status"""
            if not message:
                yield (history, "", "**Early Stop Status**\n\nNo evaluation yet.")
                return

            # Add user message to history immediately
            new_history = history + [{"role": "user", "content": message}]

            # Process the chat with real-time early stop updates
            async for response, early_stop_info in chat_with_early_stop_streaming(
                message, history
            ):
                if response is not None:
                    # Final response - add to history
                    final_history = new_history + [
                        {"role": "assistant", "content": response}
                    ]
                    yield final_history, "", early_stop_info
                else:
                    # Intermediate update - just update early stop status
                    yield new_history, "", early_stop_info

        msg.submit(respond, [msg, chatbot], [chatbot, msg, early_stop_status])

        clear.click(
            lambda: ([], "**Early Stop Status**\n\nNo evaluation yet."),
            outputs=[chatbot, early_stop_status],
        )

    interface.launch()


if __name__ == "__main__":
    main()
