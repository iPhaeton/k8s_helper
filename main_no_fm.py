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
from summary_keeper import get_summary

load_dotenv(override=True)

K8S_HELPER_MODEL_NAME = os.getenv("K8S_HELPER_MODEL_NAME")
K8S_HELPER_BASE_URL = os.getenv("K8S_HELPER_BASE_URL")
K8S_HELPER_API_KEY = os.getenv("K8S_HELPER_API_KEY")

EARLY_STOP_VALIDATOR_MODEL_NAME = os.getenv("EARLY_STOP_VALIDATOR_MODEL_NAME")
EARLY_STOP_VALIDATOR_BASE_URL = os.getenv("EARLY_STOP_VALIDATOR_BASE_URL")
EARLY_STOP_VALIDATOR_API_KEY = os.getenv("EARLY_STOP_VALIDATOR_API_KEY")

CONTEXT = os.getenv("CONTEXT")

DEFAULT_EARLY_STOPPING_REASONING = "By default"

k8s_helper_openai = AsyncOpenAI(
    api_key=K8S_HELPER_API_KEY, base_url=K8S_HELPER_BASE_URL
)
early_stop_validator_openai = AsyncOpenAI(
    api_key=EARLY_STOP_VALIDATOR_API_KEY, base_url=EARLY_STOP_VALIDATOR_BASE_URL
)

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


async def should_stop_early(summary, question, tool_call):
    print("Checking for early stopping...", flush=True)
    messages = [
        {"role": "system", "content": early_stop_validator_instructions},
        {
            "role": "user",
            "content": f"""
                Here is the user's question: {question}"
                Here is a short summary of the previous conversation: {summary}
                Here is the tool call that was made: {json.dumps(tool_call.model_dump())}.
                Is this enough to answer the user's question?
            """,
        },
    ]

    response = await early_stop_validator_openai.chat.completions.create(
        model=EARLY_STOP_VALIDATOR_MODEL_NAME, messages=messages, tools=tools
    )

    return EarlyStopEvaluation.model_validate_json(response.choices[0].message.content)


async def default_early_stop_evaluation():
    return EarlyStopEvaluation(
        should_stop=False, reasoning=DEFAULT_EARLY_STOPPING_REASONING
    )


async def chat_with_early_stop_streaming(message, history, current_summary):
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

    new_summary_promise = asyncio.create_task(get_summary(current_summary, messages))

    while not done:
        step += 1

        response = await k8s_helper_openai.chat.completions.create(
            model=K8S_HELPER_MODEL_NAME, messages=messages, tools=tools
        )

        if response.choices[0].finish_reason == "tool_calls":
            message_obj = response.choices[0].message
            tool_calls = message_obj.tool_calls

            early_stop_evaluator = default_early_stop_evaluation()

            if step == 1 and len(tool_calls) == 1:
                early_stop_evaluator = should_stop_early(
                    current_summary,
                    messages[len(messages) - 1]["content"],
                    tool_calls[0],
                )

            # Start tool execution
            tool_task = asyncio.create_task(handle_tool_call(tool_calls))

            # Wait for early stop evaluation and yield intermediate update
            processing_info = f"""
                Calling tools... 
                {'\n'.join([f'Running tool "{tool.function.name}" with args: {json.loads(tool.function.arguments)['args'][:100]}' for tool in tool_calls])}
            """
            yield processing_info, early_stop_info, current_summary
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
            yield processing_info, early_stop_info, current_summary

            if early_stop_evaluation.should_stop:
                stdout_content = json.loads(results[0]["content"])["stdout"]
                final_response = f"```\n{stdout_content}\n```"
                new_summary = await new_summary_promise
                yield final_response, early_stop_info, new_summary.summary
                return  # Exit the generator without a value
            else:
                messages.append(message_obj)
                messages.extend(results)
        else:
            done = True

    new_summary = await new_summary_promise

    # Final response
    yield response.choices[0].message.content, early_stop_info, new_summary.summary


def main():
    with gr.Blocks(title="K8s Helper") as interface:
        gr.Markdown("# Kubernetes Helper")

        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot(type="messages", height="75vh")
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Ask me about your Kubernetes cluster...",
                )

            with gr.Column(scale=1, min_width="10px"):
                early_stop_status = gr.Markdown(
                    "**Early Stop Status**\n\nNo evaluation yet.",
                    label="Early Stop Evaluation",
                )
                summary_display = gr.Markdown(
                    "**Current Summary**\n\nNo summary yet.",
                    label="Summary",
                )

        global current_summary, current_early_stop_info
        current_summary = ""
        current_early_stop_info = "**Early Stop Status**\n\nNo evaluation yet."

        async def respond(message, history):
            """Handle chat responses and update early stop status"""
            global current_summary, current_early_stop_info

            if not message:
                yield (
                    history,
                    "",
                    "**Early Stop Status**\n\nNo evaluation yet.",
                    "**Current Summary**\n\nNo summary yet.",
                )
                return

            # Add user message to history immediately
            new_history = history + [{"role": "user", "content": message}]

            # Process the chat with real-time early stop updates
            async for (
                response,
                early_stop_info,
                new_summary,
            ) in chat_with_early_stop_streaming(message, history, current_summary):
                current_summary = new_summary
                current_early_stop_info = early_stop_info
                summary_text = (
                    f"**Current Summary**\n\n{current_summary}"
                    if current_summary
                    else "**Current Summary**\n\nNo summary yet."
                )

                if response is not None:
                    # Final response - add to history
                    final_history = new_history + [
                        {"role": "assistant", "content": response}
                    ]
                    yield final_history, "", early_stop_info, summary_text
                else:
                    # Intermediate update - just update early stop status
                    yield new_history, "", early_stop_info, summary_text

        msg.submit(
            respond, [msg, chatbot], [chatbot, msg, early_stop_status, summary_display]
        )

        def handle_chatbot_clear(history):
            """Handle when the chatbot is cleared via built-in clear button"""
            global current_summary, current_early_stop_info
            if not history:  # If history is empty, chatbot was cleared
                current_summary = ""
                current_early_stop_info = "**Early Stop Status**\n\nNo evaluation yet."
                return (
                    "**Early Stop Status**\n\nNo evaluation yet.",
                    "**Current Summary**\n\nNo summary yet.",
                )
            # If history is not empty, return the current tracked values
            current_summary_text = (
                f"**Current Summary**\n\n{current_summary}"
                if current_summary
                else "**Current Summary**\n\nNo summary yet."
            )
            return (current_early_stop_info, current_summary_text)

        # Handle when chatbot is cleared using built-in clear button
        chatbot.change(
            handle_chatbot_clear,
            inputs=[chatbot],
            outputs=[early_stop_status, summary_display],
        )

    interface.launch()


if __name__ == "__main__":
    main()
