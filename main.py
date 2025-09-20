from __future__ import annotations

from dotenv import load_dotenv
from agents import Agent, Runner, trace
import gradio as gr
import os

from tools import run_kubectl, run_helm
from instructions import k8s_helper_instructions

K8S_HELPER_MODEL_NAME = os.getenv("K8S_HELPER_MODEL_NAME")

load_dotenv(override=True)


k8s_helper = Agent(
    name="k8s-helper",
    instructions=k8s_helper_instructions,
    tools=[run_kubectl, run_helm],
    model=K8S_HELPER_MODEL_NAME,
)


async def chat(message, history):
    with trace("K8s helper"):
        messages = []
        for msg in history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": message})

        response = await Runner.run(k8s_helper, messages)

        return response.final_output


def main():
    gr.ChatInterface(chat, type="messages").launch()


if __name__ == "__main__":
    main()
