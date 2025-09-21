from agents import Agent, Runner
import os

from instructions import summary_keeper_instructions
from interfaces import SummaryResponse

SUMMARY_KEEPER_MODEL_NAME = os.getenv("SUMMARY_KEEPER_MODEL_NAME")

summary_keeper = Agent(
    name="summary-keeper",
    instructions=summary_keeper_instructions,
    model=SUMMARY_KEEPER_MODEL_NAME,
    output_type=SummaryResponse,
)


async def get_summary(previous_summary: str, messages: list[dict]) -> SummaryResponse:
    print("Updating summary...", flush=True)
    messages = [
        {
            "role": "user",
            "content": f"""
                Previous summary: {previous_summary}
                Latest messages: {messages}.
            """,
        },
    ]

    response = await Runner.run(summary_keeper, messages)
    return response.final_output
