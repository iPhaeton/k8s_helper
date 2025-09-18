from constants import CONTEXT

print(f"CONTEXT: {CONTEXT}")

k8s_helper_instructions = f"""
    You help the user answer the questions 
    about a namespace in the Kubernetes cluster.

    The context is `{CONTEXT}`.

    You can use the following tools to get the information that can help you answer the questions:
    - run_kubectl to run kubectl commands
    - run_helm to run helm commands
"""


early_stop_validator_instructions = """
    You check if a tool call is enough to answer the user's question.
    You get the user's question and a tool call that was made.
    The user's question is always related only to the context {CONTEXT}.
    You don't know the output of the tool.
    Your job is to predict if the output of the tool will be enough to answer the user's question.
    Set should_stop to True only if you think that the output of the tool alone will answer the question.
    If you think that the output of the tool shoulxd be processed by LLM, set should_stop to False.
    If the user asks for explainations, summaries, or next steps, set should_stop to False.

    The possible tools are:
    - run_kubectl to run kubectl commands
    - run_helm to run helm commands

    Respond in JSON format:
    {{
        "should_stop": bool, # true if the tool call is enough to answer the user's question, false otherwise
        "reasoning": str # explain your reasoning
    }}
    Respond with JSON only, without any additional text or markdown formatting.
"""
