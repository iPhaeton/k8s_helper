from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from typing import Dict, Optional, Sequence, Union
from dotenv import load_dotenv
from agents import Agent, Runner, trace, function_tool
from pydantic import BaseModel
from typing import Any, List, Optional, Sequence, Union, Dict
import asyncio
import gradio as gr

load_dotenv(override=True)


# Strict output schema (no extra keys allowed)
class ExecutionResult(BaseModel):
    cmd: List[str]
    returncode: int
    stdout: str
    stderr: str
    json: Optional[Any] = None

    # Pydantic v2 style; for v1 use `class Config: extra = "forbid"`
    model_config = {"extra": "forbid"}


class EnvVar(BaseModel):
    name: str
    value: str
    model_config = {"extra": "forbid"}


@function_tool
def run_kubectl(
    args: Union[str, Sequence[str]],
    *,
    namespace: Optional[str] = None,
    context: Optional[str] = None,
    kubeconfig: Optional[str] = None,
    timeout: Optional[float] = 60.0,
    capture_output: bool = True,
    check: bool = False,
    env: Optional[List[EnvVar]] = None,
) -> ExecutionResult:
    """
    Run a kubectl command safely (no shell), with optional context/namespace/kubeconfig,
    timeout, and automatic JSON parsing if the output looks like JSON.

    Returns a dict:
      {
        "cmd": List[str],
        "returncode": int,
        "stdout": str,
        "stderr": str,
        "json": object | None
      }

    Raises:
      FileNotFoundError       - if kubectl isn't on PATH
      subprocess.TimeoutExpired
      subprocess.CalledProcessError (only when check=True and exit != 0)
    """
    # Ensure `kubectl` binary exists
    kubectl_path = shutil.which("kubectl")
    if not kubectl_path:
        raise FileNotFoundError("kubectl not found on PATH")

    # Normalize args safely
    if isinstance(args, str):
        arg_list = shlex.split(args)
    else:
        arg_list = list(args)

    # Build command
    cmd = [kubectl_path] + arg_list
    if namespace:
        cmd += ["--namespace", namespace]
    if context:
        cmd += ["--context", context]
    if kubeconfig:
        cmd += ["--kubeconfig", kubeconfig]

    run_kwargs = {
        "env": env,
        "timeout": timeout,
        "shell": False,
    }

    if capture_output:
        run_kwargs.update({"text": True, "capture_output": True})
    else:
        run_kwargs.update({"text": True})

    completed = subprocess.run(cmd, **run_kwargs)  # type: ignore[arg-type]

    stdout = completed.stdout if capture_output else ""
    stderr = completed.stderr if capture_output else ""

    parsed = None
    if capture_output and stdout:
        s = stdout.lstrip()
        if s.startswith("{") or s.startswith("["):
            try:
                parsed = json.loads(stdout)
            except json.JSONDecodeError:
                parsed = None

    if check and completed.returncode != 0:
        # Mirror subprocess.run(check=True) behavior using CalledProcessError
        err = subprocess.CalledProcessError(
            completed.returncode, cmd, output=stdout, stderr=stderr
        )
        raise err

    return ExecutionResult(
        cmd=cmd,
        returncode=completed.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
        json=parsed,
    )


@function_tool
def run_helm(
    args: Union[str, Sequence[str]],
    *,
    namespace: Optional[str] = None,  # --namespace
    context: Optional[str] = None,  # --kube-context
    kubeconfig: Optional[str] = None,  # --kubeconfig
    repo_config: Optional[
        str
    ] = None,  # --repository-config (e.g., path/to/repositories.yaml)
    registry_config: Optional[
        str
    ] = None,  # --registry-config (e.g., path/to/registry.json)
    timeout: Optional[float] = 60.0,  # seconds; None = no timeout
    capture_output: bool = True,
    check: bool = False,  # raise on non-zero exit
    env: Optional[List[EnvVar]] = None,  # extra env vars (e.g., HELM_*)
    workdir: Optional[str] = None,  # working directory for helm
    input_data: Optional[str] = None,  # pass stdin (e.g., values via '--values -')
) -> ExecutionResult:
    """
    Run a 'helm' command safely (no shell). Returns a dict:
      {
        "cmd": List[str],       # fully resolved command
        "returncode": int,
        "stdout": str,
        "stderr": str,
        "json": object | None   # parsed if output looks like JSON
      }

    Raises:
      FileNotFoundError           - if 'helm' is not on PATH
      subprocess.TimeoutExpired   - if execution exceeds 'timeout'
      subprocess.CalledProcessError (when check=True and exit != 0)
    """
    helm_path = shutil.which("helm")
    if not helm_path:
        raise FileNotFoundError("helm not found on PATH")

    # Normalize user args
    if isinstance(args, str):
        arg_list = shlex.split(args)
    else:
        arg_list = list(args)

    # Build command
    cmd = [helm_path] + arg_list
    if namespace:
        cmd += ["--namespace", namespace]
    if context:
        cmd += ["--kube-context", context]
    if kubeconfig:
        cmd += ["--kubeconfig", kubeconfig]
    if repo_config:
        cmd += ["--repository-config", repo_config]
    if registry_config:
        cmd += ["--registry-config", registry_config]

    run_kwargs = {
        "env": env,
        "cwd": workdir,
        "timeout": timeout,
        "shell": False,  # critical: prevent shell injection
        "text": True,  # treat stdin/stdout/stderr as text
        "input": input_data,  # None if not provided
    }

    if capture_output:
        run_kwargs.update({"capture_output": True})
    # else: inherit parent's stdio (live output), still text mode

    completed = subprocess.run(cmd, **run_kwargs)  # type: ignore[arg-type]

    stdout = completed.stdout if capture_output else ""
    stderr = completed.stderr if capture_output else ""

    parsed = None
    if capture_output and stdout:
        # Best-effort JSON detection/parse (works with 'helm list -o json', etc.)
        s = stdout.lstrip()
        if s.startswith("{") or s.startswith("["):
            try:
                parsed = json.loads(stdout)
            except json.JSONDecodeError:
                parsed = None

    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode, cmd, output=stdout, stderr=stderr
        )

    return ExecutionResult(
        cmd=cmd,
        returncode=completed.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
        json=parsed,
    )


k8s_helper_instructions = """
    You help the user answer the questions 
    about a namespace in the Kubernetes cluster.
    
    The namespace is `vvm-ci-pr-2019`.
    The context is `gke_argussec2_us-central1_dev-gke-soc`.

    You can use the following tools to get the information that can help you answer the questions:
    - run_kubectl to run kubectl commands
    - run_helm to run helm commands
"""

k8s_helper = Agent(
    name="k8s-helper",
    instructions=k8s_helper_instructions,
    tools=[run_kubectl, run_helm],
    model="gpt-5-mini",
)


async def chat(message, history):
    with trace("K8s helper"):
        messages = []
        for msg in history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        messages.append({"role": "user", "content": message})
        
        response = await Runner.run(k8s_helper, messages)

        return response.final_output


def main():
    gr.ChatInterface(chat, type="messages").launch()


if __name__ == "__main__":
    main()
