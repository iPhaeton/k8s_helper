import json
import shlex
import shutil
import subprocess
from typing import Optional, Sequence, Union
from typing import List, Optional, Sequence, Union
from agents import function_tool

from interfaces import ExecutionResult, EnvVar


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
