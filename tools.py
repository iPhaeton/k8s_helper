import asyncio
import json
import shlex
import shutil
import subprocess
from typing import List, Optional, Sequence, Union
from agents import function_tool

from interfaces import ExecutionResult, EnvVar


@function_tool
async def run_kubectl(
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
    return await run_kubectl_impl(
        args,
        namespace=namespace,
        context=context,
        kubeconfig=kubeconfig,
        timeout=timeout,
        capture_output=capture_output,
        check=check,
        env=env,
    )


async def run_kubectl_impl(
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

    if capture_output:
        # Use asyncio subprocess for async execution
        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            stdout = stdout_bytes.decode('utf-8') if stdout_bytes else ""
            stderr = stderr_bytes.decode('utf-8') if stderr_bytes else ""
            returncode = process.returncode
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise subprocess.TimeoutExpired(cmd, timeout)
    else:
        # Use asyncio subprocess without capturing output
        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=None,
            stderr=None
        )
        
        try:
            returncode = await asyncio.wait_for(process.wait(), timeout=timeout)
            stdout = ""
            stderr = ""
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise subprocess.TimeoutExpired(cmd, timeout)

    parsed = None
    if capture_output and stdout:
        s = stdout.lstrip()
        if s.startswith("{") or s.startswith("["):
            try:
                parsed = json.loads(stdout)
            except json.JSONDecodeError:
                parsed = None

    if check and returncode != 0:
        # Mirror subprocess.run(check=True) behavior using CalledProcessError
        err = subprocess.CalledProcessError(
            returncode, cmd, output=stdout, stderr=stderr
        )
        raise err

    return ExecutionResult(
        cmd=cmd,
        returncode=returncode,
        stdout=stdout or "",
        stderr=stderr or "",
        json=parsed,
    )


@function_tool
async def run_helm(
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
    input_data: Optional[str] = None,  # pass stdin
) -> ExecutionResult:
    return await run_helm_impl(
        args,
        namespace=namespace,
        context=context,
        kubeconfig=kubeconfig,
        repo_config=repo_config,
        registry_config=registry_config,
        timeout=timeout,
        capture_output=capture_output,
        check=check,
        env=env,
        workdir=workdir,
        input_data=input_data,
    )


async def run_helm_impl(
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
    input_data: Optional[str] = None,  # pass stdin
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

    if capture_output:
        # Use asyncio subprocess for async execution with stdin support
        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            cwd=workdir,
            stdin=asyncio.subprocess.PIPE if input_data else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            input_data_bytes = input_data.encode() if input_data else None
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(input=input_data_bytes),
                timeout=timeout
            )
            stdout = stdout_bytes.decode('utf-8') if stdout_bytes else ""
            stderr = stderr_bytes.decode('utf-8') if stderr_bytes else ""
            returncode = process.returncode
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise subprocess.TimeoutExpired(cmd, timeout)
    else:
        # Use asyncio subprocess without capturing output
        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            cwd=workdir,
            stdin=asyncio.subprocess.PIPE if input_data else None,
            stdout=None,
            stderr=None
        )
        
        try:
            if input_data:
                await asyncio.wait_for(
                    process.communicate(input=input_data.encode()),
                    timeout=timeout
                )
            else:
                await asyncio.wait_for(process.wait(), timeout=timeout)
            returncode = process.returncode
            stdout = ""
            stderr = ""
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise subprocess.TimeoutExpired(cmd, timeout)

    parsed = None
    if capture_output and stdout:
        # Best-effort JSON detection/parse
        s = stdout.lstrip()
        if s.startswith("{") or s.startswith("["):
            try:
                parsed = json.loads(stdout)
            except json.JSONDecodeError:
                parsed = None

    if check and returncode != 0:
        raise subprocess.CalledProcessError(
            returncode, cmd, output=stdout, stderr=stderr
        )

    return ExecutionResult(
        cmd=cmd,
        returncode=returncode,
        stdout=stdout or "",
        stderr=stderr or "",
        json=parsed,
    )


