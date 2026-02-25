from typing import Dict, Any, Optional
import subprocess
import sys
import shlex
import os
import tempfile

class ToolExecutor:
    def __init__(self):
        self.tools = {
            "python": self.execute_python,
            "shell": self.execute_shell,
            "search": self.execute_search
        }

    def execute(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.tools:
            return {"status": "error", "message": f"Tool {tool_name} not found"}

        try:
            result = self.tools[tool_name](params)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def execute_python(self, params: Dict[str, Any]) -> str:
        if os.getenv("ALLOW_UNSAFE_EXECUTION", "false").lower() != "true":
            return "Error: Python execution is disabled. Set ALLOW_UNSAFE_EXECUTION=true to enable."

        code = params.get("code", "")

        # Use a temporary file to execute code in a separate process
        # This avoids using `exec()` directly in the main process
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name

            # Run the python script
            # sourcery skip: avoid-subprocess-run
            result = subprocess.run(  # nosec
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=10,
                check=False
            )
            output = result.stdout + result.stderr

        except Exception as e:
            output = str(e)
        finally:
            # Clean up
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        return output

    def execute_shell(self, params: Dict[str, Any]) -> str:
        if os.getenv("ALLOW_UNSAFE_EXECUTION", "false").lower() != "true":
            return "Error: Shell execution is disabled. Set ALLOW_UNSAFE_EXECUTION=true to enable."

        command = params.get("command", "")
        try:
            # Use shlex to split command and shell=False for security
            args = shlex.split(command)
            # sourcery skip: avoid-subprocess-run
            result = subprocess.run(args, shell=False, capture_output=True, text=True, timeout=10, check=False) # nosec
            return result.stdout + result.stderr
        except Exception as e:
            return str(e)

    def execute_search(self, params: Dict[str, Any]) -> str:
        query = params.get("query", "")
        return f"Results for {query}: [Mock Data]"
