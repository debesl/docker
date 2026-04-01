"""
@author: debes louis
"""
import json
import subprocess
import re
from typing import List, Dict, Any


def _matlab_cellstr(paths: List[str]) -> str:
    # Build MATLAB cell array literal: {'D:\a','D:\b'}
    # MATLAB accepts backslashes in char vectors fine.
    escaped = [p.replace("'", "''") for p in paths]  # escape single quotes for MATLAB
    return "{%s}" % ",".join([f"'{p}'" for p in escaped])


def run_visqol_matlab_cli(
    matlab_exe: str,
    matlab_tools_dir: str,
    base_clean_dir: str,
    nb_dir: str,
) -> Dict[str, Any]:
    nb_dir_esc = nb_dir.replace("'", "''")
    base_clean_dir_esc = base_clean_dir.replace("'", "''")
    matlab_tools_dir_esc = matlab_tools_dir.replace("'", "''")

    matlab_cmd = (
        f"addpath('{matlab_tools_dir_esc}');"
        f"r=compute_visqol_batch('{base_clean_dir_esc}', '{nb_dir_esc}');"
        "fprintf('VISQOL_JSON_START\\n');"
        "disp(jsonencode(r));"
        "fprintf('VISQOL_JSON_END\\n');"
        "exit;"
    )
    matlab_cmd = str(matlab_cmd)

    # Run MATLAB. On Windows, matlab_exe is usually "matlab" if in PATH,
    # or a full path like r"C:\Program Files\MATLAB\R2024b\bin\matlab.exe"
    proc = subprocess.run(
        [matlab_exe, "-batch", matlab_cmd],
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "MATLAB failed.\n"
            f"Return code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )

    m = re.search(r"VISQOL_JSON_START\s*(\{.*\})\s*VISQOL_JSON_END", proc.stdout, re.S)
    if not m:
        raise RuntimeError(
            "Could not find JSON markers in MATLAB output.\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )

    return json.loads(m.group(1))


