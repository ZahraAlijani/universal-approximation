@echo off
setlocal
C:\Users\Zahra\anaconda3\Scripts\conda.exe run -p C:\Users\Zahra\anaconda3 --no-capture-output python C:\Users\Zahra\.vscode\extensions\ms-python.python-2026.4.0-win32-x64\python_files\get_output_via_markers.py "%~dp0uap_constructive_demo.py" --no-show --output-dir "%~dp0outputs"
endlocal
