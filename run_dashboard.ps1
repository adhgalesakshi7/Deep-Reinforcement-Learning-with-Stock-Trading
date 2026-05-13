$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = "C:\Users\HP\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Python runtime not found at $pythonExe"
    exit 1
}

& $pythonExe (Join-Path $projectRoot "server.py")
