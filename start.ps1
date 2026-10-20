$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 setup_runtime.py --frontend $PSScriptRoot
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python setup_runtime.py --frontend $PSScriptRoot
    } else {
        Write-Error 'Install Python 3.10 to 3.13 first https://www.python.org/downloads/windows/'
        exit 1
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $pythonPath bootstrap.py @args
exit $LASTEXITCODE
