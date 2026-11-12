
param(
    [string]$PythonExe = (Join-Path $env:LOCALAPPDATA 'Python\pythoncore-3.14-64\python.exe'),
    [string]$OllamaModel = 'qwen3:4b-thinking-2507-q4_K_M',
    [string]$TokenizerName = 'Qwen/Qwen3-4B-Thinking-2507',
    [string]$OllamaUrl = 'http://localhost:11434',
    [int]$Limit = 30,
    [int]$ReflexSize = 16,
    [double]$Temperature = 0.6,
    [int]$MaxNewTokens = 8192,
    [int]$NumCtx = 8192
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir '..\..')).Path
$pythonScript = Join-Path $repoRoot 'experiments\scripts\ollama_kflow_qwen.py'
$resultsDir = Join-Path $repoRoot 'experiments\results\aime2025_ollama'

if ($Limit -lt 1) {
    throw 'Limit must be at least 1.'
}

if ($ReflexSize -lt 1) {
    throw 'ReflexSize must be at least 1.'
}

if ($MaxNewTokens -lt 1) {
    throw 'MaxNewTokens must be at least 1.'
}

if ($NumCtx -lt 1) {
    throw 'NumCtx must be at least 1.'
}

if ($Temperature -lt 0) {
    throw 'Temperature cannot be negative.'
}

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}

$pythonVersionText = (& $PythonExe --version 2>&1 | Out-String).Trim()

if ($LASTEXITCODE -ne 0 -or $pythonVersionText -notmatch 'Python\s+(\d+)\.(\d+)') {
    throw "Could not identify Python version: $pythonVersionText"
}

$pythonMajor = [int]$Matches[1]
$pythonMinor = [int]$Matches[2]

if ([version]::Parse("$pythonMajor.$pythonMinor") -lt [version]'3.10') {
    throw "Python 3.10+ is required. Found: $pythonVersionText"
}

if (-not (Test-Path -LiteralPath $pythonScript -PathType Leaf)) {
    throw "Python script not found: $pythonScript"
}

$baseUrl = $OllamaUrl.TrimEnd('/')

try {
    $ollamaVersion = Invoke-RestMethod -Uri "$baseUrl/api/version" -TimeoutSec 5
    $ollamaTags = Invoke-RestMethod -Uri "$baseUrl/api/tags" -TimeoutSec 10
}
catch {
    throw "Cannot reach Ollama at $OllamaUrl. Details: $($_.Exception.Message)"
}

$availableModels = @(
    $ollamaTags.models | ForEach-Object { $_.name }
)

if ($availableModels -notcontains $OllamaModel) {
    throw "Model '$OllamaModel' not found. Available models: $($availableModels -join ', ')"
}

& $PythonExe -c 'import datasets, transformers'

if ($LASTEXITCODE -ne 0) {
    throw 'Required packages datasets and transformers are missing.'
}

New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null

Write-Host "Python: $pythonVersionText"
Write-Host "Ollama: $($ollamaVersion.version)"
Write-Host "Model: $OllamaModel"
Write-Host "Questions: $Limit"
Write-Host "TRT rounds: $ReflexSize"
Write-Host "Temperature: $Temperature"
Write-Host "Max new tokens: $MaxNewTokens"
Write-Host "Context: $NumCtx"
Write-Host "Results: $resultsDir"

$pythonArgs = @(
    $pythonScript,
    '--model_name', $OllamaModel,
    '--tokenizer_name', $TokenizerName,
    '--ollama_url', $OllamaUrl,
    '--limit', "$Limit",
    '--split', 'train',
    '--reflex_size', "$ReflexSize",
    '--temperature', "$Temperature",
    '--max_new_tokens', "$MaxNewTokens",
    '--num_ctx', "$NumCtx"
)

Push-Location $repoRoot

try {
    & $PythonExe @pythonArgs

    if ($LASTEXITCODE -ne 0) {
        throw "TRT Ollama run failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host "TRT evaluation completed. Results: $resultsDir"
