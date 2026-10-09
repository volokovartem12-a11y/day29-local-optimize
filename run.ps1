# run.ps1 — оптимизация локальной LLM под задачу RAG (день 29). Ключи и интернет-API не нужны.
#
#   .\run.ps1                       # всё: модели → индекс → сравнение профилей → сборка advent-rag → проверка
#   .\run.ps1 -Runs 1               # быстрее: каждый вопрос по одному разу
#   .\run.ps1 -Mode optimize        # только сравнение профилей
#   .\run.ps1 -Mode final           # собрать advent-rag из Modelfile и проверить её
#   .\run.ps1 -Mode ask -Query "Какие команды есть в чате проекта дня 9?"
#   .\run.ps1 -Mode ask -Query "..." -Baseline      # тот же вопрос модели ДО оптимизации
#   .\run.ps1 -Mode chat            # диалог с оптимизированной моделью
#   .\run.ps1 -Resume               # продолжить прерванное сравнение
#
# Если Windows блокирует запуск:  powershell -ExecutionPolicy Bypass -File .\run.ps1

param(
    [ValidateSet("all", "optimize", "final", "ask", "chat")]
    [string]$Mode = "all",
    [int]$Runs = 2,
    [string]$Query = "",
    [string[]]$Profiles = @(),
    [switch]$Resume,
    [switch]$Baseline
)

Set-Location $PSScriptRoot
$Profiles = @($Profiles | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })  # -File передаёт "a,b" одной строкой
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$api = "http://localhost:11434"
$finalModel = "advent-rag"

function Find-Python {
    $installed = @(Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe", "$env:ProgramFiles\Python3*\python.exe" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | ForEach-Object { $_.FullName })
    foreach ($candidate in @("py", "python") + $installed) {
        try {
            $output = & $candidate --version 2>&1
            if ($output -match "Python \d") { return $candidate }
        }
        catch { }
    }
    return $null
}

function Find-Ollama {
    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @("$env:LOCALAPPDATA\Programs\Ollama\ollama.exe", "$env:ProgramFiles\Ollama\ollama.exe")) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Test-Server {
    try { return (Invoke-RestMethod "$api/api/version" -TimeoutSec 3).version } catch { return $null }
}

function Get-Installed {
    return @((Invoke-RestMethod "$api/api/tags").models | ForEach-Object { $_.name })
}

# --- 1. Python и зависимости ---
$py = Find-Python
if (-not $py) { Write-Host "Python не найден. Установите: winget install Python.Python.3.12 --scope user" -ForegroundColor Red; exit 1 }
Write-Host "Python: $py" -ForegroundColor Green
& $py -m pip install -r requirements.txt --quiet --disable-pip-version-check

# --- 2. Ollama ---
$ollama = Find-Ollama
if (-not $ollama) { Write-Host "Ollama не найдена. Установите: winget install Ollama.Ollama" -ForegroundColor Red; exit 1 }
$version = Test-Server
if (-not $version) {
    Write-Host "Сервер Ollama не запущен — запускаю в фоне..." -ForegroundColor Cyan
    Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 30 -and -not $version; $i++) { Start-Sleep -Seconds 1; $version = Test-Server }
    if (-not $version) { Write-Host "Сервер не поднялся за 30 секунд. Запустите приложение Ollama вручную." -ForegroundColor Red; exit 1 }
}
Write-Host "Ollama работает локально: $api (версия $version)" -ForegroundColor Green

# --- 3. Модели: bge-m3 (поиск) + все модели из profiles.json ---
$need = @("bge-m3")
if ($Mode -eq "all" -or $Mode -eq "optimize") {
    $all = (Get-Content .\profiles.json -Raw -Encoding UTF8 | ConvertFrom-Json).profiles
    if ($Profiles.Count -gt 0) { $all = $all | Where-Object { $Profiles -contains $_.id } }
    $need += @($all | ForEach-Object { $_.model } | Sort-Object -Unique)
}
else { $need += "qwen2.5:7b" }
$names = Get-Installed
foreach ($m in $need) {
    if ($names -notcontains $m -and $names -notcontains "$m`:latest") {
        Write-Host "`nСкачиваю модель $m (один раз)..." -ForegroundColor Cyan
        & $ollama pull $m
        if ($LASTEXITCODE -ne 0) { Write-Host "Не удалось скачать $m — профили с ней будут пропущены." -ForegroundColor Yellow }
    }
}

# --- 4. Локальный индекс и пороги (как в дне 28) ---
if (-not (Test-Path ".\index\index_structure_ollama.sqlite")) {
    Write-Host "`n=== Строю локальный индекс (bge-m3) ===" -ForegroundColor Cyan
    & $py .\build_index.py --backend ollama; if ($LASTEXITCODE -ne 0) { exit 1 }
}
$needCalibrate = $true
if (Test-Path ".\config.json") { if ((Get-Content ".\config.json" -Raw) -match '"ollama"') { $needCalibrate = $false } }
if ($needCalibrate) {
    Write-Host "`n=== Подбор порогов поиска ===" -ForegroundColor Cyan
    & $py .\calibrate.py --backend ollama; if ($LASTEXITCODE -ne 0) { exit 1 }
}

# --- 5. Сравнение профилей ---
if ($Mode -eq "all" -or $Mode -eq "optimize") {
    Write-Host "`n=== Сравнение профилей: до оптимизации → параметры → промпт → квантование ===" -ForegroundColor Cyan
    Write-Host "Прогонов на вопрос: $Runs. Это займёт 20–40 минут; прервать: Ctrl+C, продолжить: .\run.ps1 -Resume" -ForegroundColor Yellow
    $a = @("--runs", $Runs, "--no-pull")
    if ($Profiles.Count -gt 0) { $a += "--profiles"; $a += $Profiles }
    if ($Resume) { $a += "--resume" }
    & $py .\optimize.py @a
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

# --- 6. Сборка оптимизированной модели и проверка ---
if ($Mode -eq "all" -or $Mode -eq "final") {
    if (-not (Test-Path ".\Modelfile")) { Write-Host "Нет Modelfile — сначала .\run.ps1 -Mode optimize" -ForegroundColor Red; exit 1 }
    Write-Host "`n=== Собираю оптимизированную модель $finalModel из Modelfile ===" -ForegroundColor Cyan
    & $ollama create $finalModel -f .\Modelfile
    if ($LASTEXITCODE -ne 0) { Write-Host "ollama create завершился с ошибкой." -ForegroundColor Red; exit 1 }
    Write-Host "`n=== Проверяю $finalModel на тех же вопросах ===" -ForegroundColor Cyan
    & $py .\optimize.py --final --runs $Runs --no-pull
    Write-Host "`nОтчёт: $PSScriptRoot\output\report_day29.md" -ForegroundColor Green
    Write-Host "Модель $finalModel теперь есть в Ollama:  ollama run $finalModel" -ForegroundColor Green
}

if ($Mode -eq "ask") {
    if (-not $Query) { Write-Host "Укажите -Query ""вопрос""" -ForegroundColor Yellow; exit 1 }
    $a = @("ask", $Query)
    if ($Baseline) { $a += "--baseline" }
    & $py .\local_rag.py @a
}
if ($Mode -eq "chat") {
    $a = @("ask")
    if ($Baseline) { $a += "--baseline" }
    & $py .\local_rag.py @a
}
