[CmdletBinding()]
param(
    [int]$ApiPort = 8000,
    [int]$WebPort = 5173,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$projectRootPattern = [regex]::Escape($projectRoot)
$logRoot = Join-Path $projectRoot 'logs'
$backendOutLog = Join-Path $logRoot 'backend.out.log'
$backendErrLog = Join-Path $logRoot 'backend.err.log'
$frontendOutLog = Join-Path $logRoot 'frontend.out.log'
$frontendErrLog = Join-Path $logRoot 'frontend.err.log'
$stdinFile = Join-Path $logRoot 'empty.stdin'

function Write-Step([string]$Message) {
    Write-Host "[cc] $Message" -ForegroundColor Cyan
}

function Wait-ForHttpEndpoint([string]$Uri, [int]$Attempts = 15) {
    $lastError = $null
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 2 | Out-Null
            return
        } catch {
            $lastError = $_.Exception
            if ($attempt -lt $Attempts) { Start-Sleep -Seconds 1 }
        }
    }
    throw "Endpoint did not become available: $Uri. $($lastError.Message)"
}

function Get-ProjectProcesses {
    $all = @(Get-CimInstance Win32_Process)
    $portOwners = @(
        Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
            Where-Object { $_.LocalPort -in @($ApiPort, $WebPort) } |
            Select-Object -ExpandProperty OwningProcess -Unique
    )
    $matches = @($all | Where-Object {
        $command = [string]$_.CommandLine
        $isProjectCommand = $command -and (
            $command -match 'scripts[\\/]run_api\.py' -or
            $command -match 'web[\\/]node_modules[\\/]\.bin[\\/]vite' -or
            $command -match 'web[\\/]node_modules[\\/]vite' -or
            $command -match 'vite[\\/]bin[\\/]vite\.js' -or
            $command -match 'npm(?:\.cmd)?[\"\s].*run\s+dev'
        )
        $isProjectCommand -and (
            $command -match $projectRootPattern -or
            [int]$_.ProcessId -in $portOwners
        )
    })

    foreach ($ownerPid in $portOwners) {
        $owner = $all | Where-Object { [int]$_.ProcessId -eq [int]$ownerPid }
        if ($owner -and $owner -notin $matches) {
            throw "Port $ApiPort or $WebPort is used by an unrelated process: PID $ownerPid ($($owner.Name))."
        }
    }
    $ids = [System.Collections.Generic.HashSet[int]]::new()
    foreach ($item in $matches) { [void]$ids.Add([int]$item.ProcessId) }
    do {
        $added = $false
        foreach ($item in $all) {
            if ($ids.Contains([int]$item.ParentProcessId) -and $ids.Add([int]$item.ProcessId)) {
                $added = $true
            }
        }
    } while ($added)
    $all | Where-Object { $ids.Contains([int]$_.ProcessId) }
}

if ($ApiPort -lt 1 -or $ApiPort -gt 65535) { throw 'ApiPort must be between 1 and 65535.' }
if ($WebPort -lt 1 -or $WebPort -gt 65535) { throw 'WebPort must be between 1 and 65535.' }

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
if (-not (Test-Path -LiteralPath $stdinFile)) {
    New-Item -ItemType File -Path $stdinFile | Out-Null
}

Write-Step 'Stopping old project API/Vite processes.'
$oldProcesses = @(Get-ProjectProcesses)
foreach ($process in $oldProcesses) {
    Write-Host "  stopping PID $($process.ProcessId): $($process.Name)"
    Stop-Process -Id ([int]$process.ProcessId) -Force -ErrorAction SilentlyContinue
}
if ($oldProcesses.Count -gt 0) { Start-Sleep -Milliseconds 500 }

if (-not $SkipBuild) {
    Write-Step 'Building the frontend.'
    Push-Location (Join-Path $projectRoot 'web')
    try {
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend build failed with exit code $LASTEXITCODE." }
    } finally { Pop-Location }
}

Write-Step 'Resolving the dba-py311 Python interpreter.'
$pythonExe = if ($env:CONDA_ROOT) {
    Join-Path (Join-Path $env:CONDA_ROOT 'envs') 'dba-py311\\python.exe'
} else {
    ''
}
if (-not $pythonExe -or -not (Test-Path -LiteralPath $pythonExe)) {
    $condaJson = (& conda env list --json | ConvertFrom-Json)
    $environmentRoot = $condaJson.envs | Where-Object { $_ -match '[\\/]dba-py311$' } | Select-Object -First 1
    $pythonExe = if ($environmentRoot) { Join-Path $environmentRoot 'python.exe' } else { '' }
}
if (-not $pythonExe -or -not (Test-Path -LiteralPath $pythonExe)) { throw "Could not resolve dba-py311 Python: $pythonExe" }

Write-Step 'Starting the backend.'
$backend = Start-Process -FilePath $pythonExe -ArgumentList @('scripts/run_api.py', '--host', '127.0.0.1', '--port', "$ApiPort") -WorkingDirectory $projectRoot -RedirectStandardInput $stdinFile -RedirectStandardOutput $backendOutLog -RedirectStandardError $backendErrLog -WindowStyle Hidden -PassThru

Write-Step 'Starting the frontend dev server.'
$frontend = Start-Process -FilePath 'npm.cmd' -ArgumentList @('run', 'dev', '--', '--host', '127.0.0.1', '--port', "$WebPort") -WorkingDirectory (Join-Path $projectRoot 'web') -RedirectStandardInput $stdinFile -RedirectStandardOutput $frontendOutLog -RedirectStandardError $frontendErrLog -WindowStyle Hidden -PassThru

Start-Sleep -Seconds 2
$backendAlive = Get-Process -Id $backend.Id -ErrorAction SilentlyContinue
$frontendAlive = Get-Process -Id $frontend.Id -ErrorAction SilentlyContinue
Write-Host ''
Write-Host 'Project started.' -ForegroundColor Green
Write-Host "  Backend:  http://127.0.0.1:$ApiPort (PID $($backend.Id))"
Write-Host "  Frontend: http://127.0.0.1:$WebPort (PID $($frontend.Id))"
Write-Host "  Backend logs:  $backendOutLog / $backendErrLog"
Write-Host "  Frontend logs: $frontendOutLog / $frontendErrLog"
if (-not $backendAlive -or -not $frontendAlive) { Write-Warning 'A launcher exited early; inspect the logs.'; exit 1 }

try {
    Wait-ForHttpEndpoint "http://127.0.0.1:$ApiPort/api/health"
    Wait-ForHttpEndpoint "http://127.0.0.1:$WebPort/"
    Write-Host 'Health checks passed.' -ForegroundColor Green
}
catch {
    Write-Warning "A health check failed: $($_.Exception.Message)"
    exit 1
}
