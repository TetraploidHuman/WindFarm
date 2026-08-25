param(
    [string]$BindHost = "127.0.0.1",
    [int]$DashboardPort = 8765,
    [int]$ApiPort = 8780,
    [int]$InteractivePort = 8870,
    [string]$RunName = "quickstart",
    [switch]$RefreshDemo,
    [switch]$SkipApi,
    [switch]$SkipDashboard,
    [switch]$SkipInteractive
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$runsDir = Join-Path $repoRoot "runs"
$runDir = Join-Path $runsDir $RunName
$env:PYTHONPATH = Join-Path $repoRoot "src"

function Resolve-Python {
    $candidates = @(
        (Join-Path $repoRoot ".venv\Scripts\python.exe"),
        "python"
    )
    foreach ($candidate in $candidates) {
        if ($candidate -eq "python") {
            try {
                $null = & $candidate --version 2>$null
                return $candidate
            } catch {
                continue
            }
        }
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    throw "No Python interpreter found. Expected .venv\Scripts\python.exe or python in PATH."
}

function Stop-PortListeners {
    param(
        [int[]]$Ports
    )

    foreach ($port in $Ports) {
        $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        if ($null -eq $connections) {
            continue
        }
        $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($processId in $pids) {
            if ($processId -and $processId -ne 0) {
                try {
                    Stop-Process -Id $processId -Force -ErrorAction Stop
                    Write-Host "Stopped process $processId on port $port"
                } catch {
                    Write-Warning "Failed to stop process $processId on port ${port}: $($_.Exception.Message)"
                }
            }
        }
        Start-Sleep -Milliseconds 300
    }
}

function Ensure-RunArtifacts {
    param(
        [string]$PythonExe
    )

    $requiredFiles = @(
        (Join-Path $runDir "config.json"),
        (Join-Path $runDir "terrain.json"),
        (Join-Path $runDir "model.json"),
        (Join-Path $runDir "mission_report.json")
    )

    $needsRefresh = $RefreshDemo.IsPresent
    foreach ($requiredFile in $requiredFiles) {
        if (-not (Test-Path $requiredFile)) {
            $needsRefresh = $true
            break
        }
    }

    if (-not $needsRefresh) {
        return
    }

    New-Item -ItemType Directory -Path $runsDir -Force | Out-Null
    $seedRuns = @("demo02", "demo01", "live-demo", "session-live")
    foreach ($seedRun in $seedRuns) {
        $seedDir = Join-Path $runsDir $seedRun
        $seedOk = $true
        foreach ($requiredFile in $requiredFiles) {
            $seedFile = Join-Path $seedDir (Split-Path $requiredFile -Leaf)
            if (-not (Test-Path $seedFile)) {
                $seedOk = $false
                break
            }
        }
        if ($seedOk -and -not $RefreshDemo.IsPresent) {
            New-Item -ItemType Directory -Path $runDir -Force | Out-Null
            Copy-Item -Path (Join-Path $seedDir "*") -Destination $runDir -Recurse -Force
            Write-Host "Reused existing artifacts from $seedDir"
            return
        }
    }

    & $PythonExe -m windfarm.cli run-demo --config (Join-Path $repoRoot "config.json") --runs-dir $runsDir --run-name $RunName
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to prepare demo artifacts."
    }
}

function Start-BackgroundService {
    param(
        [string]$Name,
        [string]$PythonExe,
        [string[]]$Arguments
    )

    New-Item -ItemType Directory -Path $runsDir -Force | Out-Null
    $stdoutPath = Join-Path $runsDir "$Name.out.log"
    $stderrPath = Join-Path $runsDir "$Name.err.log"

    $process = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $Arguments `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru `
        -WindowStyle Hidden

    Write-Host "$Name started with PID $($process.Id)"
}

$pythonExe = Resolve-Python
Stop-PortListeners -Ports @($DashboardPort, $ApiPort, $InteractivePort)
Ensure-RunArtifacts -PythonExe $pythonExe

if (-not $SkipDashboard) {
    Start-BackgroundService `
        -Name "run_project_dashboard" `
        -PythonExe $pythonExe `
        -Arguments @(
            "-m", "windfarm.cli", "serve-dashboard",
            "--report", (Join-Path $runDir "mission_report.json"),
            "--host", $BindHost,
            "--port", "$DashboardPort",
            "--interval-seconds", "1.5"
        )
}

if (-not $SkipInteractive) {
    Start-BackgroundService `
        -Name "run_project_interactive" `
        -PythonExe $pythonExe `
        -Arguments @(
            "-m", "windfarm.cli", "serve-navigate",
            "--config", (Join-Path $runDir "config.json"),
            "--terrain", (Join-Path $runDir "terrain.json"),
            "--model", (Join-Path $runDir "model.json"),
            "--host", $BindHost,
            "--port", "$InteractivePort",
            "--interval-seconds", "0.2",
            "--sim-minutes-per-tick", "1"
        )
}

if (-not $SkipApi) {
    Start-BackgroundService `
        -Name "run_project_api" `
        -PythonExe $pythonExe `
        -Arguments @(
            "-m", "windfarm.cli", "serve-api",
            "--host", $BindHost,
            "--port", "$ApiPort"
        )
}

Write-Host ""
if (-not $SkipDashboard) {
    Write-Host "Replay dashboard:  http://$BindHost`:$DashboardPort"
}
if (-not $SkipInteractive) {
    Write-Host "Interactive panel: http://$BindHost`:$InteractivePort"
}
if (-not $SkipApi) {
    Write-Host "API endpoint:      http://$BindHost`:$ApiPort/api/v1/health"
}
Write-Host "Run directory:     $runDir"
