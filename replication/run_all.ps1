param(
    [switch]$SkipModels,
    [string]$DataDir = "data",
    [string]$Model0 = "models\dynasent_model0.bin",
    [string]$Model1 = "models\dynasent_model1.bin"
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path logs | Out-Null

$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$combined = "logs\ALL_RESULTS.txt"

$header = @()
$header += "DynaSent replication -- full run"
$header += "Date        : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss K')"
$header += "Machine     : $env:COMPUTERNAME"
$header += "Working dir : $(Get-Location)"
$header += "Python      : $(python --version 2>&1)"
$header += "Git commit  : $(git rev-parse --short HEAD 2>$null)"
$header += ""
$header += "Package versions:"
$header += (python -m pip freeze 2>&1 | Select-String -Pattern `
    "^(numpy|pandas|scikit-learn|scipy|statsmodels|matplotlib|torch|transformers|datasets)==")
$header += ""
$header += ("=" * 72)
$header | Out-File -FilePath $combined -Encoding utf8
$header | Write-Host

$steps = @(
    @{ name = "check_setup";       argv = @("replication\check_setup.py") },
    @{ name = "dataset_tables";    argv = @("replication\dataset_tables.py") },
    @{ name = "human_performance"; argv = @("replication\human_performance.py") },
    @{ name = "example_tables";    argv = @("replication\example_tables.py") },
    @{ name = "external_checks";   argv = @("replication\external_data.py") },
    @{ name = "table1";            argv = @("replication\external_data.py",
                                            "--table1", "--out", $DataDir) }
)

if (-not $SkipModels) {
    $steps += @{ name = "evaluate_model0"; argv = @("replication\evaluate.py",
                                                    "--model", $Model0,
                                                    "--which", "model0",
                                                    "--external-dir", $DataDir) }
    $steps += @{ name = "evaluate_model1"; argv = @("replication\evaluate.py",
                                                    "--model", $Model1,
                                                    "--which", "model1",
                                                    "--external-dir", $DataDir) }
}

foreach ($step in $steps) {
    $name = $step.name
    $log = "logs\${name}_$stamp.txt"

    $banner = @("", ("=" * 72), "### $name",
                "### python $($step.argv -join ' ')",
                "### started $(Get-Date -Format 'HH:mm:ss')", ("=" * 72))
    $banner | Tee-Object -FilePath $combined -Append | Write-Host

    $output = & python $step.argv 2>&1 | Tee-Object -FilePath $log
    $output | Write-Host
    $output | Out-File -FilePath $combined -Append -Encoding utf8

    if ($LASTEXITCODE -ne 0) {
        "### $name exited with code $LASTEXITCODE" |
            Tee-Object -FilePath $combined -Append | Write-Host
    }
}

$footer = @("", ("=" * 72),
            "Finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
            "Per-step logs in logs\, tables in replication\results\")
$footer | Tee-Object -FilePath $combined -Append | Write-Host
