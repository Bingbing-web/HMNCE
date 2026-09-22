param(
    [string]$Source = 'D:\SY\DRAE2\DRAE1\datasets'
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$target = Join-Path $projectRoot 'datasets_external'

if (-not (Test-Path -LiteralPath $Source)) {
    throw "Dataset source does not exist: $Source"
}
if (Test-Path -LiteralPath $target) {
    throw "Target already exists: $target"
}

New-Item -ItemType Junction -Path $target -Target (Resolve-Path -LiteralPath $Source).Path | Out-Null
Write-Host "Created dataset junction: $target"
Write-Host "Use: --data-root $target"
