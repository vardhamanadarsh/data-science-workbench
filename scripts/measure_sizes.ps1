# Builds a throwaway environment per extra and prints its real size on disk.
# Dev tools (pytest, mypy, ruff...) are excluded, so the numbers show what a user needs.
# Run from the project folder:  powershell -ExecutionPolicy Bypass -File scripts\measure_sizes.ps1
$env:Path = "C:\Users\25msceco068\.local\bin;$env:Path"

$sets = @(
    @{ Name = "Core only";                  Extras = @() },
    @{ Name = "Core + ml";                  Extras = @("ml") },
    @{ Name = "Core + explain";             Extras = @("explain") },
    @{ Name = "Core + automl";              Extras = @("automl") },
    @{ Name = "Core + reports";             Extras = @("reports") },
    @{ Name = "Everything used by the app"; Extras = @("ml", "automl", "explain", "reports") }
)

foreach ($s in $sets) {
    $dir = Join-Path $env:TEMP ("wbsize_" + ($s.Name -replace "[^A-Za-z0-9]", "_"))
    if (Test-Path $dir) { Remove-Item $dir -Recurse -Force }
    $env:UV_PROJECT_ENVIRONMENT = $dir
    $uvArgs = @("sync", "--no-install-project", "--python", "3.11", "--no-dev")
    foreach ($e in $s.Extras) { $uvArgs += @("--extra", $e) }
    uv @uvArgs
    if ($LASTEXITCODE -ne 0) {
        "{0,-30} FAILED (see the error above)" -f $s.Name
        continue
    }
    $bytes = (Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    "{0,-30} {1,8:N0} MB" -f $s.Name, ($bytes / 1MB)
    Remove-Item $dir -Recurse -Force
}
Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue