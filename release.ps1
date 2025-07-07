# Release preparation script for mcp4mcp
param(
    [string]$Version = "0.1.0"
)

Write-Host "🚀 Preparing release v$Version" -ForegroundColor Green
Write-Host ""

# Check if working directory is clean
$gitStatus = git status --porcelain
if ($gitStatus) {
    Write-Host "❌ Working directory is not clean. Please commit changes first." -ForegroundColor Red
    git status --short
    exit 1
}

# Update version in pyproject.toml if needed
Write-Host "📝 Checking version in pyproject.toml..." -ForegroundColor Yellow
$currentVersion = (Get-Content pyproject.toml | Select-String 'version = ' | ForEach-Object { $_.Line -replace '.*version = "([^"]*)".*', '$1' })
if ($currentVersion -ne $Version) {
    Write-Host "   Updating version from $currentVersion to $Version" -ForegroundColor Blue
    (Get-Content pyproject.toml) -replace 'version = ".*"', "version = `"$Version`"" | Set-Content pyproject.toml
    git add pyproject.toml
    git commit -m "Bump version to $Version"
}

# Build and test locally
Write-Host "🔨 Building package..." -ForegroundColor Blue
python -m build

Write-Host "✅ Testing package..." -ForegroundColor Green
python -m pip install --quiet twine
twine check dist/*

Write-Host "🏷️  Creating tag and pushing..." -ForegroundColor Magenta
git tag "v$Version"
git push origin main
git push origin "v$Version"

Write-Host ""
Write-Host "🎉 Release v$Version initiated!" -ForegroundColor Green
Write-Host ""
Write-Host "Monitor the release at:" -ForegroundColor Yellow
Write-Host "https://github.com/hmatt1/mcp4mcp/actions"
Write-Host ""
Write-Host "After completion, your package will be available at:" -ForegroundColor Yellow
Write-Host "https://pypi.org/project/mcp4mcp/"
Write-Host ""
Write-Host "Users can install with:" -ForegroundColor Yellow
Write-Host "pip install mcp4mcp"
