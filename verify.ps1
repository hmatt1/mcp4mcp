# Pre-release verification script for mcp4mcp

Write-Host "🔍 Pre-release verification for mcp4mcp" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Check Python version
Write-Host "📍 Python version:" -ForegroundColor Yellow
python --version

# Check if package can be imported
Write-Host "📦 Testing package import..." -ForegroundColor Yellow
try {
    python -c "import mcp4mcp; print('✅ Import successful')"
} catch {
    Write-Host "❌ Import failed" -ForegroundColor Red
}

# Check main entry point
Write-Host "🎯 Testing main entry point..." -ForegroundColor Yellow
try {
    python main.py > $null 2>&1
    Write-Host "✅ Main entry point works" -ForegroundColor Green
} catch {
    Write-Host "❌ Main entry point failed" -ForegroundColor Red
}

# Check if we can build
Write-Host "🔨 Testing package build..." -ForegroundColor Yellow
try {
    python -m build > $null 2>&1
    Write-Host "✅ Package builds successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Package build failed" -ForegroundColor Red
}

# Check package contents
Write-Host "📋 Package contents:" -ForegroundColor Yellow
try {
    python -m twine check dist/* 2>$null
    Write-Host "✅ Package passes twine check" -ForegroundColor Green
} catch {
    Write-Host "❌ Package fails twine check" -ForegroundColor Red
}

# Check git status
Write-Host "🔧 Git status:" -ForegroundColor Yellow
$gitStatus = git status --porcelain
if ($gitStatus) {
    Write-Host "❌ Working directory has uncommitted changes:" -ForegroundColor Red
    git status --short
    Write-Host "   Please commit before releasing" -ForegroundColor Yellow
} else {
    Write-Host "✅ Working directory is clean" -ForegroundColor Green
}

# Check if GitHub secret is set
Write-Host "🔐 GitHub secrets:" -ForegroundColor Yellow
$secretCheck = gh secret list | Select-String "PYPI_API_TOKEN"
if ($secretCheck) {
    Write-Host "✅ PYPI_API_TOKEN is set" -ForegroundColor Green
} else {
    Write-Host "❌ PYPI_API_TOKEN not found" -ForegroundColor Red
}

Write-Host ""
Write-Host "🎯 Ready for release? Run:" -ForegroundColor Cyan
Write-Host "   .\release.ps1 [version]" -ForegroundColor White
Write-Host "   Example: .\release.ps1 0.1.0" -ForegroundColor White
