# Build and package script for mcp4mcp

Write-Host "🚀 Building mcp4mcp package..." -ForegroundColor Green

# Clean previous builds
Write-Host "🧹 Cleaning previous builds..." -ForegroundColor Yellow
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue dist, build, *.egg-info

# Install build dependencies
Write-Host "📦 Installing build dependencies..." -ForegroundColor Blue
pip install build twine

# Build the package
Write-Host "🔨 Building package..." -ForegroundColor Blue
python -m build

# Check the package
Write-Host "✅ Checking package..." -ForegroundColor Green
twine check dist/*

Write-Host "📋 Build complete! Files in dist/:" -ForegroundColor Cyan
Get-ChildItem dist/ | Format-Table Name, Length, LastWriteTime

Write-Host ""
Write-Host "🎉 Package ready for distribution!" -ForegroundColor Green
Write-Host ""
Write-Host "To test locally:" -ForegroundColor Yellow
Write-Host "  pip install dist/mcp4mcp-*.whl"
Write-Host ""
Write-Host "To upload to PyPI:" -ForegroundColor Yellow
Write-Host "  twine upload dist/*"
Write-Host ""
Write-Host "To upload to Test PyPI:" -ForegroundColor Yellow
Write-Host "  twine upload --repository testpypi dist/*"
