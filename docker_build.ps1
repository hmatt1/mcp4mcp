# Docker build and run script for mcp4mcp

param(
    [string]$ImageName = "mcp4mcp",
    [string]$Tag = "latest"
)

Write-Host "🐳 Building Docker image..." -ForegroundColor Blue

# Build the Docker image
docker build -t "${ImageName}:${Tag}" .

Write-Host "✅ Docker image built successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "To run the container:" -ForegroundColor Yellow
Write-Host "  docker run -p 8000:8000 ${ImageName}:${Tag}"
Write-Host ""
Write-Host "To push to registry:" -ForegroundColor Yellow
Write-Host "  docker tag ${ImageName}:${Tag} yourusername/${ImageName}:${Tag}"
Write-Host "  docker push yourusername/${ImageName}:${Tag}"
