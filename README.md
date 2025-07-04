
# mcp4mcp - Meta MCP Server

A Meta MCP Server that provides persistent memory and intelligent guidance for MCP development projects.

## Features

- **Project State Management**: Track tools, their status, and development progress
- **Intelligent Analysis**: AI-powered suggestions and duplication detection  
- **Persistent Storage**: SQLite-based storage for project data
- **Session Tracking**: Monitor development sessions and activities
- **Code Scanning**: Automatic discovery of MCP tools in your codebase

## Installation

```bash
pip install -e .
```

## Usage

Start the MCP server:

```bash
python server.py
```

The server provides tools for:

- `get_project_state` - Load current project state
- `update_project_state` - Update project information
- `scan_project_files` - Scan files for MCP tools
- `check_before_build` - Check for duplicates before building
- `suggest_next_action` - Get AI-powered development suggestions
- `track_development_session` - Log development activities

## Storage

Project data is stored in `~/.mcp4mcp/projects.db` using SQLite for efficient querying and persistence.

## Development

The project follows a modular structure:

- `models.py` - Pydantic data models
- `storage.py` - Database operations
- `tools/` - MCP tool implementations
- `analyzers/` - Code analysis and similarity detection
- `utils/` - Shared utilities

## Testing

Run tests with:

```bash
python -m pytest tests/
```
