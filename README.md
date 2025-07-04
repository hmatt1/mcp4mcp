
# mcp4mcp - Meta MCP Server

A Meta MCP Server that provides persistent memory and intelligent guidance for MCP development projects.

## 🎯 Overview

**mcp4mcp** is a specialized MCP server designed to help developers build better MCP projects. It provides:

- **Persistent Project Memory**: Track tools, their status, and development progress across sessions
- **AI-Powered Intelligence**: Get suggestions, detect duplicates, and avoid conflicts
- **Development Session Tracking**: Monitor your development activities and progress
- **Code Analysis**: Automatic discovery and analysis of MCP tools in your codebase
- **Similarity Detection**: Find similar tools to avoid duplication and improve consistency

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -e .

# Start the server
python server.py

# Or use the main entry point
python main.py server
```

### Basic Usage

```bash
# Run the demo to see all features
python main.py demo

# Run tests
python main.py test
```

## 🛠️ Features

### Core Tools

The server provides 11 MCP tools organized into three categories:

#### State Management
- **`get_project_state_tool`** - Load current project state and tools
- **`update_project_state_tool`** - Update project information and add/modify tools
- **`scan_project_files_tool`** - Automatically scan files for MCP tools

#### Intelligence & Analysis
- **`check_before_build_tool`** - Check for conflicts before building new tools
- **`suggest_next_action_tool`** - Get AI-powered development suggestions
- **`analyze_tool_similarity_tool`** - Analyze tools for similarity and duplication

#### Development Tracking
- **`track_development_session_tool`** - Log development activities and progress
- **`end_development_session_tool`** - End a development session with summary
- **`get_development_sessions_tool`** - Get recent development sessions
- **`get_session_analytics_tool`** - Get development analytics and insights
- **`update_tool_status_tool`** - Update individual tool status

## 📋 Usage Examples

### 1. Project Management

```python
# Start a new development session
await track_development_session(
    "Started working on file tools",
    "my_project",
    "file_reader",
    "Implementing CSV file reading capability"
)

# Update project with new tools
await update_project_state(
    "my_project",
    "File processing MCP server",
    [
        {
            "name": "read_csv",
            "description": "Read CSV files",
            "status": "planned"
        },
        {
            "name": "write_csv", 
            "description": "Write CSV files",
            "status": "planned"
        }
    ]
)
```

### 2. Conflict Detection

```python
# Check before building a new tool
result = await check_before_build(
    "csv_processor",
    "Process CSV files by reading and writing",
    "my_project"
)

# Result will show potential conflicts with existing tools
print(f"Conflicts found: {len(result['conflicts'])}")
print(f"Recommendations: {result['recommendations']}")
```

### 3. AI-Powered Suggestions

```python
# Get intelligent suggestions
suggestions = await suggest_next_action(
    "my_project",
    "I've implemented file reading, what should I do next?"
)

print("AI Suggestions:")
for suggestion in suggestions['suggestions']:
    print(f"- {suggestion}")
```

### 4. Code Scanning

```python
# Automatically discover tools in your codebase
scan_result = await scan_project_files("my_project", "./src")

print(f"Found {scan_result['tools_found']} tools:")
for tool in scan_result['tools']:
    print(f"- {tool['name']}: {tool['description']}")
```

## 🏗️ Architecture

### Project Structure

```
mcp4mcp/
├── server.py              # FastMCP server entry point
├── mcp4mcp/
│   ├── models.py           # Pydantic data models
│   ├── storage.py          # SQLite storage backend
│   ├── tools/              # MCP tool implementations
│   │   ├── state_management.py
│   │   ├── intelligence.py
│   │   └── tracking.py
│   ├── analyzers/          # Code analysis modules
│   │   ├── code_scanner.py
│   │   └── similarity.py
│   └── utils/              # Utility functions
├── tests/                  # Comprehensive test suite
└── examples/               # Usage examples and demos
```

### Data Models

#### ProjectState
```python
class ProjectState(BaseModel):
    name: str
    description: str
    tools: Dict[str, Tool]
    sessions: List[DevelopmentSession]
    analysis: Optional[ProjectAnalysis]
    created_at: datetime
    updated_at: datetime
```

#### Tool
```python
class Tool(BaseModel):
    name: str
    description: str
    status: ToolStatus  # PLANNED, IN_PROGRESS, IMPLEMENTED, TESTED
    file_path: Optional[str]
    function_name: Optional[str]
    parameters: List[Dict[str, Any]]
    return_type: Optional[str]
    similarity_scores: Dict[str, float]
```

## 💾 Storage

All project data is stored in SQLite at `~/.mcp4mcp/projects.db` with the following tables:

- **projects** - Project metadata and state
- **tools** - Individual tool definitions and status
- **sessions** - Development session tracking
- **session_actions** - Detailed session activities

## 🧪 Testing

Comprehensive test suite covering:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_models.py -v      # Data models
python -m pytest tests/test_storage.py -v     # Storage backend
python -m pytest tests/test_tools.py -v       # Tool functionality
python -m pytest tests/test_server.py -v      # Server integration
```

## 📚 Examples

### Example Project

The `examples/example_project/` directory contains a sample MCP server with:

- File manipulation tools (read, write, list)
- Mathematical calculation tools (calculator, sqrt, power, factorial)
- Proper FastMCP integration

### Demo Script

Run the comprehensive demo:

```bash
python examples/demo_usage.py
```

This demonstrates:
- Project creation and management
- Tool scanning and analysis
- Development session tracking
- AI-powered suggestions
- Conflict detection

## 🔧 Configuration

### Environment Variables

- `MCP4MCP_DB_PATH` - Custom database path (default: `~/.mcp4mcp/projects.db`)
- `MCP4MCP_LOG_LEVEL` - Logging level (default: `INFO`)

### FastMCP Integration

```python
from fastmcp import FastMCP
from mcp4mcp.tools.state_management import register_state_tools
from mcp4mcp.tools.intelligence import register_intelligence_tools
from mcp4mcp.tools.tracking import register_tracking_tools

mcp = FastMCP("your-mcp-server")

# Register mcp4mcp tools
register_state_tools(mcp)
register_intelligence_tools(mcp)
register_tracking_tools(mcp)

# Register your own tools
@mcp.tool()
def your_tool():
    return "Hello from your tool!"

mcp.run()
```

## 🚀 Development

### Adding New Tools

1. Create tool functions in appropriate module (`mcp4mcp/tools/`)
2. Add tests in `tests/test_tools.py`
3. Register tools in `server.py`
4. Update documentation

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🤝 Support

- **Issues**: Report bugs and request features on GitHub
- **Documentation**: Full API documentation in the code
- **Examples**: See `examples/` directory for usage patterns

---

**mcp4mcp** - Making MCP development smarter, one tool at a time! 🚀
