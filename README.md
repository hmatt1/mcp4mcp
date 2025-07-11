# mcp4mcp

> **Meta MCP Server** - AI-powered development assistant for building better MCP projects

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastMCP](https://img.shields.io/badge/FastMCP-compatible-green.svg)](https://github.com/jlowin/fastmcp)
[![License: Unlicense](https://img.shields.io/badge/License-Unlicense-blue.svg)](https://unlicense.org/)

**mcp4mcp** automatically tracks your MCP tools, detects duplicates, suggests improvements, and provides AI-powered guidance throughout your development process.

## ⚡ Zero-Install Setup (Recommended)

Like `npx` for Node.js, use `uvx` for instant Python package execution:

```json
{
  "mcpServers": {
    "mcp4mcp": {
      "command": "uvx",
      "args": ["--from", "mcp4mcp", "mcp4mcp-server"]
    }
  }
}
```

**No installation needed!** Claude Desktop automatically downloads and runs mcp4mcp when needed.

## 🚀 Quick Start

### Installation

```bash
# Install mcp4mcp from PyPI
pip install mcp4mcp
```

### Setup with Claude Desktop

Add this to your Claude Desktop `claude_desktop_config.json` file:

```json
{
  "mcpServers": {
    "mcp4mcp": {
      "command": "mcp4mcp-server",
      "args": []
    }
  }
}
```

Claude Desktop will automatically start and manage the mcp4mcp server. Then just chat:

*"Help me build an MCP server for file processing. Use mcp4mcp to track progress and avoid duplicates."*

### Manual Usage (Optional)

```bash
# Or run manually for testing
mcp4mcp-server

# Or run the demo
mcp4mcp demo
```

### Development Setup

For local development and testing:

```bash
# Clone and install from source
git clone https://github.com/hmatt1/mcp4mcp.git
cd mcp4mcp
pip install -e .
```

**Test with Claude Desktop (Development):**

```json
{
  "mcpServers": {
    "mcp4mcp-dev": {
      "command": "python",
      "args": ["C:\\path\\to\\your\\mcp4mcp\\server.py"]
    }
  }
}
```

Replace `C:\\path\\to\\your\\mcp4mcp\\` with your actual project path.

**Troubleshooting:**
- Ensure dependencies: `pip install fastmcp pydantic aiosqlite`
- Test import: `python -c "import mcp4mcp; print('OK')"`
- Test server: `python server.py`

### Using with Claude Desktop

**mcp4mcp** integrates seamlessly with **Claude Desktop** for AI-powered MCP development:

```bash
# 1. Start mcp4mcp server (in one terminal)
mcp4mcp-server

# 2. Configure Claude Desktop MCP settings
# Add to your Claude Desktop configuration:
# Server Name: mcp4mcp
# Command: mcp4mcp-server

# 3. Chat with Claude Desktop about MCP development
"Help me build an MCP server for database operations. Use mcp4mcp to track progress."
```

**What Claude Desktop can do with mcp4mcp:**
- 🎯 **Track development sessions** automatically
- 🔍 **Check for duplicate tools** before building
- 💡 **Provide AI suggestions** based on your project state  
- 📊 **Monitor progress** across development sessions
- 🔄 **Discover existing tools** in your codebase

### Development Installation

```bash
# Install from source for development
git clone https://github.com/hmatt1/mcp4mcp.git
cd mcp4mcp && pip install -e .
python server.py
```

## 💡 What You Get

- 🧠 **AI suggestions** for next development steps
- 🔍 **Duplicate detection** before you build conflicting tools
- 📊 **Progress tracking** across development sessions
- 🔄 **Auto-discovery** of tools in your codebase
- 📈 **Analytics** on your development patterns

## 🛠️ Core Tools

|Tool                            |Purpose                               |
|--------------------------------|--------------------------------------|
|`get_project_state_tool`        |Load your current project and tools   |
|`update_project_state_tool`     |Add/update tools and project info     |
|`scan_project_files_tool`       |Auto-discover tools in your code      |
|`check_before_build_tool`       |Check for conflicts before building   |
|`suggest_next_action_tool`      |Get AI-powered development suggestions|
|`analyze_tool_similarity_tool`  |Find similar/duplicate tools          |
|`track_development_session_tool`|Log your development activities       |
|`get_session_analytics_tool`    |View development insights             |

## 🤖 Using with Claude Desktop

**mcp4mcp** is designed to work perfectly with **Claude Desktop** for intelligent MCP development:

### Quick Setup

```bash
# Terminal 1: Start mcp4mcp server
mcp4mcp-server

# Terminal 2: Configure Claude Desktop MCP settings
# In Claude Desktop:
# - Go to Settings → MCP Servers
# - Add new server:
#   Name: mcp4mcp
#   Command: mcp4mcp-server
#   Args: (leave empty)
```

### Example Claude Desktop Sessions

#### Building a New MCP Server
```
You: "Help me build an MCP server for file processing with CSV and JSON support."

Claude Desktop automatically:
1. 🎯 Calls track_development_session_tool("file-processing-mcp", "Building CSV/JSON tools")
2. 🔍 Uses check_before_build_tool to see if similar tools exist
3. 💡 Calls suggest_next_action_tool for personalized guidance
4. 📊 Tracks progress as you build each tool
```

#### Avoiding Duplicate Work
```
You: "I want to add a data validation tool to my project."

Claude Desktop automatically:
1. 🔍 Scans existing tools with scan_project_files_tool
2. ⚠️  Alerts if similar validation tools already exist
3. 💡 Suggests reusing or extending existing tools instead
4. 🎯 Tracks the decision in your development session
```

#### Getting Development Insights
```
You: "What should I work on next for my MCP project?"

Claude Desktop automatically:
1. 📊 Calls get_session_analytics_tool for project insights
2. 💡 Uses suggest_next_action_tool based on current state
3. 🎯 Provides personalized recommendations
4. 📈 Shows development patterns and progress
```

### Benefits of Claude Desktop + mcp4mcp

- **🧠 Intelligent Context**: Claude Desktop understands MCP development patterns
- **🔍 Automatic Conflict Detection**: Prevents duplicate tool development
- **📊 Continuous Tracking**: Every development action is logged automatically
- **💡 Contextual Suggestions**: AI guidance based on your specific project state
- **🚀 Accelerated Development**: Focus on building, not project management

## 📋 Usage Examples

### Start Development Session

```python
# Log what you're working on
await track_development_session(
    "Building file processing tools", 
    "my_project",
    "csv_reader"
)
```

### Check Before Building

```python
# Avoid duplicates
result = await check_before_build(
    "file_processor", 
    "Process CSV files",
    "my_project" 
)

if result['conflicts']:
    print("⚠️ Similar tools exist - consider reusing instead")
```

### Get AI Suggestions

```python
# Get personalized guidance
suggestions = await suggest_next_action(
    "my_project",
    "Just finished the CSV reader, what's next?"
)

for suggestion in suggestions['suggestions']:
    print(f"💡 {suggestion}")
```

### Auto-Discover Tools

```python
# Scan your codebase
result = await scan_project_files("my_project", "./src")
print(f"🔍 Found {result['new_tools']} new tools")
```
## 🤖 AI-Powered MCP Development

Use this prompt template with any LLM to build MCP servers that leverage mcp4mcp:

```markdown
# MCP Server Development with mcp4mcp

You are an expert MCP (Model Context Protocol) developer building a new MCP server. 
You have access to mcp4mcp tools that provide intelligent development assistance.

## Your Development Process:

1. **Start Each Session**: Always begin by calling `track_development_session_tool` 
   to log what you're working on

2. **Before Building Any Tool**: Call `check_before_build_tool` to check for 
   conflicts and similar existing tools

3. **Get AI Guidance**: Use `suggest_next_action_tool` for personalized 
   development recommendations based on project state

4. **Update Progress**: Use `update_project_state_tool` to track tools as you 
   build them (planned → in_progress → completed)

5. **Discover Existing Tools**: Use `scan_project_files_tool` to automatically 
   find tools in the codebase

6. **Check for Duplicates**: Run `analyze_tool_similarity_tool` periodically 
   to find similar tools that could be consolidated

## Current Task:
Build a [DOMAIN] MCP server with tools for [SPECIFIC_FUNCTIONALITY].

## Project Details:
- Project name: [PROJECT_NAME]
- Description: [PROJECT_DESCRIPTION] 
- Key requirements: [LIST_REQUIREMENTS]

Start by calling the appropriate mcp4mcp tools to understand the current state 
and get AI-powered suggestions for the best approach.
```

### Example Prompt Usage:

```markdown
# MCP Server Development with mcp4mcp

You are an expert MCP developer building a new MCP server. You have access to mcp4mcp tools.

## Current Task:
Build a file processing MCP server with tools for reading, writing, and transforming CSV/JSON files.

## Project Details:
- Project name: file-processor-mcp
- Description: MCP server for file operations with data transformation capabilities
- Key requirements: 
  * Read CSV and JSON files
  * Write data in multiple formats  
  * Transform data between formats
  * Validate file schemas
  * Handle large files efficiently

Start by calling mcp4mcp tools to check current state and get development guidance.
```

### Example Prompt: File System MCP Server

```markdown
# MCP Server Development with mcp4mcp

You are an expert MCP (Model Context Protocol) developer building a new MCP server.
You have access to mcp4mcp tools that provide intelligent development assistance.

## Current Task:
Build a file system MCP server with comprehensive file and directory management capabilities.

## Project Details:
- Project name: filesystem-mcp
- Description: A robust MCP server providing file system operations with safety limits and encoding support
- Key requirements:
  * List files and directories with recursive options and safety limits
  * Read file contents with multiple encoding support (utf-8, binary, etc.)
  * Support reading files in chunks and with size limits
  * Generate hex dumps of binary files
  * Search files by extension, regex patterns, and other criteria
  * Create new files safely with proper error handling
  * Make line replacements in text files with backup options
  * Proper error handling and validation for all operations
  * Support for common character encodings
  * Built-in safety limits to prevent abuse
  
## Implementation Guidelines:
1. Start with core file listing and reading capabilities
2. Add extension filtering and search functionality
3. Implement file creation and modification features
4. Add safety checks and validation throughout
5. Support multiple encoding formats
6. Implement chunked reading for large files
7. Add hex dump capability for binary files

## Architecture Considerations:
- Use a modular design to separate different file operations
- Implement proper validation and safety checks
- Consider adding configuration options for limits and defaults
- Include comprehensive error handling
- Add logging for file operations

## How to Use mcp4mcp Tools for This Project

To ensure a robust, efficient, and non-duplicative development process, rely on mcp4mcp tools at every stage:

1. **Track Your Development Sessions**  
   Use `track_development_session_tool` to log each new session, including your current focus (e.g., "Implementing file listing"). This creates a clear project history and helps with analytics.

2. **Check for Existing Tools Before Building**  
   Before implementing any new feature (such as file reading or extension search), always call `check_before_build_tool` to detect similar or duplicate tools in your project. This prevents wasted effort and encourages code reuse.

3. **Get AI Guidance for Each Step**  
   Use `suggest_next_action_tool` to get personalized, context-aware recommendations for your next development steps. This leverages mcp4mcp's AI to optimize your workflow and avoid common pitfalls.

4. **Scan the Project for Existing Code**  
   Regularly run `scan_project_files_tool` to auto-discover tools and code patterns in your codebase. This helps you find reusable components and ensures your project state is always up to date.

5. **Update Project State as You Build**  
   As you add or modify tools, use `update_project_state_tool` to keep your project metadata current. Mark tools as planned, in progress, or completed for accurate tracking.

6. **Analyze for Similarity and Duplicates**  
   Periodically run `analyze_tool_similarity_tool` to identify overlapping or redundant tools, so you can consolidate functionality and keep your codebase clean.

7. **Review Analytics and Progress**  
   Use `get_session_analytics_tool` to review your development patterns, session history, and tool usage. This helps you reflect on your workflow and plan improvements.

By integrating these mcp4mcp tools into your workflow, you ensure that your file system MCP server is built efficiently, avoids duplication, and benefits from continuous AI-powered guidance and tracking.
```

## Basic Usage Example of File System MCP Server:

```python
# Example tool invocations once built
files = await list_files_tool(
    path="./src",
    recursive=True,
    max_depth=3,
    exclude_patterns=["*.pyc", "__pycache__"]
)

content = await read_file_tool(
    path="./config.json",
    encoding="utf-8",
    max_size_mb=10
)

search_results = await find_files_tool(
    pattern="*.py",
    search_dir="./src",
    recursive=True
)

await replace_in_file_tool(
    path="./settings.ini",
    old_string="debug=false",
    new_string="debug=true",
    backup=True
)
```
## 🔧 Integration

Add mcp4mcp to any FastMCP project:

```python
from fastmcp import FastMCP
from mcp4mcp.tools.state_management import register_state_tools
from mcp4mcp.tools.intelligence import register_intelligence_tools  
from mcp4mcp.tools.tracking import register_tracking_tools

# Your MCP server
mcp = FastMCP("your-server")

# Add mcp4mcp intelligence
register_state_tools(mcp)
register_intelligence_tools(mcp) 
register_tracking_tools(mcp)

# Your tools
@mcp.tool()
def your_tool():
    return "Hello World"

mcp.run()
```

## 📊 Development Analytics

View your development patterns:

```python
# Get insights on your development
analytics = await get_session_analytics("my_project", days=7)

print(f"📈 This week:")
print(f"  Sessions: {analytics['total_sessions']}")  
print(f"  Time: {analytics['total_development_time']}")
print(f"  Tools: {len(analytics['tools_worked_on'])}")
```

## 🗃️ Data Storage

All data stored locally in `~/.mcp4mcp/projects.db` - no external dependencies.

## 🧪 Testing

```bash
# Run demo
python main.py demo

# Run tests  
python main.py test

# FastMCP diagnostics
python run_diagnostic.py
```

## 🛠️ Development

```bash
# Setup
git clone https://github.com/hmatt1/mcp4mcp.git
cd mcp4mcp
pip install -e ".[dev]"

# Test
python -m pytest tests/ -v
```

## 📄 License

This is free and unencumbered software released into the public domain. See the [UNLICENSE](UNLICENSE) file for details.

## 🤝 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/hmatt1/mcp4mcp/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/hmatt1/mcp4mcp/discussions)
- 📚 **Examples**: See `examples/` directory

-----

**mcp4mcp** - Intelligence for MCP development 🧠✨