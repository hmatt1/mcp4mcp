
"""
mcp4mcp - Meta MCP Server entry point
"""

from fastmcp import FastMCP
from mcp4mcp.tools.state_management import register_state_tools
from mcp4mcp.tools.intelligence import register_intelligence_tools
from mcp4mcp.tools.tracking import register_tracking_tools

# Create FastMCP server
mcp = FastMCP("mcp4mcp - Meta MCP Development Server")

# Register all tool modules
register_state_tools(mcp)
register_intelligence_tools(mcp)
register_tracking_tools(mcp)

if __name__ == "__main__":
    print("Starting mcp4mcp server...")
    print("Available tools:")
    print("- get_project_state_tool: Load current project state")
    print("- update_project_state_tool: Update project information") 
    print("- scan_project_files_tool: Scan files for MCP tools")
    print("- update_tool_status_tool: Update tool status")
    print("- check_before_build_tool: Check for duplicates before building")
    print("- suggest_next_action_tool: Get AI-powered suggestions")
    print("- analyze_tool_similarity_tool: Analyze tool similarity")
    print("- track_development_session_tool: Log development activities")
    print("- end_development_session_tool: End development session")
    print("- get_development_sessions_tool: Get recent sessions")
    print("- get_session_analytics_tool: Get development analytics")
    print("\nStarting FastMCP server...")
    mcp.run()
