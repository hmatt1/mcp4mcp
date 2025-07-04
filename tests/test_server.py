"""
Tests for mcp4mcp server integration
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from fastmcp import FastMCP
from mcp4mcp.tools.state_management import register_state_tools
from mcp4mcp.tools.intelligence import register_intelligence_tools
from mcp4mcp.tools.tracking import register_tracking_tools


class TestServerIntegration:
    """Test server integration"""
    
    def setup_method(self):
        """Setup test server"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        # Override the global DB_PATH for testing
        import mcp4mcp.storage
        self.original_db_path = mcp4mcp.storage.DB_PATH
        mcp4mcp.storage.DB_PATH = self.db_path
        
        # Create test server
        self.mcp = FastMCP("test-mcp4mcp")
        register_state_tools(self.mcp)
        register_intelligence_tools(self.mcp)
        register_tracking_tools(self.mcp)
    
    def teardown_method(self):
        """Cleanup test environment"""
        # Restore original DB_PATH
        import mcp4mcp.storage
        mcp4mcp.storage.DB_PATH = self.original_db_path
        
        # Remove test database
        if self.db_path.exists():
            os.remove(self.db_path)
        
        # Clean up temp directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_server_creation(self):
        """Test that server is created with tools"""
        assert self.mcp.name == "test-mcp4mcp"
        # Check if server has tools registered using the correct FastMCP API
        assert hasattr(self.mcp, 'tools')
        assert len(self.mcp.tools) > 0
    
    def test_tool_registration(self):
        """Test that all expected tools are registered"""
        tool_names = list(self.mcp.tools.keys())
        
        # State management tools
        assert "get_project_state_tool" in tool_names
        assert "update_project_state_tool" in tool_names
        assert "scan_project_files_tool" in tool_names
        
        # Intelligence tools
        assert "check_before_build_tool" in tool_names
        assert "suggest_next_action_tool" in tool_names
        assert "analyze_tool_similarity_tool" in tool_names
        
        # Tracking tools
        assert "track_development_session_tool" in tool_names
        assert "end_development_session_tool" in tool_names
    
    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test that tools can be executed"""
        # Find the get_project_state_tool
        assert "get_project_state_tool" in self.mcp.tools
        
        get_project_tool = self.mcp.tools["get_project_state_tool"]
        
        # Execute the tool using the correct FastMCP API
        result = await get_project_tool(project_name="test_project")
        
        assert result["success"] is True
        assert "project" in result
        assert result["project"]["name"] == "test_project"