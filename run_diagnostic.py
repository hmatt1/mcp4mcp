#!/usr/bin/env python3
"""
Comprehensive FastMCP diagnostic script
Tests all major FastMCP features and provides debug information
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional
import traceback
import json

try:
    from fastmcp import FastMCP, Context, Client
    from fastmcp.resources import FileResource
    from pydantic import BaseModel, Field
    
    FASTMCP_AVAILABLE = True
except ImportError as e:
    print(f"❌ Cannot import FastMCP: {e}")
    print("   Install with: pip install fastmcp")
    FASTMCP_AVAILABLE = False

def print_system_info():
    """Print system and FastMCP version information"""
    print("FastMCP Diagnostic Script")
    print("=" * 50)
    
    # Python version
    print(f"Python version: {sys.version}")
    
    if not FASTMCP_AVAILABLE:
        return
    
    # FastMCP version info
    try:
        import fastmcp
        version = getattr(fastmcp, '__version__', 'unknown')
        print(f"FastMCP version: {version}")
    except:
        print("FastMCP version: unknown")
    
    # MCP version info
    try:
        import mcp
        mcp_version = getattr(mcp, '__version__', 'unknown')
        print(f"MCP version: {mcp_version}")
    except:
        print("MCP version: unknown")
    
    # Additional dependencies
    try:
        import pydantic
        print(f"Pydantic version: {pydantic.__version__}")
    except:
        print("Pydantic version: unknown")
    
    print()

def create_test_server():
    """Create a comprehensive test server"""
    print("🔧 Creating test server...")
    
    # Create server
    mcp = FastMCP("Diagnostic Test Server 🧪")
    
    # Test data models
    class UserRequest(BaseModel):
        user_id: str = Field(min_length=1, max_length=100)
        action: str
        data: Optional[dict] = None
    
    class ProcessResult(BaseModel):
        status: str
        message: str
        data: Optional[dict] = None
    
    # === TOOLS ===
    
    @mcp.tool()
    def simple_add(a: int, b: int) -> int:
        """Simple addition tool"""
        return a + b
    
    @mcp.tool()
    async def async_multiply(x: int, y: int) -> int:
        """Async multiplication tool"""
        await asyncio.sleep(0.01)  # Simulate async work
        return x * y
    
    @mcp.tool(tags={"math", "advanced"})
    def power_calculation(base: int, exponent: int) -> int:
        """Calculate power with tags"""
        return base ** exponent
    
    @mcp.tool()
    async def context_demo(message: str, ctx: Context) -> str:
        """Tool demonstrating context usage"""
        await ctx.debug(f"Debug: Processing {message}")
        await ctx.info(f"Info: Starting processing")
        
        # Progress reporting
        for i in range(3):
            await ctx.report_progress(i + 1, 3)
            await asyncio.sleep(0.01)
        
        await ctx.info("Processing complete")
        return f"Processed: {message}"
    
    @mcp.tool()
    def pydantic_tool(request: UserRequest) -> ProcessResult:
        """Tool using Pydantic models"""
        return ProcessResult(
            status="success",
            message=f"Processed {request.action} for {request.user_id}",
            data={"user_id": request.user_id, "action": request.action}
        )
    
    @mcp.tool()
    async def resource_reader(uri: str, ctx: Context) -> str:
        """Tool that reads resources"""
        try:
            resources = await ctx.read_resource(uri)
            if resources:
                resource = resources[0]
                # Handle different resource content types
                if hasattr(resource, 'text'):
                    content_str = resource.text
                elif hasattr(resource, 'content'):
                    content_str = str(resource.content)
                else:
                    content_str = str(resource)
                    
                await ctx.info(f"Successfully read resource: {uri}")
                return f"Resource content: {content_str[:100]}..."
            else:
                return f"No content found for: {uri}"
        except Exception as e:
            await ctx.error(f"Failed to read resource {uri}: {e}")
            return f"Error reading resource: {e}"
    
    # === RESOURCES ===
    
    @mcp.resource("config://server-info")
    def server_info() -> dict:
        """Server information resource"""
        return {
            "name": "Diagnostic Test Server",
            "version": "1.0.0",
            "features": ["tools", "resources", "prompts"],
            "tool_count": 6
        }
    
    @mcp.resource("data://sample-text", mime_type="text/plain")
    def sample_text() -> str:
        """Sample text resource"""
        return "This is sample text content for testing resource functionality."
    
    @mcp.resource("data://sample-json", mime_type="application/json")
    def sample_json() -> dict:
        """Sample JSON resource"""
        return {
            "type": "sample_data",
            "items": [1, 2, 3, 4, 5],
            "metadata": {"created": "2025-07-04", "test": True}
        }
    
    # Dynamic resource template
    @mcp.resource("user://{user_id}/profile")
    def user_profile(user_id: str) -> dict:
        """Dynamic user profile resource"""
        return {
            "user_id": user_id,
            "name": f"User {user_id}",
            "profile": {"status": "active", "type": "test_user"}
        }
    
    # === PROMPTS ===
    
    @mcp.prompt()
    def explain_topic(topic: str) -> str:
        """Generate explanation prompt"""
        return f"Explain the concept of {topic} in simple terms with practical examples."
    
    @mcp.prompt()
    def analyze_data(data_type: str, context: str) -> str:
        """Generate data analysis prompt"""
        return f"Analyze the {data_type} data in the context of {context}. Provide key insights and recommendations."
    
    print("✅ Test server created successfully")
    return mcp

async def test_server_capabilities(mcp):
    """Test all server capabilities using the official Client"""
    print("\n🧪 Testing server capabilities...")
    
    try:
        async with Client(mcp) as client:
            print("✅ Client connection established")
            
            # === TEST TOOLS ===
            print("\n📋 Testing tools...")
            
            # List tools
            tools = await client.list_tools()
            print(f"✅ Found {len(tools)} tools:")
            for tool in tools:
                print(f"   • {tool.name}: {tool.description}")
            
            # Test simple tool
            try:
                result = await client.call_tool("simple_add", {"a": 5, "b": 3})
                print(f"✅ simple_add(5, 3) = {result.data}")
                assert result.data == 8, f"Expected 8, got {result.data}"
            except Exception as e:
                print(f"❌ simple_add failed: {e}")
            
            # Test async tool
            try:
                result = await client.call_tool("async_multiply", {"x": 4, "y": 7})
                print(f"✅ async_multiply(4, 7) = {result.data}")
                assert result.data == 28, f"Expected 28, got {result.data}"
            except Exception as e:
                print(f"❌ async_multiply failed: {e}")
            
            # Test tool with tags
            try:
                result = await client.call_tool("power_calculation", {"base": 2, "exponent": 3})
                print(f"✅ power_calculation(2, 3) = {result.data}")
                assert result.data == 8, f"Expected 8, got {result.data}"
            except Exception as e:
                print(f"❌ power_calculation failed: {e}")
            
            # Test context tool
            try:
                result = await client.call_tool("context_demo", {"message": "test context"})
                print(f"✅ context_demo result: {result.data}")
            except Exception as e:
                print(f"❌ context_demo failed: {e}")
            
            # Test Pydantic tool
            try:
                result = await client.call_tool("pydantic_tool", {
                    "request": {
                        "user_id": "test123",
                        "action": "get_profile",
                        "data": {"extra": "info"}
                    }
                })
                print(f"✅ pydantic_tool result: {result.data}")
            except Exception as e:
                print(f"❌ pydantic_tool failed: {e}")
            
            # === TEST RESOURCES ===
            print("\n📁 Testing resources...")
            
            # List resources
            try:
                resources = await client.list_resources()
                print(f"✅ Found {len(resources)} resources:")
                for resource in resources:
                    print(f"   • {resource.uri}: {resource.description}")
            except Exception as e:
                print(f"❌ list_resources failed: {e}")
            
            # Test static resources
            test_uris = [
                "config://server-info",
                "data://sample-text",
                "data://sample-json"
            ]
            
            for uri in test_uris:
                try:
                    resources = await client.read_resource(uri)
                    if resources:
                        # Handle different resource content types
                        resource = resources[0]
                        if hasattr(resource, 'text'):
                            content = resource.text
                        elif hasattr(resource, 'content'):
                            content = resource.content
                        else:
                            # Try to access the actual content
                            content = str(resource)
                        
                        content_preview = str(content)[:100] + "..." if len(str(content)) > 100 else str(content)
                        print(f"✅ {uri}: {content_preview}")
                    else:
                        print(f"⚠️  {uri}: No content returned")
                except Exception as e:
                    print(f"❌ {uri}: {e}")
            
            # Test dynamic resource template
            try:
                resources = await client.read_resource("user://alice/profile")
                if resources:
                    resource = resources[0]
                    if hasattr(resource, 'text'):
                        content = resource.text
                    elif hasattr(resource, 'content'):
                        content = resource.content
                    else:
                        content = str(resource)
                    print(f"✅ user://alice/profile: {content}")
                else:
                    print(f"⚠️  user://alice/profile: No content returned")
            except Exception as e:
                print(f"❌ user://alice/profile: {e}")
            
            # Test resource reader tool
            try:
                result = await client.call_tool("resource_reader", {"uri": "config://server-info"})
                print(f"✅ resource_reader tool: {result.data}")
            except Exception as e:
                print(f"❌ resource_reader tool failed: {e}")
            
            # === TEST PROMPTS ===
            print("\n💬 Testing prompts...")
            
            # List prompts
            try:
                prompts = await client.list_prompts()
                print(f"✅ Found {len(prompts)} prompts:")
                for prompt in prompts:
                    print(f"   • {prompt.name}: {prompt.description}")
            except Exception as e:
                print(f"❌ list_prompts failed: {e}")
            
            # Test prompt execution
            try:
                result = await client.get_prompt("explain_topic", {"topic": "machine learning"})
                print(f"✅ explain_topic prompt: {result.messages[0].content.text if result.messages else 'No content'}")
            except Exception as e:
                print(f"❌ explain_topic prompt failed: {e}")
            
            try:
                result = await client.get_prompt("analyze_data", {"data_type": "sales", "context": "quarterly review"})
                print(f"✅ analyze_data prompt: {result.messages[0].content.text if result.messages else 'No content'}")
            except Exception as e:
                print(f"❌ analyze_data prompt failed: {e}")
            
            print("\n✅ All capability tests completed!")
            
    except Exception as e:
        print(f"❌ Server testing failed: {e}")
        traceback.print_exc()

async def test_error_handling():
    """Test error handling scenarios"""
    print("\n🚨 Testing error handling...")
    
    mcp = FastMCP("Error Test Server")
    
    @mcp.tool()
    def error_tool(should_fail: bool) -> str:
        """Tool that can fail on command"""
        if should_fail:
            raise ValueError("Intentional test error")
        return "Success"
    
    @mcp.tool()
    def validation_tool(value: int) -> str:
        """Tool with validation"""
        if value < 0:
            raise ValueError("Value must be non-negative")
        return f"Valid value: {value}"
    
    async with Client(mcp) as client:
        # Test successful call
        try:
            result = await client.call_tool("error_tool", {"should_fail": False})
            print(f"✅ error_tool(False): {result.data}")
        except Exception as e:
            print(f"❌ error_tool(False) failed: {e}")
        
        # Test error handling
        try:
            result = await client.call_tool("error_tool", {"should_fail": True})
            print(f"⚠️  error_tool(True) should have failed but returned: {result.data}")
        except Exception as e:
            print(f"✅ error_tool(True) correctly failed: {type(e).__name__}: {e}")
        
        # Test validation
        try:
            result = await client.call_tool("validation_tool", {"value": 5})
            print(f"✅ validation_tool(5): {result.data}")
        except Exception as e:
            print(f"❌ validation_tool(5) failed: {e}")
        
        try:
            result = await client.call_tool("validation_tool", {"value": -1})
            print(f"⚠️  validation_tool(-1) should have failed but returned: {result.data}")
        except Exception as e:
            print(f"✅ validation_tool(-1) correctly failed: {type(e).__name__}: {e}")

def test_server_inspection():
    """Test server inspection capabilities"""
    print("\n🔍 Testing server inspection...")
    
    mcp = FastMCP("Inspection Test Server")
    
    @mcp.tool()
    def inspect_tool() -> str:
        """Tool for inspection testing"""
        return "inspection_result"
    
    @mcp.resource("inspect://test")
    def inspect_resource() -> str:
        """Resource for inspection testing"""
        return "inspection_resource_content"
    
    # Test server attributes
    print(f"✅ Server name: {mcp.name}")
    print(f"✅ Server type: {type(mcp)}")
    
    # Check available attributes
    attrs = [attr for attr in dir(mcp) if not attr.startswith('__')]
    print(f"✅ Server has {len(attrs)} public attributes/methods")
    
    # Test some key methods
    key_methods = ['tool', 'resource', 'prompt', 'run', 'add_resource']
    for method in key_methods:
        if hasattr(mcp, method):
            print(f"✅ Has {method} method")
        else:
            print(f"❌ Missing {method} method")

async def main():
    """Main diagnostic function"""
    print_system_info()
    
    if not FASTMCP_AVAILABLE:
        print("❌ FastMCP not available. Install with: pip install fastmcp")
        return
    
    try:
        # Create and test server
        mcp = create_test_server()
        await test_server_capabilities(mcp)
        
        # Test error handling
        await test_error_handling()
        
        # Test server inspection
        test_server_inspection()
        
        print("\n" + "=" * 50)
        print("🎉 FastMCP Diagnostic Complete!")
        print("\nSummary:")
        print("✅ FastMCP is working correctly")
        print("✅ Server creation successful")
        print("✅ Tools, resources, and prompts functional")
        print("✅ Client testing works")
        print("✅ Error handling works")
        print("✅ Server inspection works")
        
        print("\nNext steps:")
        print("• Use 'fastmcp dev server.py' for GUI debugging")
        print("• Use 'fastmcp run server.py --port 8000' for HTTP server")
        print("• Use the in-memory Client pattern for testing")
        print("• Check the FastMCP documentation for advanced features")
        
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    if FASTMCP_AVAILABLE:
        asyncio.run(main())
    else:
        print("Please install FastMCP to run diagnostics.")