#!/usr/bin/env python3
"""
Diagnostic script to understand FastMCP API
"""

try:
    from fastmcp import FastMCP
    
    print("FastMCP API Diagnostic")
    print("=" * 30)
    
    # Create a test server
    mcp = FastMCP("diagnostic-server")
    
    # Register a simple test tool
    @mcp.tool()
    def test_tool() -> str:
        """A simple test tool"""
        return "test"
    
    print(f"✅ Created FastMCP server: {mcp.name}")
    
    # Inspect the server object
    print(f"\n🔍 Server attributes:")
    attrs = [attr for attr in dir(mcp) if not attr.startswith('__')]
    for attr in sorted(attrs):
        try:
            value = getattr(mcp, attr)
            if callable(value):
                print(f"   {attr}() - method")
            else:
                print(f"   {attr} = {type(value).__name__}")
        except:
            print(f"   {attr} - (error accessing)")
    
    # Try different ways to access tools
    print(f"\n🔧 Attempting to access tools:")
    
    # Method 1: Direct tools attribute
    try:
        tools = mcp.tools
        print(f"   ✅ mcp.tools: {type(tools)} with {len(tools)} items")
        if hasattr(tools, 'keys'):
            print(f"      Tool names: {list(tools.keys())}")
    except AttributeError:
        print(f"   ❌ mcp.tools: not available")
    except Exception as e:
        print(f"   ❌ mcp.tools: error - {e}")
    
    # Method 2: _tools attribute
    try:
        tools = mcp._tools
        print(f"   ✅ mcp._tools: {type(tools)} with {len(tools)} items")
        if hasattr(tools, 'keys'):
            print(f"      Tool names: {list(tools.keys())}")
    except AttributeError:
        print(f"   ❌ mcp._tools: not available")
    except Exception as e:
        print(f"   ❌ mcp._tools: error - {e}")
    
    # Method 3: list_tools method
    try:
        tools_list = mcp.list_tools()
        print(f"   ✅ mcp.list_tools(): {type(tools_list)} with {len(tools_list)} items")
        if tools_list:
            print(f"      First tool: {tools_list[0]}")
    except AttributeError:
        print(f"   ❌ mcp.list_tools(): not available")
    except Exception as e:
        print(f"   ❌ mcp.list_tools(): error - {e}")
    
    # Method 4: get_tools method
    try:
        tools = mcp.get_tools()
        print(f"   ✅ mcp.get_tools(): {type(tools)} with {len(tools)} items")
    except AttributeError:
        print(f"   ❌ mcp.get_tools(): not available")
    except Exception as e:
        print(f"   ❌ mcp.get_tools(): error - {e}")
    
    # Method 5: Check if tools are stored in registry
    try:
        if hasattr(mcp, 'registry'):
            registry = mcp.registry
            print(f"   ✅ mcp.registry: {type(registry)}")
            if hasattr(registry, 'tools'):
                tools = registry.tools
                print(f"      registry.tools: {type(tools)} with {len(tools)} items")
    except Exception as e:
        print(f"   ❌ mcp.registry: error - {e}")
    
    print(f"\n📋 Summary:")
    print(f"   FastMCP version: {getattr(FastMCP, '__version__', 'unknown')}")
    print(f"   Server created successfully: {mcp.name}")
    print(f"   Tool registration appears to work (no errors)")
    
    # Try to find the actual way to access tools
    print(f"\n🎯 Recommended test approach:")
    print(f"   1. Import tool functions directly for testing")
    print(f"   2. Test FastMCP integration separately")
    print(f"   3. Use mcp.run() only for actual server execution")

except ImportError as e:
    print(f"❌ Cannot import FastMCP: {e}")
    print("   Make sure fastmcp is installed: pip install fastmcp")
except Exception as e:
    print(f"❌ Error with FastMCP: {e}")
    import traceback
    traceback.print_exc()