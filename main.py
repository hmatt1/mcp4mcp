
"""
mcp4mcp - Meta MCP Server entry point
"""

import asyncio
import sys
from pathlib import Path

async def run_demo():
    """Run the demo"""
    try:
        from examples.demo_usage import main as demo_main
        await demo_main()
    except ImportError as e:
        print(f"Demo not available: {e}")
        print("Run 'python server.py' to start the MCP server")

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            print("Running mcp4mcp demo...")
            asyncio.run(run_demo())
            return
        elif sys.argv[1] == "server":
            print("Starting mcp4mcp server...")
            from server import mcp
            mcp.run()
            return
        elif sys.argv[1] == "test":
            print("Running tests...")
            import subprocess
            result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
            sys.exit(result.returncode)
    
    print("mcp4mcp - Meta MCP Server")
    print("========================")
    print()
    print("A Meta MCP Server that provides persistent memory and intelligent")
    print("guidance for MCP development projects.")
    print()
    print("Usage:")
    print("  python main.py server  - Start the FastMCP server")
    print("  python main.py demo    - Run usage demonstrations")
    print("  python main.py test    - Run the test suite")
    print()
    print("Direct usage:")
    print("  python server.py       - Start the server directly")
    print("  python examples/demo_usage.py - Run demo directly")
    print()
    print("Features:")
    print("- Project state management and tool tracking")
    print("- AI-powered analysis and suggestions")
    print("- Persistent SQLite storage")
    print("- Development session tracking")
    print("- Code scanning and similarity detection")
    print()
    print("All project data is stored in ~/.mcp4mcp/projects.db")

if __name__ == "__main__":
    main()
