#!/usr/bin/env python3
"""
Test runner script for mcp4mcp with better error reporting
"""

import sys
import subprocess
import os
from pathlib import Path

def run_tests():
    """Run tests with better error reporting"""
    
    # Ensure we're in the project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("Running mcp4mcp tests...")
    print("=" * 50)
    
    # Run specific test files one by one to isolate issues
    test_files = [
        "tests/test_models.py",
        "tests/test_storage.py", 
        "tests/test_tools.py",
        "tests/test_server.py"
    ]
    
    failed_tests = []
    
    for test_file in test_files:
        if not Path(test_file).exists():
            print(f"❌ Test file {test_file} not found")
            continue
            
        print(f"\n🧪 Running {test_file}...")
        print("-" * 30)
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                test_file, "-v", "--tb=short"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f"✅ {test_file} passed")
                # Show only summary for passed tests
                lines = result.stdout.split('\n')
                summary_lines = [line for line in lines if 'passed' in line and '::' not in line]
                if summary_lines:
                    print(f"   {summary_lines[-1]}")
            else:
                print(f"❌ {test_file} failed")
                failed_tests.append(test_file)
                print("STDOUT:")
                print(result.stdout)
                print("STDERR:")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_file} timed out")
            failed_tests.append(test_file)
        except Exception as e:
            print(f"💥 Error running {test_file}: {e}")
            failed_tests.append(test_file)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    
    if not failed_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"❌ {len(failed_tests)} test file(s) failed:")
        for test_file in failed_tests:
            print(f"   - {test_file}")
        
        print("\n💡 To debug individual tests, run:")
        print("   python -m pytest tests/test_specific.py::TestClass::test_method -v -s")
        
        return 1

def run_single_test():
    """Run a single test for debugging"""
    if len(sys.argv) < 2:
        print("Usage: python test_runner.py [test_file_or_pattern]")
        print("Example: python test_runner.py tests/test_models.py")
        print("Example: python test_runner.py tests/test_server.py::TestServerIntegration::test_server_creation")
        return 1
    
    test_target = sys.argv[1]
    
    print(f"🔍 Running specific test: {test_target}")
    print("-" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            test_target, "-v", "-s", "--tb=long"
        ], timeout=60)
        
        return result.returncode
        
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out")
        return 1
    except Exception as e:
        print(f"💥 Error running test: {e}")
        return 1

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        return run_single_test()
    else:
        return run_tests()

if __name__ == "__main__":
    sys.exit(main())