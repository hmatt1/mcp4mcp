import os
from pathlib import Path

def get_python_files(root_dir, script_name):
    python_files = []
    skip_dirs = {'__pycache__', '.git', '.pytest_cache', '.mypy_cache', 'node_modules', '.venv', 'venv', 'env', 'build', 'dist'}
    
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
        
        for file in files:
            if file.endswith('.py') and file != script_name:
                python_files.append(Path(root) / file)
    
    return sorted(python_files)

def read_file_safe(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except:
            return '# Error reading file'

def main():
    script_name = Path(__file__).name
    current_dir = Path.cwd()
    output_file = current_dir / 'codebase.md'
    
    python_files = get_python_files(current_dir, script_name)
    
    if not python_files:
        print('No Python files found.')
        return
    
    print(f'Found {len(python_files)} Python files')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('# Python Codebase Analysis\n\n')
        f.write(f'Total files: {len(python_files)}\n\n')
        
        for file_path in python_files:
            relative_path = file_path.relative_to(current_dir)
            content = read_file_safe(file_path)
            
            f.write(f'## {relative_path}\n\n')
            f.write('```python\n')
            f.write(content)
            f.write('\n```\n\n')
    
    print(f'Created: {output_file}')

if __name__ == '__main__':
    main()