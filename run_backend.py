import os
import sys
import subprocess
import signal
import time

def main():
    # Change to backend directory
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    os.chdir(backend_dir)
    
    # Add backend directory to Python path
    sys.path.insert(0, backend_dir)
    
    # Run uvicorn with proper signal handling
    try:
        subprocess.run([
            sys.executable, '-m', 'uvicorn', 
            'main:app', 
            '--reload', 
            '--host', '0.0.0.0', 
            '--port', '8000'
        ], check=True)
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        time.sleep(1)  # Give time for cleanup
    except subprocess.CalledProcessError as e:
        print(f"Server exited with error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
