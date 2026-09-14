"""Exercise HTTP and real inference locally; does not prove the container build."""
from pathlib import Path
import subprocess
import sys
import threading
from api import BoundedHTTPServer, handler, load_model

server = BoundedHTTPServer(('127.0.0.1', 0), handler(load_model()))
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('smoke.py')),
                             f'http://127.0.0.1:{server.server_port}'], check=False)
finally:
    server.shutdown()
    server.server_close()
raise SystemExit(result.returncode)
