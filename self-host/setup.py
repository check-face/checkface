"""Acquire all verified model assets after explicit license acknowledgement."""
import subprocess
import sys

if '--accept-research-license' not in sys.argv[1:]:
    raise SystemExit('Read README licensing terms, then pass --accept-research-license for permitted local evaluation.')
for script in ('prepare_model.py', 'prepare_encoder.py'):
    subprocess.run([sys.executable, script, *sys.argv[1:]], check=True)
