import cProfile
import subprocess

def run(script, output):
    cProfile.run(f"exec(open('{script}.py').read())", filename=f"profilingData/{output}.dat")
    subprocess.run(['snakeviz', f"profilingData/{output}.dat"])

# Usage example -- in new python file
# from qutils.qProfile import run
# run('mambaCR3BP','profiledCR3BPMamba')