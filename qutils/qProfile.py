import cProfile
import subprocess

def runPerformanceProfiling(script, output):
    cProfile.run(f"exec(open('{script}.py').read())", filename=f"profilingData/{output}.dat")
    subprocess.run(['snakeviz', f"profilingData/{output}.dat"])

def runMemoryProfiling(script):
    # Construct the command to run the Python script with memory_profiler
    command = ['python', '-m', 'memory_profiler', script]

    # Run the command
    result = subprocess.run(command, text=True, capture_output=True)

    # Print the output from memory_profiler
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)

# Usage example -- in new python file
# from qutils.qProfile import runPerformanceProfiling
# runPerformanceProfiling('mambaCR3BP','profiledCR3BPMamba')