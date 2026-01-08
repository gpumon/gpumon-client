from gpufl.analyzer import GpuFlightSession
import os

# Create dummy log files with expected names from existing stress logs if they don't exist
# or just use the stress logs by renaming/linking them if the analyzer supported it.
# However, the analyzer expects "gfl_block.log.kernel.0.log" etc.
# Let's try to copy them for the sample to work.

def prepare_logs():
    mapping = {
        "stress.kernel.0.log": "gfl_block.log.kernel.0.log",
        "stress.scope.0.log": "gfl_block.log.scope.0.log",
        "stress.system.0.log": "gfl_block.log.system.0.log"
    }
    for src, dst in mapping.items():
        if os.path.exists(src) and not os.path.exists(dst):
            import shutil
            shutil.copy(src, dst)

prepare_logs()

analyzer = GpuFlightSession("./")

analyzer.print_summary()

analyzer.inspect_scopes()

analyzer.inspect_hotspots()