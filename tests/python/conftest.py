import pytest
import json
import os
from pathlib import Path

@pytest.fixture
def mock_log_dir(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    
    prefix = "test_run"
    
    # Create mock .kernel.log
    kernel_log = log_dir / f"{prefix}.kernel.0.log"
    kernel_events = [
        {
            "type": "kernel", "name": "vectorAdd", "device_id": 0, "app": "test_app",
            "start_ns": 1000, "end_ns": 2000, "api_start_ns": 900, "api_exit_ns": 2100,
            "grid": "(1,1,1)", "block": "(256,1,1)", "occupancy": 0.8,
            "user_scope": "global|main_loop|vectorAdd",
            "dyn_shared_bytes": 0, "static_shared_bytes": 100, "num_regs": 32,
            "local_bytes": 0, "const_bytes": 0
        },
        {
            "type": "kernel", "name": "matrixMul", "device_id": 0, "app": "test_app",
            "start_ns": 3000, "end_ns": 5000, "api_start_ns": 2900, "api_exit_ns": 5100,
            "grid": "(1,1,1)", "block": "(16,16,1)", "occupancy": 0.6,
            "user_scope": "global|main_loop|matrixMul",
            "dyn_shared_bytes": 1024, "static_shared_bytes": 200, "num_regs": 64,
            "local_bytes": 0, "const_bytes": 128
        }
    ]
    with open(kernel_log, "w") as f:
        for ev in kernel_events:
            f.write(json.dumps(ev) + "\n")

    # Create mock .scope.log
    scope_log = log_dir / f"{prefix}.scope.0.log"
    scope_events = [
        {"type": "scope_begin", "name": "main_loop", "ts_ns": 500, "device_id": 0, "app": "test_app"},
        {"type": "scope_end", "name": "main_loop", "ts_ns": 6000, "device_id": 0, "app": "test_app"}
    ]
    with open(scope_log, "w") as f:
        for ev in scope_events:
            f.write(json.dumps(ev) + "\n")

    # Create mock .system.log
    system_log = log_dir / f"{prefix}.system.0.log"
    system_events = [
        {
            "type": "system_sample", "ts_ns": 1000, "app": "test_app",
            "devices": [{"device_id": 0, "util_gpu": 50, "used_mib": 1024, "total_mib": 8192}]
        },
        {
            "type": "system_sample", "ts_ns": 4000, "app": "test_app",
            "devices": [{"device_id": 0, "util_gpu": 80, "used_mib": 2048, "total_mib": 8192}]
        }
    ]
    with open(system_log, "w") as f:
        for ev in system_events:
            f.write(json.dumps(ev) + "\n")
            
    return str(log_dir), prefix
