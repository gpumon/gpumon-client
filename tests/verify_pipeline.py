import gpufl as gfl
import os
import time
import tempfile
import shutil
import glob
import json

def test_pipeline():
    print("--- Starting GPUFL Multi-Log Pipeline Verification ---")
    temp_dir = tempfile.mkdtemp()
    log_base_name = "ci_test"
    log_base_path = os.path.join(temp_dir, log_base_name)
    print(f"1. Log path set to: {log_base_path}")

    keep = False
    try:
        print("2. Initializing GPUFL...")
        # This should trigger "init" which is broadcast to kernel, scope, and system logs
        gfl.init("CI_Test_App", log_base_path, 0)

        print("3. Running Scope...")
        # This should write only to the scope log
        with gfl.Scope("ci_scope_01", "test_tag"):
            time.sleep(0.01)

        print("4. Shutting down...")
        # This triggers "shutdown", broadcast to all logs
        gfl.shutdown()

        print("5. Verifying Log Files...")
        files = sorted(os.listdir(temp_dir))
        print(f"   Files found in {temp_dir}:")
        for f in files:
            print(f"    - {f}")

        if not files:
            print("FAILED: No log files were created at all.")
            keep = True
            raise SystemExit(1)

        # distinct categories expected from the C++ logger
        categories = ["kernel", "scope", "system"]

        # We expect files named like: ci_test.kernel.0.log
        for cat in categories:
            expected_name = f"{log_base_name}.{cat}.0.log"
            full_path = os.path.join(temp_dir, expected_name)

            if not os.path.exists(full_path):
                print(f"FAILED: Expected log file missing: {expected_name}")
                keep = True
                raise SystemExit(1)

            print(f"   [OK] Found {cat} log: {expected_name}")

            # 6. Verify Content
            with open(full_path, 'r') as f:
                lines = f.readlines()

            # Every file should have at least init and shutdown
            has_init = any('"type":"init"' in line for line in lines)
            has_shutdown = any('"type":"shutdown"' in line for line in lines)

            if not (has_init and has_shutdown):
                print(f"FAILED: {cat} log missing lifecycle events (init/shutdown).")
                keep = True
                raise SystemExit(1)

            # Specific check for scope log
            if cat == "scope":
                has_scope = any('"type":"scope_begin"' in line for line in lines)
                if has_scope:
                    print("   [OK] Scope events found in scope log.")
                else:
                    print("FAILED: No scope events found in scope log.")
                    keep = True
                    raise SystemExit(1)

        print("\nSUCCESS: All log files created and content verified.")

    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        keep = True
        raise

    finally:
        if keep:
            print(f"Keeping temp dir for inspection: {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_pipeline()