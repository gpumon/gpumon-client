import gpufl as gfl
import os
import json
import time
import tempfile
import shutil


def ls_dir(d):
    try:
        return sorted(os.listdir(d))
    except Exception as e:
        return [f"<ls failed: {e}>"]
def test_pipeline():
    print("--- Starting GPUFL Pipeline Verification ---")

    temp_dir = tempfile.mkdtemp()
    log_base = os.path.join(temp_dir, "ci_test")
    print(f"1. Log path set to: {log_base}")
    print("   temp_dir contents (start):", ls_dir(temp_dir))

    keep_dir = False
    try:
        print("2. Initializing GPUFL...")
        ok = gfl.init("CI_Test_App", log_base, 0)
        print("   gfl.init returned:", ok)
        print("   temp_dir contents (after init):", ls_dir(temp_dir))

        if not ok:
            print("FAILED: gfl.init() returned False on this runner.")
            keep_dir = True
            raise SystemExit(1)

        print("3. Running Scope...")
        with gfl.Scope("ci_scope_01", "test_tag"):
            time.sleep(0.05)

        print("   temp_dir contents (after scope):", ls_dir(temp_dir))

        print("4. Shutting down...")
        gfl.shutdown()
        print("   temp_dir contents (after shutdown):", ls_dir(temp_dir))

        print("5. Verifying Log Files...")

        candidates = [
            f"{log_base}.scope.log",
            f"{log_base}.0.log",
        ]
        found = [p for p in candidates if os.path.exists(p)]

        if not found:
            print("FAILED: Missing scope log file. Tried:")
            for p in candidates:
                print("  -", p)
            print("Files in temp_dir:")
            for f in ls_dir(temp_dir):
                print("  -", f)
            keep_dir = True
            raise SystemExit(1)

        print("Found scope log:", found[0])

    except SystemExit:
        raise
    except Exception:
        print("FAILED with exception:")
        traceback.print_exc()
        keep_dir = True
        raise
    finally:
        if keep_dir:
            print("Keeping temp dir for inspection:", temp_dir)
        else:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_pipeline()