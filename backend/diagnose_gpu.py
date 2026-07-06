import os
import time
import shutil
import multiprocessing

def run_diagnostics():
    print("==================================================")
    print("       CONTAINER STARTUP DIAGNOSTICS              ")
    print("==================================================")
    
    # 1. Check Python & PyTorch environment
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available in PyTorch: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Device Count: {torch.cuda.device_count()}")
            
            # Measure CUDA initialization latency
            start_time = time.time()
            try:
                # Force CUDA context initialization
                _ = torch.zeros(1).cuda()
                init_time = time.time() - start_time
                print(f"CUDA Context Init Time: {init_time:.3f} seconds")
                if init_time > 5.0:
                    print("NOTE: CUDA context initialization took > 5s. This is common on cold starts but can indicate driver/runtime overhead.")
            except Exception as e:
                print(f"ERROR: Failed to initialize CUDA context: {e}")
        else:
            print("WARNING: CUDA is NOT available to PyTorch. Operations will run on CPU.")
            
        # 1b. Check Flash Attention availability
        try:
            import flash_attn
            print(f"Flash Attention: Installed (Version {flash_attn.__version__})")
        except ImportError:
            print("Flash Attention: NOT installed")
        except Exception as e:
            print(f"Flash Attention: Import/initialization failed ({e})")
            
        print(f"PyTorch Thread Count (Intraop): {torch.get_num_threads()}")
    except ImportError:
        print("ERROR: PyTorch is not installed in this environment.")
        return

    # 2. Check System Resources & Threading Environment Variables
    cores = multiprocessing.cpu_count()
    print(f"Available CPU Cores: {cores}")
    
    omp_threads = os.environ.get('OMP_NUM_THREADS')
    mkl_threads = os.environ.get('MKL_NUM_THREADS')
    print(f"OMP_NUM_THREADS: {omp_threads if omp_threads else 'Not Set (Defaults to all cores)'}")
    print(f"MKL_NUM_THREADS: {mkl_threads if mkl_threads else 'Not Set (Defaults to all cores)'}")
    
    if (not omp_threads or int(omp_threads) > 8) and cores > 8:
        print("NOTE: High CPU core count detected with high/unset OMP_NUM_THREADS.")
        print("      This can slow down model loading/deserialization.")
        print("      Consider setting OMP_NUM_THREADS to a lower value (e.g., 4 or 8) in your container environment.")

    # 3. Check Shared Memory limit (/dev/shm)
    if os.path.exists('/dev/shm'):
        try:
            shm_stats = shutil.disk_usage('/dev/shm')
            shm_total_gb = shm_stats.total / (1024 ** 3)
            print(f"Shared Memory (/dev/shm) Total: {shm_total_gb:.2f} GB")
            if shm_total_gb < 0.5:
                print("WARNING: Shared memory (/dev/shm) is very low (< 512MB).")
                print("         PyTorch multiprocessing (DataLoader) will fail or be extremely slow.")
                print("         Please run the docker container with '--ipc=host' or '--shm-size=8g'.")
        except Exception as e:
            print(f"Could not read /dev/shm details: {e}")
    else:
        print("/dev/shm does not exist or is not accessible.")
        
    print("==================================================\n")

if __name__ == "__main__":
    run_diagnostics()
