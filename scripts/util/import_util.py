def script_imports(allow_zluda: bool = True):
    import logging
    import os
    import sys
    from pathlib import Path

    # Filter out the Triton warning on startup.
    # xformers is not installed anymore, but might still exist for some installations.
    logging \
        .getLogger("xformers") \
        .addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

    # Insert ourselves as the highest-priority library path, so our modules are
    # always found without any risk of being shadowed by another import path.
    # 3 .parent calls to navigate from /scripts/util/import_util.py to the main directory
    onetrainer_lib_path = Path(__file__).absolute().parent.parent.parent
    sys.path.insert(0, str(onetrainer_lib_path))

    if allow_zluda and sys.platform.startswith('win'):
        from modules.zluda import ROCm, ZLUDAInstaller

        if os.path.exists(ZLUDAInstaller.path):
            amd_gpus = ROCm.get_agents()
            if len(amd_gpus) == 0:
                print('No AMD GPU found, skipping ZLUDA load')
                return
            index = 0
            for idx, gpu in enumerate(amd_gpus):
                index = idx
                if not gpu.is_apu:
                    # although apu was found, there can be a dedicated card. do not break loop.
                    # if no dedicated card was found, apu will be used.
                    break
            try:
                ZLUDAInstaller.set_default_agent(amd_gpus[index])
                ZLUDAInstaller.load()
                print(f'Using ZLUDA in {ZLUDAInstaller.path}')
            except Exception as e:
                print(f'Failed to load ZLUDA: {e}')

            from modules.zluda import ZLUDA

            ZLUDA.initialize()
