from util.import_util import script_imports

script_imports(allow_zluda=False)

import sys

from modules.zluda import ZLUDAInstaller

if __name__ == '__main__':
    try:
        ZLUDAInstaller.install()
    except Exception as e:
        print(f'Failed to install ZLUDA: {e}')
        sys.exit(1)

    print(f'ZLUDA installed: {ZLUDAInstaller.path}')
