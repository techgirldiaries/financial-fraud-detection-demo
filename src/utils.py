"""
Utility functions, e.g., Windows runtime debug.
"""

import os
import sys
import ctypes

def check_windows_runtime_samples():
    if os.name != 'nt':
        print('Not on Windows; skipping runtime check.')
        return
    libraries = ['vcruntime140_1.dll', 'vcruntime140.dll', 'api-ms-win-crt-runtime-l1-1-0.dll']
    print('Python executable:', sys.executable)
    print('PATH length:', len(os.environ.get('PATH', '')))
    print('Checking common MSVC runtime DLLs...')
    for l in libraries:
        try:
            ctypes.WinDLL(l)
            print(f'{l} OK')
        except Exception as e:
            print(f'{l} MISSING: {e}')