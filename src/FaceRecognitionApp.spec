# -*- mode: python ; coding: utf-8 -*-

import PyInstaller.utils.hooks
import mediapipe
import os

# Get mediapipe data files
mediapipe_datas = PyInstaller.utils.hooks.collect_data_files('mediapipe', include_py_files=False)


a = Analysis(
    ['main.py'], # 指向 src 下的 main.py
    pathex=['.'], # 當前目錄 (src)
    binaries=[],
    datas=[
        # Add mediapipe data files
        *mediapipe_datas
    ],
    hiddenimports=[
        'encodings', # 加入這個
        'scipy._lib.array_api_compat.numpy.fft', # <--- 加入這個缺少的模組
        # Add hidden imports if needed based on runtime errors
        # 'charset_normalizer.md__mypyc', # Example
        # 'scipy.special._cdflib',      # Example
    ],
    hookspath=['.'], # <--- 將此變更為指向目前目錄
    hooksconfig={},
    runtime_hooks=['rthook_mediapipe.py'], # <--- 移除 'hooks/' 前置詞
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FaceRecognitionApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # 保持 True 以便繼續除錯
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas, # This now includes mediapipe_datas
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FaceRecognitionApp',
)
