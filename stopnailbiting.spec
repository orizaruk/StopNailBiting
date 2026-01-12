# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for StopNailBiting

import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# Get the project root directory
project_root = os.path.dirname(os.path.abspath(SPEC))

# Collect Shapely's dynamic libraries (geos_c.dll, etc.)
shapely_binaries = collect_dynamic_libs('shapely')

# Collect MediaPipe data files (required for model loading)
mediapipe_datas = collect_data_files('mediapipe')

a = Analysis(
    ['main.py'],
    pathex=[project_root],
    binaries=shapely_binaries,
    datas=[
        # Bundle the MediaPipe model files
        ('models/hand_landmarker.task', 'models'),
        ('models/face_landmarker.task', 'models'),
        # Bundle the sound file
        ('assets/noise.wav', 'assets'),
    ] + mediapipe_datas,
    hiddenimports=[
        'mediapipe',
        'mediapipe.python',
        'mediapipe.python.solutions',
        'cv2',
        'shapely',
        'shapely.geometry',
        'pygame',
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='StopNailBiting',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window on Windows
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='resources/icon.ico',  # Uncomment when icon is added
)
