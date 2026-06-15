# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for StopNailBiting

import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Get the project root directory
project_root = os.path.dirname(os.path.abspath(SPEC))

# Collect MediaPipe data files (required for model loading)
mediapipe_datas = collect_data_files('mediapipe')

a = Analysis(
    ['main.py'],
    pathex=[project_root],
    binaries=[],
    datas=[
        # Bundle the MediaPipe model files
        ('models/hand_landmarker.task', 'models'),
        ('models/face_landmarker.task', 'models'),
        ('models/efficientdet_lite0.tflite', 'models'),
        # Bundle the sound file
        ('assets/noise.wav', 'assets'),
    ] + mediapipe_datas,
    hiddenimports=[
        'mediapipe',
        'mediapipe.python',
        'mediapipe.python.solutions',
        'cv2',
        'miniaudio',
        'numpy',
        'screeninfo',
        # WinRT for media control
        'winrt',
        'winrt.windows.media.control',
        'winrt.windows.foundation',
        'winrt.windows.foundation.collections',
        # CoreAudio fallback for browser audio muting
        'pycaw',
        'pycaw.pycaw',
        'comtypes',
        'comtypes.client',
        'psutil',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Exclude removed/unused heavy packages so PyInstaller never bundles them
    # even if they linger in the build venv. shapely (GEOS) and pygame (SDL)
    # were replaced; the rest are common transitive bloat the app never imports.
    excludes=[
        'shapely',
        'pygame',
        'matplotlib',
        'scipy',
        'jax',
        'jaxlib',
        'pandas',
    ],
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
