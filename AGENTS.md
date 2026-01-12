# Repository Guidelines

## Project Structure & Module Organization
- `main.py` runs the live webcam detection loop and alert system.
- `landmark_image_maker.py` visualizes hand/face landmark indices for debugging.
- `models/` contains MediaPipe task files (`hand_landmarker.task`, `face_landmarker.task`).
- `assets/` is used for local test images and alert audio (commonly gitignored).
- `requirements.txt` tracks Python dependencies.

## Build, Test, and Development Commands
Set up a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Run the main app (webcam detection + alerts):
```bash
python main.py
```
Debug landmark indices:
```bash
python landmark_image_maker.py
```
There is no separate build step; scripts are run directly with Python.

## Coding Style & Naming Conventions
- Python style: 4-space indentation, PEP 8 spacing, and keep lines reasonably short.
- Use `snake_case` for functions/variables and `UPPER_SNAKE_CASE` for constants (see config values in `main.py`).
- No formatter or linter is configured; keep changes minimal and consistent with surrounding code.

## Testing Guidelines
- No automated test suite is present.
- Validate changes by running `python main.py` with a webcam and confirm alerts behave as expected.
- If adding tests, document how to run them in this file and prefer `tests/` for placement.

## Commit & Pull Request Guidelines
- Recent commit messages are short and sometimes informal; no strict convention is enforced.
- Preferred practice: concise, imperative messages that describe the change (e.g., "Improve z-depth filter").
- PRs should include: a brief summary, manual test steps, and notes on any detection behavior changes. Screenshots or short clips help when modifying the alert UI.

## Configuration Notes
- Detection sensitivity, depth thresholds, and cooldown timing live near the top of `main.py`.
- Keep alert audio in `assets/` and avoid committing large media files unless necessary.
