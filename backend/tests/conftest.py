import sys
from pathlib import Path

# pytest runs from backend/, but doesn't add it to sys.path itself.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
