from pathlib import Path

from dotenv import load_dotenv

_PACKAGE_DIR = Path(__file__).resolve().parent

load_dotenv(_PACKAGE_DIR.parent / ".env")
load_dotenv(_PACKAGE_DIR / ".env", override=True)
