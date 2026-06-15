# -*- coding: utf-8 -*-
"""Daily 6am NZ job: regenerate the Simpleton Summary's plain-English
'what changed in the last 24 hours' brief. Registered as Crypto_simpleton_brief_daily.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.simpleton_daily_brief import main  # noqa: E402

if __name__ == "__main__":
    main()
