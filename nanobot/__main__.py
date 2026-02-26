"""
Entry point for running nanobot as a module: python -m nanobot
"""

from nanobot.cli.commands import app

import warnings
warnings.filterwarnings("ignore", category=RequestsDependencyWarning)

if __name__ == "__main__":
    app()
