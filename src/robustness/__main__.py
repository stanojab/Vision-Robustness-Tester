"""Allow ``python -m robustness ...`` as an alternative to the ``robustness`` command."""

import sys

from robustness.cli import main

sys.exit(main())
