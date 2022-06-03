__version__ = '0.0.11'

import os as _os
import sys as _sys

__resources_path__ = _os.path.join(
    _os.path.dirname(_sys.modules['now'].__file__), 'resources'
)
