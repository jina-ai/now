import os as _os
import sys as _sys

__version__ = '0.0.4'
__resources_path__ = _os.path.join(
    _os.path.dirname(_sys.modules['now'].__file__), 'resources'
)
