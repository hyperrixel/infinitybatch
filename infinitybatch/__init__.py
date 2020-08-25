"""
        _                 _
       (_)               | |
 _ __   _  __  __   ___  | |
| '__| | | \ \/ /  / _ \ | |
| |    | |  >  <  |  __/ | |
|_|    |_| /_/\_\  \___| |_|

infinitybatch
-------------

Copyright rixel 2020
Distributed under the MIT License.
See accompanying file LICENSE.
"""



# Modul level constants.
__author__ = 'rixel'
__copyright__ = "Copyright 2020, infinitybatch"
__credits__ = ['rixel']
__license__ = 'MIT'
__version__ = '1.0.0'
__status__ = 'Production'



from importlib import import_module
from ._content import InfinityBatch, InfinityBatchError, InfinityBatchWarning

import_module('.tools', 'infinitybatch')
__all__ = ['InfinityBatch', 'InfinityBatchError', 'InfinityBatchWarning', 'tools']
