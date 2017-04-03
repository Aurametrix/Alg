import numpy as np
import types
import sys
import inspect
import pdb


# we import * so that the symbol X can be accessed as faiss.X

try:
    from swigfaiss_gpu import *
except ImportError as e:
    if e.args[0] != 'ImportError: No module named swigfaiss_gpu':
        # swigfaiss_gpu is there but failed to load: Warn user about it.
        sys.stderr.write("Failed to load GPU Faiss: %s\n" % e.args[0])
        sys.stderr.write("Faiss falling back to CPU-only.\n")
    from swigfaiss import *
