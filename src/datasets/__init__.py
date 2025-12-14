"""Dataset package initialization and registration."""

from ..registry import register_dataset

from .brats19 import (
    BraTS19VolumeDataset,
)

# Import builders so they register themselves
from .brats19 import (
    Brats19SegBuilder,
    Brats19UEBuilder,
)

# Register dataset implementations with the unified registry
register_dataset('brats19_seg')(BraTS19VolumeDataset)
__all__ = [
    'BraTS19VolumeDataset',
]
