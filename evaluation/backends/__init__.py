"""
Backend registry for the 3 models under comparison.

Imports are lazy -- deliberately. The three models have mutually awkward
dependency stacks (transformers / voxcpm / a fish-speech checkout), and a
missing `voxcpm` install must not stop you from running MMS. Importing this
package therefore costs nothing and requires nothing; the cost lands in
get_backend().
"""

BACKEND_KEYS = ("mms", "voxcpm2", "fish-s2")


def get_backend(name, **kwargs):
    """name -> a constructed (not yet loaded) Backend. Extra kwargs are passed
    through to the backend's __init__ (device, seed, ref_audio, fish_repo...)."""
    if name == "mms":
        from .mms import MmsBackend

        return MmsBackend(**kwargs)
    if name == "voxcpm2":
        from .voxcpm2 import VoxCpm2Backend

        return VoxCpm2Backend(**kwargs)
    if name == "fish-s2":
        from .fish_s2 import FishS2Backend

        return FishS2Backend(**kwargs)
    raise ValueError(
        f"unknown model {name!r}; expected one of {', '.join(BACKEND_KEYS)}"
    )
