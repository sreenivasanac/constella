import numpy as np
import pytest


@pytest.fixture
def blobs():
    """Three well-separated topics of 40 items each, as unit-ish float32 vectors with ids."""
    rng = np.random.default_rng(0)
    centres = np.eye(3, 16, dtype=np.float32)
    vectors = np.concatenate([c + 0.05 * rng.standard_normal((40, 16)) for c in centres]).astype(np.float32)
    return vectors, [f"i{n}" for n in range(len(vectors))]
