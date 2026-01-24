# NumPyro Adapter Pattern Design Document

## Overview

This document describes the refactoring of `arviz_base.io_numpyro` to use an adapter pattern. The goal is to consolidate three separate converter classes (`MCMCConverter`, `SVIConverter`, `NestedMCMCConverter`) into a single `NumPyroConverter` class that works with adapters.

The adapters (`MCMCAdapter`, `SVIAdapter`, `NestedMCMCAdapter`) normalize the different inference object interfaces so that `NumPyroConverter` can handle all inference types with shared logic.

## Motivation

NumPyro supports multiple inference algorithms with different interfaces and data structures:
- **MCMC**: Chain-based sampling with sample statistics
- **SVI**: Variational inference with guide-based approximations
- **NestedSampler**: Nested sampling for Bayesian evidence computation

The current implementation has three separate converter classes that duplicate common conversion logic. Each converter handles the interface differences of its specific inference type, leading to:
- Code duplication across converters (posterior conversion, log likelihood, priors, etc.)
- Difficult to maintain consistency when adding features
- Harder to test common logic

## Design Goals

1. **Separation of Concerns**: Isolate inference-specific logic (adapters) from generic conversion logic (converter)
2. **Reduce Duplication**: Share common conversion code across all inference types
3. **Maintainability**: Single converter class is easier to maintain and test
4. **Type Safety**: Clear adapter interface with type hints
5. **Backward Compatibility**: Maintain existing public API (`from_numpyro`, `from_numpyro_svi`, `from_numpyro_nested_mcmc`)

## Architecture

```
NumPyroInferenceAdapter(ABC)
├── MCMCAdapter(NumPyroInferenceAdapter)
├── SVIAdapter(NumPyroInferenceAdapter)
└── NestedMCMCAdapter(NumPyroInferenceAdapter)

NumPyroConverter  # Works with any adapter

# Public API (unchanged)
from_numpyro(posterior, ...)  # Creates MCMCAdapter
from_numpyro_svi(svi, svi_result, ...)  # Creates SVIAdapter
from_numpyro_nested_mcmc(nested_sampler, ...)  # Creates NestedMCMCAdapter
```

### Adapter Interface

`NumPyroInferenceAdapter` (Abstract Base):
```python
class NumPyroInferenceAdapter(ABC):
    @property
    @abstractmethod
    def model(self):
        """Return the model function"""

    @property
    @abstractmethod
    def sample_dims(self) -> list[str]:
        """Return dimension names (e.g., ["chain", "draw"] or ["sample"])"""

    @property
    @abstractmethod
    def sample_shape(self) -> tuple[int, ...]:
        """Return shape of samples"""

    @abstractmethod
    def get_samples(self, group_by_chain: bool = False):
        """Extract posterior samples"""

    @abstractmethod
    def _infer_sample_shape(self):
        """Compute sample shape from inference object"""
```

### Adapter Implementations

**MCMCAdapter**
- Wraps `numpyro.infer.MCMC` objects
- `sample_dims = ["chain", "draw"]`
- `sample_shape = (num_chains, num_samples // thinning)`
- Provides sample statistics (divergences, energy, etc.)

**SVIAdapter**
- Wraps `(svi, svi_result)` pair
- `sample_dims = ["sample"]`
- `sample_shape = (num_samples,)`
- Generates samples from trained guide distribution
- No sample statistics available

**NestedMCMCAdapter**
- Wraps `numpyro.infer.NestedSampler` object
- `sample_dims = ["sample"]`
- `sample_shape = (num_samples,)`
- Extracts samples from nested sampler
- No sample statistics available

### Unified Converter

`NumPyroConverter` replaces the three separate converters:
```python
class NumPyroConverter:
    def __init__(self, adapter: NumPyroInferenceAdapter, ...):
        self.adapter = adapter
        # Common conversion logic

    def posterior_to_xarray(self):
        samples = self.adapter.get_samples()
        # Use adapter.sample_dims for dimension names

    def to_datatree(self):
        # Same for all inference types
```

### Public API (Unchanged)

```python
def from_numpyro(posterior, ...):
    adapter = MCMCAdapter(posterior, ...)
    with rc_context(rc={"data.sample_dims": ["chain", "draw"]}):
        return NumPyroConverter(adapter, ...).to_datatree()

def from_numpyro_svi(svi, svi_result, num_samples, ...):
    adapter = SVIAdapter(svi, svi_result, num_samples, ...)
    with rc_context(rc={"data.sample_dims": ["sample"]}):
        return NumPyroConverter(adapter, ...).to_datatree()

def from_numpyro_nested_mcmc(nested_sampler, num_samples, ...):
    adapter = NestedMCMCAdapter(nested_sampler, num_samples, ...)
    with rc_context(rc={"data.sample_dims": ["sample"]}):
        return NumPyroConverter(adapter, ...).to_datatree()
```

## Benefits

1. **Single Converter**: All conversion logic in one place (NumPyroConverter)
2. **Clean Separation**: Inference-specific logic isolated in adapters
3. **Less Duplication**: Share posterior_to_xarray, log_likelihood_to_xarray, etc.
4. **Easier Testing**: Test adapters and converter independently
5. **Future Extensions**: New inference types only need a new adapter class
