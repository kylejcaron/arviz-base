# File generated with docstub

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import numpyro
from xarray import Dataset, DataTree

from arviz_base.base import dict_to_dataset, requires
from arviz_base.rcparams import rc_context, rcParams
from arviz_base.utils import expand_dims

def _add_dims(
    dims_a: dict[str, list[str]], dims_b: dict[str, list[str]]
) -> dict[str, list[str]]: ...
def infer_dims(
    model: Callable,
    model_args: tuple[Any, ...] | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
) -> dict[str, list[str]]: ...

class NumPyroInferenceAdapter(ABC):
    @property
    @abstractmethod
    def model(self) -> None: ...
    @property
    @abstractmethod
    def sample_dims(self) -> list[str]: ...
    @property
    @abstractmethod
    def sample_shape(self) -> tuple[int, ...]: ...
    @abstractmethod
    def get_samples(self, group_by_chain: bool = ...) -> None: ...
    @abstractmethod
    def _infer_sample_shape(self) -> None: ...
    @abstractmethod
    def _get_train_args_kwargs(self) -> None: ...

class MCMCAdapter(NumPyroInferenceAdapter):
    def __init__(
        self,
        posterior,
        num_chains=...,
        prior=...,
        posterior_predictive=...,
        predictions=...,
    ) -> None: ...
    @property
    def model(self) -> None: ...
    @property
    def sample_dims(self) -> list[str]: ...
    @property
    def sample_shape(self) -> tuple[int, ...]: ...
    def get_samples(self, group_by_chain: bool = ...) -> None: ...
    def _infer_sample_shape(self) -> None: ...
    def _get_train_args_kwargs(self) -> None: ...

class SVIAdapter(NumPyroInferenceAdapter):
    def __init__(
        self, svi, svi_result, num_samples, model_args=..., model_kwargs=...
    ) -> None: ...
    @property
    def model(self) -> None: ...
    @property
    def sample_dims(self) -> list[str]: ...
    @property
    def sample_shape(self) -> tuple[int, ...]: ...
    def get_samples(self, group_by_chain: bool = ...) -> None: ...
    def _infer_sample_shape(self) -> None: ...
    def _get_train_args_kwargs(self) -> None: ...

class NestedMCMCAdapter(NumPyroInferenceAdapter):
    def __init__(
        self, nested_sampler, num_samples, model_args=..., model_kwargs=...
    ) -> None: ...
    @property
    def model(self) -> None: ...
    @property
    def sample_dims(self) -> list[str]: ...
    @property
    def sample_shape(self) -> tuple[int, ...]: ...
    def get_samples(self, group_by_chain: bool = ...) -> None: ...
    def _infer_sample_shape(self) -> None: ...
    def _get_train_args_kwargs(self) -> None: ...

class BaseNumPyroConverter:
    def __init__(
        self,
        adapter: NumPyroInferenceAdapter,
        *,
        prior: dict | None = ...,
        posterior_predictive: dict | None = ...,
        predictions: dict | None = ...,
        constant_data: dict | None = ...,
        predictions_constant_data: dict | None = ...,
        log_likelihood=...,
        index_origin: int | None = ...,
        coords: dict | None = ...,
        dims: dict[str, list[str]] | None = ...,
        pred_dims: dict | None = ...,
        extra_event_dims: dict | None = ...,
    ) -> None: ...
    @property
    def model(self) -> None: ...
    def sample_stats_to_xarray(self) -> Dataset | None: ...
    def _get_model_trace(self, model, model_args, model_kwargs, key) -> None: ...
    def _prepare_predictive_data(self, dct: dict) -> dict: ...
    def posterior_to_xarray(self) -> None: ...
    def log_likelihood_to_xarray(self) -> None: ...
    def translate_posterior_predictive_dict_to_xarray(self, dct, dims) -> None: ...
    def posterior_predictive_to_xarray(self) -> None: ...
    def predictions_to_xarray(self) -> None: ...
    def priors_to_xarray(self) -> None: ...
    def observed_data_to_xarray(self) -> None: ...
    def constant_data_to_xarray(self) -> None: ...
    def predictions_constant_data_to_xarray(self) -> None: ...
    def to_datatree(self) -> None: ...
    def infer_dims(self) -> dict[str, list[str]]: ...
    def infer_pred_dims(self) -> dict[str, list[str]]: ...

class NumPyroConverter(BaseNumPyroConverter):
    def __init__(self, adapter: NumPyroInferenceAdapter, **kwargs) -> None: ...
    def sample_stats_to_xarray(self) -> Dataset | None: ...
    def _mcmc_sample_stats_to_xarray(self) -> None: ...

def from_numpyro(
    posterior: numpyro.infer.mcmc.MCMC | None = ...,
    *,
    prior: dict | None = ...,
    posterior_predictive: dict | None = ...,
    predictions: dict | None = ...,
    constant_data: dict | None = ...,
    predictions_constant_data: dict | None = ...,
    log_likelihood=...,
    index_origin: int | None = ...,
    coords: dict | None = ...,
    dims: dict[str, list[str]] | None = ...,
    pred_dims: dict | None = ...,
    extra_event_dims: dict | None = ...,
    num_chains: int = ...,
) -> DataTree: ...
def from_numpyro_svi(
    svi: numpyro.infer.svi.SVI,
    *,
    svi_result: numpyro.infer.svi.SVIRunResult,
    model_args: tuple | None = ...,
    model_kwargs: dict | None = ...,
    prior: dict | None = ...,
    posterior_predictive: dict | None = ...,
    predictions: dict | None = ...,
    constant_data: dict | None = ...,
    predictions_constant_data: dict | None = ...,
    log_likelihood=...,
    index_origin: int | None = ...,
    coords: dict | None = ...,
    dims: dict[str, list[str]] | None = ...,
    pred_dims: dict | None = ...,
    extra_event_dims: dict | None = ...,
    num_samples: int = ...,
) -> DataTree: ...
def from_numpyro_nested_mcmc(
    nested_sampler: numpyro.infer.NestedSampler,
    *,
    model_args=...,
    model_kwargs=...,
    prior: dict | None = ...,
    posterior_predictive: dict | None = ...,
    predictions: dict | None = ...,
    constant_data: dict | None = ...,
    predictions_constant_data: dict | None = ...,
    log_likelihood: bool | None = ...,
    index_origin: int | None = ...,
    coords: dict | None = ...,
    dims: dict[str, list[str]] | None = ...,
    pred_dims: dict | None = ...,
    extra_event_dims: dict | None = ...,
    num_samples: int = ...,
) -> DataTree: ...
