"""DK-only fail-closed proof before discarding a derived embedder cache.

A graph.pkl from frozen legacy code may pickle a ~90 MB SentenceTransformer.
That object is an executable *cache*, not an authored memory. Never silently
omit it from an exact receipt: compare it to a freshly constructed, pinned
named model (weights, tokenizer, effective configuration and deterministic
outputs), preserve the old pair independently, then explicitly detach it.
Unsupported cache classes, hooks, training mode and mismatches refuse.
"""
from __future__ import annotations

import hashlib
import json
import types
from collections import OrderedDict
from typing import Any

import numpy as np

_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_PROBES = ("QMG model-cache attestation v2", "semantic entity temporal memory")
_HOOKS = (
    "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks",
    "_forward_hooks_with_kwargs", "_forward_pre_hooks_with_kwargs",
    "_forward_hooks_always_called", "_state_dict_hooks", "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks",
)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _simple(value: Any, path: str) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError(f"{path}: nonfinite model setting")
        return value
    if type(value) in (list, tuple):
        return [_simple(v, path) for v in value]
    if type(value) is set:
        if any(type(v) is not str for v in value):
            raise ValueError(f"{path}: unsupported model setting set")
        return sorted(value)
    if _class_name(value) == "tokenizers.AddedToken":
        state = value.__getstate__()
        if type(state) is not dict:
            raise ValueError(f"{path}: invalid tokenizer added token")
        return {"added_token": _simple(state, path)}
    if type(value) in (types.BuiltinFunctionType, types.FunctionType):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", "")
        if (not module.startswith(("torch.", "transformers.", "sentence_transformers."))
                or not name or getattr(value, "__closure__", None)):
            raise ValueError(f"{path}: untrusted executable model setting")
        return {"callable_module": module, "name": name}
    if type(value) in (dict, OrderedDict):
        if any(type(k) not in (str, int) for k in value):
            raise ValueError(f"{path}: unsupported model setting key")
        # Preserve order and key type for tokenizer's integer-key decoder.
        return [[[type(k).__name__, k], _simple(v, f"{path}.{k}")]
                for k, v in value.items()]
    raise ValueError(f"{path}: unsupported model setting type {type(value).__name__}")


def _tensor(tensor: Any, path: str) -> dict:
    import torch
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{path}: missing model tensor")
    arr = tensor.detach().cpu().contiguous().numpy()
    if arr.dtype.kind not in "fiu" or not np.isfinite(arr).all():
        raise ValueError(f"{path}: unsupported/nonfinite model tensor")
    return {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "sha256": hashlib.sha256(arr.tobytes(order="C")).hexdigest()}


def _class_name(obj: Any) -> str:
    return type(obj).__module__ + "." + type(obj).__qualname__


def _assert_no_hooks(model: Any) -> None:
    for module in model.modules():
        if getattr(module, "training", True):
            raise ValueError("model cache must be in inference/eval mode")
        for name in _HOOKS:
            if getattr(module, name, {}):
                raise ValueError("model cache has custom executable hooks")
        if getattr(module, "_parameters", None):
            # Parameters are covered by state_dict; an unregistered parameter
            # cannot safely be assumed reconstructible from the named model.
            for name, value in module._parameters.items():
                if value is None:
                    raise ValueError("model cache has null registered parameter")


def _fingerprint(model: Any) -> dict:
    from sentence_transformers import SentenceTransformer
    if type(model) is not SentenceTransformer:
        raise ValueError("unsupported model cache class")
    if getattr(model, "backend", None) != "torch" or getattr(model, "trust_remote_code", None):
        raise ValueError("unsupported model backend or remote model code")
    _assert_no_hooks(model)
    keys = list(model._modules)
    if keys != ["0", "1", "2"]:
        raise ValueError("unexpected DK embedding model modules")
    transformer, pooling, normalizer = (model._modules[k] for k in keys)
    if (_class_name(transformer) != "sentence_transformers.base.modules.transformer.Transformer"
            or _class_name(pooling) != "sentence_transformers.sentence_transformer.modules.pooling.Pooling"
            or _class_name(normalizer) != "sentence_transformers.sentence_transformer.modules.normalize.Normalize"):
        raise ValueError("unexpected DK embedding model architecture")
    tok = getattr(transformer, "processor", None)
    if (_class_name(tok) != "transformers.models.bert.tokenization_bert.BertTokenizer"
            or not hasattr(tok, "backend_tokenizer")):
        raise ValueError("unexpected DK tokenizer")
    state = tok.backend_tokenizer.__getstate__()
    if type(state) is not bytes:
        raise ValueError("tokenizer state is not deterministic bytes")
    try:
        tokenizer_backend = json.loads(state.decode("utf-8", "strict"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("tokenizer backend is not canonical JSON") from exc
    if type(tokenizer_backend) is not dict or not {"model", "normalizer", "pre_tokenizer"} <= set(tokenizer_backend):
        raise ValueError("tokenizer backend schema is unrecognized")
    # HF encode() supplies padding/truncation on every call. These two Rust
    # settings retain the previous call's transient values across pickle but
    # are overwritten before QMG's next encode; the model's max_seq_length is
    # captured in module configuration and equivalence probes run below.
    tokenizer_backend.pop("padding", None)
    tokenizer_backend.pop("truncation", None)
    formatter = transformer.input_formatter
    if getattr(formatter, "processor", None) is not tok:
        raise ValueError("tokenizer and input formatter lost their shared identity")
    fstate = vars(formatter)
    if set(fstate) != {"message_format", "model_type", "supported_modalities", "processor"}:
        raise ValueError("unaccounted formatter configuration")
    tokenizer_fields = {k: v for k, v in vars(tok).items() if k != "_tokenizer"}
    if not tokenizer_fields or any(callable(v) for v in tokenizer_fields.values()):
        raise ValueError("unaccounted tokenizer behavior")
    init_kwargs = tokenizer_fields.get("init_kwargs")
    if type(init_kwargs) is not dict:
        raise ValueError("tokenizer init settings are malformed")
    init_kwargs = dict(init_kwargs)
    load_hint = init_kwargs.pop("local_files_only", None)
    if load_hint is not None and type(load_hint) is not bool:
        raise ValueError("tokenizer local-files hint is malformed")
    # Loader-only hint cannot affect an already constructed tokenizer; the
    # effective backend/vocabulary and runtime settings are fingerprinted.
    tokenizer_fields["init_kwargs"] = init_kwargs
    metadata = {k: v for k, v in vars(model).items()
                if k in {"backend", "prompts", "default_prompt_name", "truncate_dim",
                         "module_kwargs", "training", "trust_remote_code",
                         "is_hpu_graph_enabled", "_similarity_fn_name"}}
    expected_meta = {"backend", "prompts", "default_prompt_name", "truncate_dim",
                     "module_kwargs", "training", "trust_remote_code",
                     "is_hpu_graph_enabled", "_similarity_fn_name"}
    if set(metadata) != expected_meta:
        raise ValueError("unaccounted sentence-transformer configuration")
    modules = []
    for name, module in model.named_modules():
        mvars = vars(module)
        public = {k: v for k, v in mvars.items()
                  if not k.startswith("_") and k not in {"processor", "input_formatter", "config",
                                                       "model_card_data"}}
        if "model_forward_params" in public:
            # Stored as a set of forward-argument names; iteration order is
            # randomized across Python processes, so canonicalize it.
            pass
        module_info = {"name": name, "class": _class_name(module),
                       "public": _simple(public, f"module.{name}")}
        if hasattr(module, "config"):
            cfg = module.config
            if not hasattr(cfg, "to_json_string"):
                raise ValueError("unaccounted transformer config type")
            module_info["config"] = cfg.to_json_string(use_diff=False)
        modules.append(module_info)
    tensors = [[name, _tensor(tensor, name)] for name, tensor in model.state_dict().items()]
    body = {"schema": "qmg-dk-model-cache-v1", "class": _class_name(model),
            "metadata": _simple(metadata, "model"),
            "modules": modules,
            "tensors": tensors,
            "tokenizer": {"class": _class_name(tok),
                          "backend_state_sha256": _hash(tokenizer_backend),
                          "fields": _simple(tokenizer_fields, "tokenizer")},
            "formatter": _simple({k: v for k, v in fstate.items() if k != "processor"},
                                  "formatter")}
    return {"full": _hash(body),
            "weights": _hash(tensors),
            "tokenizer": _hash(body["tokenizer"]),
            "configuration": _hash({"metadata": body["metadata"],
                                    "modules": body["modules"],
                                    "formatter": body["formatter"]})}


def verify_and_detach_embedder(graph: Any) -> str | None:
    """Return proof hash; mutate only AFTER strict current/fresh equivalence.

    The entire old graph pair must be privately preserved by the reviewed
    operator before this function is used on a legacy handoff. Do not turn a
    probe pass into a claim that model library or assets will stay unchanged
    indefinitely: release identity and installed model files are deployment
    gates, and a later mismatch refuses rather than re-embeds silently.
    """
    cached = graph._embedder
    if cached is None:
        return None
    if graph._model_name != _MODEL:
        raise ValueError("DK model cache cannot be reproduced from this model ID")
    from sentence_transformers import SentenceTransformer
    before = _fingerprint(cached)
    fresh = SentenceTransformer(_MODEL)
    fresh.eval()  # legacy encode() leaves the persisted cache in eval mode
    after = _fingerprint(fresh)
    if before != after:
        raise ValueError("cached model weights/tokenizer/config differ from named model")
    for probe in _PROBES:
        a = np.asarray(cached.encode([probe], normalize_embeddings=True)[0])
        b = np.asarray(fresh.encode([probe], normalize_embeddings=True)[0])
        if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
            raise ValueError("cached model behavior differs from named model")
    # Do not strip BM25 here: its exact persisted state is included by v2.
    # Existing memory embeddings are unmodified; the next encode lazily loads
    # the SAME pinned model (as proven above) from disk.
    graph._embedder = None
    return before["full"]
