from .layers.layer import Layer
from ..data.dataloader import DataLoader
from .evaluation import evaluation
import pickle, os, json, time, struct
from collections import OrderedDict
from ..tensor.utils import serialization as _ser
from ..tensor.utils.json_utils import _tensor_to_json, _json_to_tensor, _numel
from ..tensor.utils.utils import format_bytes

import numpy as np

_MODEL_MAGIC = b"NRXMDL\x00\x00"
_MODEL_HEADER = struct.Struct("<8sHQ")


class Model(Layer):
    def __init__(self):
        super().__init__()

    

    def predict(self, input):
        input_with_batch = input.unsqueeze(axis=0)
        return self(input_with_batch)[0]

    def evaluate(self, dataset, file=None, batch_size=64, threshold=0.5):
        result = evaluation(self, dataset, batch_size=batch_size, threshold=threshold)
        if file is not None:
            with open(file, "w") as t:
                json.dump(result, t, indent=2)
        return result

    def _meta(self, modules_summery=False):
        all_params = list(self.named_parameters())
        all_buffers = list(self.named_buffers())

        total_param_e = sum(_numel(t) for _, t in all_params)
        total_buffer_e = sum(_numel(t) for _, t in all_buffers)

        meta = {
            "format": _ser.FORMAT,
            "format_version": _ser.FORMAT_VERSION,
            "framework_version": _ser.FRAMEWORK_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "built": self._built,
            "input_shape": list(self._input_shape) if self._input_shape else None,
            "totals": {
                "param_tensors": len(all_params),
                "param_elements": total_param_e,
                "buffer_tensors": len(all_buffers),
                "buffer_elements": total_buffer_e,
            },
        }
        if modules_summery:
            meta["modules"] = (self._module_summary(),)
        return meta

    def save_json(self, path, modules_summery=False):
        if not self._built:
            raise ValueError(
                "Refusing to save an unbuilt model — no parameters exist yet."
            )

        meta = self._meta(modules_summery)

        state = {}
        for key, tensor in self._iter_state():
            blob = _tensor_to_json(tensor)
            state[key] = blob

        payload = {
            "meta": meta,
            "state": state,
        }

        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=False)

        msg = (
            f"✅ Model saved in json to {path} ({format_bytes(os.path.getsize(path))})"
        )
        print(msg)

    def load_json(self, path, strict=True, return_report=False):
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file '{path}' doesn’t exist.")

        with open(path) as f:
            payload = json.load(f)

        if payload["meta"]["format"] != _ser.FORMAT:
            raise ValueError(f"not a {_ser.FORMAT} file: {payload['meta']['format']}")
        if payload["meta"]["format_version"] > _ser.FORMAT_VERSION:
            raise ValueError(
                f"checkpoint format v{payload['meta']['format_version']} is newer "
                f"than this build (v{_ser.FORMAT_VERSION})"
            )

        state_in_shape = tuple(payload["meta"]["input_shape"])
        if self._built:
            if self._input_shape != state_in_shape:
                raise ValueError(
                    f"Input shape mismatch: model shape {self._input_shape} but state {state_in_shape}"
                )
        else:
            self.input(*state_in_shape)

        sd = OrderedDict()
        for k, v in payload["state"].items():
            sd[k] = _json_to_tensor(v)

        report = self.load_state_dict(sd, strict=strict)
        report["meta"] = payload["meta"]

        return report if return_report else self

    def save(self, path, modules_summery=False):
        if not self._built:
            raise ValueError(
                "Refusing to save an unbuilt model — no parameters exist yet."
            )

        entries = []

        for key, tensor in self._iter_state():
            blob = _ser.encode_tensor(tensor, key=key)
            entries.append((key, blob))

        meta = self._meta(modules_summery)
        meta["n_tensors"] = len(entries)
        meta_bytes = json.dumps(meta, indent=2).encode("utf-8")

        with open(path, "wb") as f:
            f.write(
                _MODEL_HEADER.pack(_MODEL_MAGIC, _ser.FORMAT_VERSION, len(meta_bytes))
            )
            f.write(meta_bytes)
            for _key, blob in entries:
                f.write(_ser._HEADER_LEN.pack(len(blob)))  # 4-byte len_i
                f.write(blob)

        print(f"✅ Model saved to {path} ({format_bytes(os.path.getsize(path))})")

    def load(self, path, strict=True, return_report=False):
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file '{path}' doesn’t exist.")

        file_size = os.path.getsize(path)

        with open(path, "rb") as f:
            head = f.read(_MODEL_HEADER.size)
            if len(head) != _MODEL_HEADER.size:
                raise ValueError(
                    f"'{path}' too small to be a model file ({file_size} bytes)"
                )
            magic, version, meta_len = _MODEL_HEADER.unpack(head)

            if magic != _MODEL_MAGIC:
                raise ValueError(
                    f"'{path}' is not a Nerux model file: bad magic {magic!r}"
                )
            if version > _ser.FORMAT_VERSION:
                raise ValueError(
                    f"model format v{version} is newer than this build "
                    f"(v{_ser.FORMAT_VERSION})"
                )

            meta_bytes = f.read(meta_len)
            if len(meta_bytes) != meta_len:
                raise ValueError(
                    f"'{path}' truncated in JSON header: "
                    f"expected {meta_len} bytes, got {len(meta_bytes)}"
                )
            try:
                meta = json.loads(meta_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                raise ValueError(f"'{path}' has corrupt JSON header: {e}") from e

            # ── 2) Shape / build validation (same as before) ──────
            if meta["format"] != _ser.FORMAT:
                raise ValueError(f"not a {_ser.FORMAT} file: got {meta['format']!r}")
            if meta["format_version"] > _ser.FORMAT_VERSION:
                raise ValueError(
                    f"checkpoint format v{meta['format_version']} is newer "
                    f"than this build (v{_ser.FORMAT_VERSION})"
                )

            state_in_shape = tuple(meta["input_shape"]) if meta["input_shape"] else None
            if self._built:
                if self._input_shape != state_in_shape:
                    raise ValueError(
                        f"input shape mismatch: model is "
                        f"{self._input_shape}, checkpoint is {state_in_shape}"
                    )
            else:
                if state_in_shape is None:
                    raise ValueError(
                        "checkpoint has no input_shape and model is unbuilt — "
                        "call self.input(...) before load()"
                    )
                self.input(*state_in_shape)

            n_tensors = meta["n_tensors"]

            sd = OrderedDict()
            for i in range(n_tensors):
                len_raw = f.read(_ser._HEADER_LEN.size)
                if len(len_raw) != _ser._HEADER_LEN.size:
                    raise ValueError(
                        f"'{path}': truncated at tensor {i} — "
                        f"missing length prefix (offset {f.tell()})"
                    )
                (blob_len,) = _ser._HEADER_LEN.unpack(len_raw)

                remaining = file_size - f.tell()
                if blob_len > remaining:
                    raise ValueError(
                        f"'{path}': tensor {i} declares blob_len={blob_len} "
                        f"but only {remaining} bytes remain"
                    )

                blob = f.read(blob_len)
                if len(blob) != blob_len:
                    raise ValueError(
                        f"'{path}': tensor {i} truncated — "
                        f"expected {blob_len} bytes, got {len(blob)}"
                    )

                try:
                    key, tensor = _ser.decode_tensor(blob, copy=True)
                except ValueError as e:
                    raise ValueError(
                        f"'{path}': tensor {i} (blob_len={blob_len}) "
                        f"failed to decode: {e}"
                    ) from e

                if key is None:
                    raise ValueError(
                        f"'{path}': tensor {i} has no key — model files "
                        f"must have keyed tensors"
                    )
                if key in sd:
                    raise ValueError(f"'{path}': duplicate key {key!r} at tensor {i}")
                sd[key] = tensor

            trailing = f.read(1)
            if trailing:
                raise ValueError(
                    f"'{path}': {n_tensors} tensors consumed but "
                    f"{file_size - f.tell() + 1}+ trailing bytes remain — "
                    f"corrupt file?"
                )

        report = self.load_state_dict(sd, strict=strict)
        report["meta"] = meta

        return report if return_report else self
