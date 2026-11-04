import os
import struct
import numpy as np
from collections import OrderedDict
from ..tensor import Tensor
from .. import factory as init
from .utils import format_bytes
# ─── Format constants ──────────────────────────────────────────────
FORMAT = "state_dict_bin"
FORMAT_VERSION = 1
FRAMEWORK_VERSION=1
_FILE_MAGIC = b"SDBIN\x00\x00\x00"
_FILE_HEADER = struct.Struct("<8sHQ")  # magic(8) | version(2) | n_tensors(8)

# ── Tensor blob layout (produced by encode_tensor) ─────────────────
#   header_len(4) | header | raw C-order bytes
#
# Header field order:
#   key_len (2B, <H)  -- 0 means no key
#   ndim    (1B, u8)
#   nbytes  (8B, <Q)
#   flags   (1B, u8)  -- bits 0..4 dtype code | bit 5 requires_grad
#   key     (key_len bytes, utf-8)   -- omitted when key_len == 0
#   shape   (ndim × 8B, <q each)
#
# ── State-dict container layout ────────────────────────────────────
#   FILE_HEADER
#   len_1(4) | blob_1
#   len_2(4) | blob_2
#   ...
#
_HEADER_LEN = struct.Struct("<I")  # header_len (inside blob) and len_i (container)
_HEADER_FIXED = struct.Struct("<HBQB")  # key_len, ndim, nbytes, flags
_SHAPE_DIM = struct.Struct("<q")

_DTYPE_CODE = {
    "f16": (0, np.dtype("<f2")),
    "f32": (1, np.dtype("<f4")),
    "f64": (2, np.dtype("<f8")),
    "i8": (3, np.dtype("i1")),
    "i16": (4, np.dtype("<i2")),
    "i32": (5, np.dtype("<i4")),
    "i64": (6, np.dtype("<i8")),
    "u8": (7, np.dtype("u1")),
    "u16": (8, np.dtype("<u2")),
    "u32": (9, np.dtype("<u4")),
    "u64": (10, np.dtype("<u8")),
    "bool": (11, np.dtype("?")),
}
_CODE_TO_DTYPE = {code: dt for code, dt in _DTYPE_CODE.values()}
_CODE_TO_NAME = {code: name for name, (code, _) in _DTYPE_CODE.items()}

_FLAG_GRAD_BIT = 5
_FLAG_DTYPE_MASK = 0x1F


# ─── helpers ──────────────────────────────────────────────────────
def _dtype_code_for(arr: np.ndarray) -> int:
    dt = np.dtype(arr.dtype).newbyteorder("<")
    for name, (code, supported) in _DTYPE_CODE.items():
        if supported == dt:
            return code
    raise TypeError(
        f"Unsupported dtype {arr.dtype} for saving. "
        f"Supported: {list(_DTYPE_CODE)}. Cast the tensor before saving."
    )


def _validate_shape(shape, nbytes: int, dtype: np.dtype, where: str):
    expected = dtype.itemsize
    for d in shape:
        if d < 0:
            raise ValueError(f"{where}: negative dimension {d} in shape {shape}")
        expected *= d
    if expected != nbytes:
        raise ValueError(
            f"{where}: inconsistent layout — shape {tuple(shape)} × {dtype} "
            f"implies {expected} bytes, header declares {nbytes}"
        )


# ─── header encode / decode ───────────────────────────────────────
def _encode_header(key, arr: np.ndarray, requires_grad: bool) -> bytes:
    key_bytes = key.encode("utf-8") if key is not None else b""
    if len(key_bytes) > 0xFFFF:
        raise ValueError(f"key too long ({len(key_bytes)} > 65535 bytes): {key!r}")
    if arr.ndim > 255:
        raise ValueError(f"ndim {arr.ndim} > 255 for key {key!r}")

    flags = _dtype_code_for(arr) & _FLAG_DTYPE_MASK
    if requires_grad:
        flags |= 1 << _FLAG_GRAD_BIT

    fixed = _HEADER_FIXED.pack(len(key_bytes), arr.ndim, arr.nbytes, flags)
    shape_bytes = b"".join(_SHAPE_DIM.pack(int(d)) for d in arr.shape)
    return fixed + key_bytes + shape_bytes


def _decode_header(header: memoryview, where: str) -> dict:
    header = bytes(header)

    if len(header) < _HEADER_FIXED.size:
        raise ValueError(
            f"{where}: header too small ({len(header)} bytes, "
            f"need ≥ {_HEADER_FIXED.size})"
        )
    key_len, ndim, nbytes, flags = _HEADER_FIXED.unpack_from(header, 0)
    expected = _HEADER_FIXED.size + key_len + ndim * _SHAPE_DIM.size
    if expected != len(header):
        raise ValueError(
            f"{where}: header size mismatch — fields imply {expected} bytes "
            f"(key_len={key_len}, ndim={ndim}), got {len(header)}"
        )

    pos = _HEADER_FIXED.size
    key = header[pos : pos + key_len].decode("utf-8") if key_len else None
    pos += key_len
    shape = tuple(
        _SHAPE_DIM.unpack_from(header, pos + i * _SHAPE_DIM.size)[0]
        for i in range(ndim)
    )

    dtype_code = flags & _FLAG_DTYPE_MASK
    if dtype_code not in _CODE_TO_DTYPE:
        raise ValueError(f"{where}: unknown dtype code {dtype_code}")

    dtype = _CODE_TO_DTYPE[dtype_code]
    _validate_shape(shape, nbytes, dtype, where)

    return {
        "key": key,
        "dtype": dtype,
        "shape": shape,
        "nbytes": nbytes,
        "requires_grad": bool(flags & (1 << _FLAG_GRAD_BIT)),
    }


# ─── Tensor blob: encode / decode (self-contained, no outer length) ─
def encode_tensor(tensor: Tensor, key: str | None = None) -> bytes:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"encode_tensor expects a Tensor, got {type(tensor).__name__}")

    arr = np.ascontiguousarray(tensor.data)
    _validate_shape(
        arr.shape,
        arr.nbytes,
        arr.dtype.newbyteorder("<"),
        f"encode_tensor(key={key!r})",
    )

    header = _encode_header(key, arr, bool(tensor.requires_grad))
    return _HEADER_LEN.pack(len(header)) + header + arr.tobytes(order="C")


def decode_tensor(blob, *, copy=True):
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError(f"decode_tensor expects bytes-like, got {type(blob).__name__}")

    mv = memoryview(blob).cast("B")

    if len(mv) < _HEADER_LEN.size:
        raise ValueError(
            f"blob too short: need ≥ {_HEADER_LEN.size} bytes for header_len, "
            f"got {len(mv)}"
        )
    (header_len,) = _HEADER_LEN.unpack_from(mv, 0)

    if len(mv) < _HEADER_LEN.size + header_len:
        raise ValueError(
            f"blob truncated: header_len={header_len} needs "
            f"{_HEADER_LEN.size + header_len} bytes total, got {len(mv)}"
        )

    header = mv[_HEADER_LEN.size : _HEADER_LEN.size + header_len]
    where = f"decode_tensor(header_len={header_len})"
    info = _decode_header(header, where)

    data_start = _HEADER_LEN.size + header_len
    data_end = data_start + info["nbytes"]
    if data_end > len(mv):
        raise ValueError(
            f"{where}: blob truncated for tensor {info['key']!r} — "
            f"header declares nbytes={info['nbytes']} ending at offset "
            f"{data_end}, blob has {len(mv)} bytes"
        )

    raw = mv[data_start:data_end]
    arr = np.frombuffer(raw, dtype=info["dtype"]).reshape(info["shape"])
    array = arr.copy() if copy else arr

    tensor = init.tensor(
        array, dtype=info["dtype"], requires_grad=info["requires_grad"]
    )
    if bool(info["key"]):
        return info["key"], tensor
    return tensor


# ─── state_dict container: save / load ────────────────────────────
def save_dict(state_dict, path):
    """
    File layout:
        FILE_HEADER | len_1(4) | blob_1 | len_2(4) | blob_2 | ...
    Each `blob_i` is exactly what `encode_tensor(tensor, key=key)` returns.
    """
    if not isinstance(state_dict, (dict, OrderedDict)):
        raise TypeError(
            f"state_dict must be dict/OrderedDict, got {type(state_dict).__name__}"
        )

    blobs = []
    for i, (key, tensor) in enumerate(state_dict.items()):
        if not isinstance(key, str):
            raise TypeError(
                f"state_dict keys must be str — entry {i} has {type(key).__name__}"
            )
        if not isinstance(tensor, Tensor):
            raise TypeError(
                f"state_dict[{key!r}] must be a Tensor, got {type(tensor).__name__}"
            )
        blobs.append(encode_tensor(tensor, key=key))

    with open(path, "wb") as f:
        f.write(_FILE_HEADER.pack(_FILE_MAGIC, FORMAT_VERSION, len(blobs)))
        for blob in blobs:
            f.write(_HEADER_LEN.pack(len(blob)))
            f.write(blob)

    print(
        f"✅ state_dict saved to {path} {format_bytes(os.path.getsize(path))} {len(blobs)} tensors)"
    )
    return path


def load_dict(path, *, copy_params=True):
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found: {path}")

    sd = OrderedDict()

    with open(path, "rb") as f:
        file_size = os.fstat(f.fileno()).st_size

        head = f.read(_FILE_HEADER.size)
        if len(head) != _FILE_HEADER.size:
            raise ValueError(
                f"'{path}' too small: need ≥ {_FILE_HEADER.size} bytes for "
                f"file header, got {len(head)}"
            )
        magic, version, n_tensors = _FILE_HEADER.unpack(head)

        if magic != _FILE_MAGIC:
            raise ValueError(f"'{path}' is not a state_dict file: bad magic {magic!r}")
        if version > FORMAT_VERSION:
            raise ValueError(
                f"'{path}': format v{version} is newer than this build "
                f"(v{FORMAT_VERSION})"
            )

        for i in range(n_tensors):
            len_raw = f.read(_HEADER_LEN.size)
            if len(len_raw) != _HEADER_LEN.size:
                raise ValueError(
                    f"'{path}': truncated at tensor {i} — missing length prefix "
                    f"(offset {f.tell() - len(len_raw)})"
                )
            (blob_len,) = _HEADER_LEN.unpack(len_raw)

            remaining = file_size - f.tell()
            if blob_len > remaining:
                raise ValueError(
                    f"'{path}': tensor {i} declares blob_len={blob_len} but "
                    f"only {remaining} bytes remain in file"
                )

            blob = f.read(blob_len)
            if len(blob) != blob_len:
                raise ValueError(
                    f"'{path}': tensor {i} truncated — expected {blob_len} "
                    f"bytes, got {len(blob)}"
                )

            try:
                key, tensor = decode_tensor(blob, copy=copy_params)
            except ValueError as e:
                raise ValueError(
                    f"'{path}': tensor {i} (blob_len={blob_len}) failed to decode: {e}"
                ) from e

            if key is None:
                raise ValueError(
                    f"'{path}': tensor {i} has no key — state_dict entries "
                    f"must be keyed (use encode_tensor alone for keyless blobs)"
                )
            if key in sd:
                raise ValueError(f"'{path}': duplicate key {key!r} at tensor {i}")
            sd[key] = tensor

        trailing = f.read(1)
        if trailing:
            raise ValueError(
                f"'{path}': {n_tensors} tensors consumed but "
                f"{file_size - f.tell() + 1}+ trailing bytes remain — corrupt file?"
            )

    return sd


def save(tensor, path):
    blob = encode_tensor(tensor)
    with open(path, "wb") as f:
        f.write(blob)


def load(path):
    tensor = None
    with open(path, "rb") as f:
        tensor = decode_tensor(f.read())
        if isinstance(tensor, tuple):
            tensor = tensor[1]
    return tensor
