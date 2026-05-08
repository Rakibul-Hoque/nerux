import numpy as np
import os
import shutil
import subprocess


class Tid_count:
    global_tid_count = 0

    @classmethod
    def get_current(cls):
        return cls.global_tid_count

    @classmethod
    def get_incremented(cls):
        cls.global_tid_count += 1
        return cls.global_tid_count


def print_tensor(tensor):
    data_str = np.array2string(np.asarray(tensor.data), separator=", ")

    base = f"tensor({data_str}"
    if tensor.grad is not None:
        grad_data = np.asarray(tensor.grad.data)
        grad_str = np.array2string(grad_data, separator=", ")
        base += f", grad={grad_str}"
    if tensor.requires_grad:
        base += ", grad=True"
    base += f", id=T{tensor._id})"

    return base


def get_graph(tensor, max_depth=None, use_color=True):
    visited = set()
    lines = []

    # --- ANSI colors ---
    if use_color:
        C_RESET = "\033[0m"
        C_CYAN = "\033[96m"
        C_YELLOW = "\033[93m"
        C_GREEN = "\033[92m"
        C_RED = "\033[91m"
        C_GRAY = "\033[90m"
    else:
        C_RESET = C_CYAN = C_YELLOW = C_GREEN = C_RED = C_GRAY = ""

    def visit(tensor, prefix, is_last, depth):
        if max_depth is not None and depth > max_depth:
            lines.append(prefix + "└── ...")
            return

        tid = tensor._id

        if tensor in visited:
            lines.append(
                prefix
                + ("└── " if is_last else "├── ")
                + f"{C_GRAY}[visited id=T{tid}]{C_RESET}"
            )
            return
        visited.add(tensor)

        connector = "└── " if is_last else "├── "

        grad_color = C_GREEN if tensor.requires_grad else C_RED

        lines.append(
            prefix
            + connector
            + C_CYAN
            + f"Tensor(shape={tensor.shape}, grad="
            + grad_color
            + f"{tensor.requires_grad}"
            + C_CYAN
            + f", id=T{tid})"
            + C_RESET
        )

        if tensor.creator:
            lines.append(
                prefix
                + ("    " if is_last else "│   ")
                + C_YELLOW
                + f"op: {tensor.creator.__class__.__name__}"
                + C_RESET
            )

            inputs = tensor.creator.inputs
            for i, inp in enumerate(inputs):
                is_last_child = i == len(inputs) - 1
                new_prefix = prefix + ("    " if is_last else "│   ")

                visit(inp, new_prefix, is_last_child, depth + 1)

    visit(tensor, "", True, 0)
    return "\n".join(lines)


def export_graph(tensor, file, as_image=False):
    visited = set()
    lines = []

    def add(line):
        lines.append(line)

    # --- Graph header ---
    add("digraph ComputationGraph {")
    add('    bgcolor="#0f0f0f";')

    add("""
    node [fontname="Helvetica", fontsize=10];
    edge [color="#aaaaaa"];
    """)

    def dfs(t):
        tid = t._id

        if tid in visited:
            return
        visited.add(tid)

        shape = t.shape
        grad = t.requires_grad

        color = "#4CAF50" if grad else "#F44336"

        label = f"{tid}\\nshape={shape}"

        add(f'''
        "{tid}" [
            label="{label}",
            shape=ellipse,
            style=filled,
            fillcolor="{color}",
            fontcolor="white"
        ];
        ''')

        if t.creator:
            op_name = t.creator.__class__.__name__
            op_id = f"OP_{id(t.creator)}"

            add(f'''
            "{op_id}" [
                label="{op_name}",
                shape=box,
                style=filled,
                fillcolor="#2196F3",
                fontcolor="white"
            ];
            ''')

            add(f'"{op_id}" -> "{tid}";')

            for inp in t.creator.inputs:
                inp_id = inp._id
                add(f'"{inp_id}" -> "{op_id}";')
                dfs(inp)

    dfs(tensor)
    add("}")

    dot_string = "\n".join(lines)

    # ----------------------------
    # Save as DOT file
    # ----------------------------
    if not as_image:
        if not file.endswith(".dot"):
            file = file + ".dot"

        with open(file, "w") as f:
            f.write(dot_string)

        return file

    # ----------------------------
    # Export as Image
    # ----------------------------

    # Check Graphviz installation
    if shutil.which("dot") is None:
        raise RuntimeError(
            "Graphviz is not installed or 'dot' is not in PATH.\n"
            "Install from https://graphviz.org/download/"
        )

    # Validate extension
    valid_exts = {"png", "jpg", "jpeg", "svg", "pdf"}
    ext = os.path.splitext(file)[1].lower().replace(".", "")

    if ext == "":
        raise ValueError(f"No file extension provided. Use one of: {valid_exts}")

    if ext not in valid_exts:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {valid_exts}")

    # Run Graphviz
    process = subprocess.run(
        ["dot", f"-T{ext}", "-o", file],
        input=dot_string.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if process.returncode != 0:
        raise RuntimeError("Graphviz failed:\n" + process.stderr.decode())

    return file


def _format_time(seconds):
    if seconds >= 1:
        return f"{seconds:.3f} s"
    elif seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    elif seconds >= 1e-6:
        return f"{seconds * 1e6:.1f} µs"
    else:
        return f"{seconds * 1e9:.1f} ns"


def reduce_grad(grad, shape):
    # Strip extra leading dims
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # Sum over broadcasted dims (shape had size 1, grad may have size > 1)
    axes = tuple(i for i, dim in enumerate(shape) if dim == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)

    if shape == ():
        grad = grad.sum()

    assert grad.shape == shape, f"reduce_grad shape mismatch: {grad.shape} vs {shape}"
    return grad


class TensorView:
    def __init__(self, data):
        self.daat = data

    def __iadd__(self, other):
        raise RuntimeError("Use tensor.add_() instead of value +=")

    def __isub__(self, other):
        raise RuntimeError("Use tensor.sub_() instead")

    def __imul__(self, other):
        raise RuntimeError("Use tensor.mul_() instead")

    def __itruediv__(self, other):
        raise RuntimeError("Use tensor.div_() instead")

    def __array__(self):
        return self.data
