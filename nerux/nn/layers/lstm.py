# nerux/nn/layers/lstm.py

import numpy as np
from ...tensor import Tensor
from .base import Base
from .dropout import Dropout
from ..functional.lstm import LSTMCellFunction


class LSTM(Base):
    """
    Multi-layer, bidirectional LSTM.

    Args:
        hidden_size   : number of features in hidden state
        num_layers    : number of stacked LSTM layers            (default 1)
        bias          : if False, no bias terms                  (default True)
        dropout       : dropout probability between layers        (default 0.0)
        bidirectional : if True, process sequence both ways       (default False)
        proj_size     : if > 0, project hidden state to proj_size (default 0)

    Input:
        x             : (batch, seq_len, input_size)
        h0, c0        : optional initial states
                        each (num_layers * num_directions, batch, hidden_size)
                        or (num_layers * num_directions, batch, proj_size) for h0
                        when proj_size > 0

    Returns:
        output        : (batch, seq_len, hidden_size * num_directions)
                        or (batch, seq_len, proj_size * num_directions)
        (h_n, c_n)    : final states, same shape as (h0, c0)
    """

    def __init__(
        self,
        hidden_size,
        num_layers=1,
        bias=True,
        dropout=0.0,
        bidirectional=False,
        proj_size=0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = bias
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self.num_directions = 2 if bidirectional else 1
        self.out_size = proj_size if proj_size > 0 else hidden_size
        self._dropouts = []
        # _cells is now just a registry of parameter KEY NAMES, not Tensor refs
        # shape: {(layer, direction): {"W_ih": "W_ih_L0_D0", "W_hh": ..., ...}}
        self._cell_keys = {}

    # ── build ─────────────────────────────────────────────────────────────────

    def build(self, in_shape):
        input_size = in_shape[-1]

        for layer in range(self.num_layers):
            layer_input_size = (
                input_size if layer == 0 else self.out_size * self.num_directions
            )
            for d in range(self.num_directions):
                self._build_cell(layer, d, layer_input_size)

            if self.dropout_rate > 0 and layer < self.num_layers - 1:
                drop = self.add_layer(f"dropout_{layer}", Dropout(self.dropout_rate))
                self._dropouts.append(drop)
            else:
                self._dropouts.append(None)

    def _build_cell(self, layer, direction, input_size):
        H = self.hidden_size
        label = f"L{layer}_D{direction}"
        scale = 1.0 / np.sqrt(H)

        # h fed back into gates has dimension proj_size when projection is used
        # W_hh must match that: (4H, proj_size) not (4H, hidden_size)
        h_input_size = self.proj_size if self.proj_size > 0 else H

        W_ih = np.random.uniform(-scale, scale, (4 * H, input_size))
        W_hh = np.random.uniform(-scale, scale, (4 * H, h_input_size))  # ← fix

        b_ih = np.zeros(4 * H)
        b_ih[H : 2 * H] = 1.0
        b_hh = np.zeros(4 * H)

        self.add_parameter(f"W_ih_{label}", Tensor(W_ih, requires_grad=True))
        self.add_parameter(f"W_hh_{label}", Tensor(W_hh, requires_grad=True))

        keys = {"W_ih": f"W_ih_{label}", "W_hh": f"W_hh_{label}"}

        if self.use_bias:
            self.add_parameter(f"b_ih_{label}", Tensor(b_ih, requires_grad=True))
            self.add_parameter(f"b_hh_{label}", Tensor(b_hh, requires_grad=True))
            keys["b_ih"] = f"b_ih_{label}"
            keys["b_hh"] = f"b_hh_{label}"

        if self.proj_size > 0:
            # W_hr projects hidden_size → proj_size: (proj_size, hidden_size)
            scale_p = 1.0 / np.sqrt(H)
            W_hr = np.random.uniform(-scale_p, scale_p, (self.proj_size, H))
            self.add_parameter(f"W_hr_{label}", Tensor(W_hr, requires_grad=True))
            keys["W_hr"] = f"W_hr_{label}"

        self._cell_keys[(layer, direction)] = keys

    # ── live cell accessor — always reads from _parameters ───────────────────

    def _cell(self, layer, direction):
        """
        Returns a dict of live Tensor references by reading _parameters now.
        This means after set_parameters() updates _parameters, every subsequent
        _cell() call automatically sees the loaded weights.
        """
        keys = self._cell_keys[(layer, direction)]
        return {role: self._parameters[pkey] for role, pkey in keys.items()}

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x, states=None):
        """
        x      : Tensor (batch, seq_len, input_size)
        states : (h0, c0) each Tensor
                 (num_layers*num_directions, batch, hidden_or_proj_size)
                 or None → zeros
        """

        N, T, _ = x.shape
        num_dir = self.num_directions

        h0, c0 = self._init_states(states, N)

        layer_input = x
        final_h, final_c = [], []

        for layer in range(self.num_layers):
            fwd_out, h_fwd, c_fwd = self._run_direction(
                layer_input,
                layer,
                direction=0,
                h_init=h0[layer * num_dir],
                c_init=c0[layer * num_dir],
            )

            if self.bidirectional:
                bwd_out, h_bwd, c_bwd = self._run_direction(
                    layer_input,
                    layer,
                    direction=1,
                    h_init=h0[layer * num_dir + 1],
                    c_init=c0[layer * num_dir + 1],
                    reverse=True,
                )
                layer_output = Tensor.concat([fwd_out, bwd_out], axis=2)
                final_h.extend([h_fwd, h_bwd])
                final_c.extend([c_fwd, c_bwd])
            else:
                layer_output = fwd_out
                final_h.append(h_fwd)
                final_c.append(c_fwd)

            drop = self._dropouts[layer]
            if drop is not None and self.training:
                layer_output = drop(layer_output)

            layer_input = layer_output

        h_n = Tensor.stack(final_h, axis=0)
        c_n = Tensor.stack(final_c, axis=0)
        return layer_output, (h_n, c_n)

    # ── direction runner ──────────────────────────────────────────────────────

    def _run_direction(self, x, layer, direction, h_init, c_init, reverse=False):
        """
        Run one direction of one layer across the full sequence.

        Returns:
            outputs : Tensor (N, T, out_size)
            h_last  : Tensor (N, out_size)
            c_last  : Tensor (N, hidden_size)
        """

        N, T, _ = x.shape
        cell = self._cell(layer, direction)  # ← live lookup every call

        W_ih = cell["W_ih"]
        W_hh = cell["W_hh"]
        b = self._get_bias(cell)

        h, c = h_init, c_init
        step_outputs = []
        seq = range(T - 1, -1, -1) if reverse else range(T)

        for t in seq:
            x_t = x[:, t, :]
            packed = LSTMCellFunction.apply(x_t, h, c, W_ih, W_hh, b)
            h_new = packed[0]
            c_new = packed[1]

            if self.proj_size > 0:
                h_new = h_new @ cell["W_hr"].T

            h, c = h_new, c_new
            step_outputs.append(h_new)

        if reverse:
            step_outputs = step_outputs[::-1]

        return Tensor.stack(step_outputs, axis=1), h, c

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_bias(self, cell):
        if self.use_bias:
            return cell["b_ih"] + cell["b_hh"]
        return Tensor.zeros((4 * self.hidden_size,))

    def _init_states(self, states, N):
        total = self.num_layers * self.num_directions
        H = self.hidden_size
        h_dim = self.proj_size if self.proj_size > 0 else H

        if states is None:
            h0 = [Tensor.zeros((N, h_dim)) for _ in range(total)]
            c0 = [Tensor.zeros((N, H)) for _ in range(total)]  # H not P, not h_dim
        else:
            h_tensor, c_tensor = states
            h0 = [h_tensor[i] for i in range(total)]
            c0 = [c_tensor[i] for i in range(total)]

        return h0, c0

    def reset_parameters(self):
        H = self.hidden_size
        scale = 1.0 / np.sqrt(H)

        for layer, direction in self._cell_keys:
            cell = self._cell(layer, direction)
            cell["W_ih"].data[:] = np.random.uniform(-scale, scale, cell["W_ih"].shape)
            cell["W_hh"].data[:] = np.random.uniform(-scale, scale, cell["W_hh"].shape)
            if self.use_bias:
                cell["b_ih"].data[:] = 0.0
                cell["b_ih"].data[H : 2 * H] = 1.0
                cell["b_hh"].data[:] = 0.0

    def __repr__(self):
        return (
            f"LSTM(hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, "
            f"bidirectional={self.bidirectional}, "
            f"proj_size={self.proj_size}, "
            f"dropout={self.dropout_rate})"
        )


class LSTM2(Base):
    """
    Multi-layer, bidirectional LSTM.

    Args:
        hidden_size   : number of features in hidden state
        num_layers    : number of stacked LSTM layers            (default 1)
        bias          : if False, no bias terms                  (default True)
        dropout       : dropout probability between layers        (default 0.0)
        bidirectional : if True, process sequence both ways       (default False)
        proj_size     : if > 0, project hidden state to proj_size (default 0)

    Input:
        x             : (batch, seq_len, input_size)
        h0, c0        : optional initial states
                        each (num_layers * num_directions, batch, hidden_size)
                        or (num_layers * num_directions, batch, proj_size) for h0
                        when proj_size > 0

    Returns:
        output        : (batch, seq_len, hidden_size * num_directions)
                        or (batch, seq_len, proj_size * num_directions)
        (h_n, c_n)    : final states, same shape as (h0, c0)
    """

    def __init__(
        self,
        hidden_size,
        num_layers=1,
        bias=True,
        dropout=0.0,
        bidirectional=False,
        proj_size=0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = bias
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self.num_directions = 2 if bidirectional else 1
        self.out_size = proj_size if proj_size > 0 else hidden_size
        self._dropouts = []
        # _cells is now just a registry of parameter KEY NAMES, not Tensor refs
        # shape: {(layer, direction): {"W_ih": "W_ih_L0_D0", "W_hh": ..., ...}}
        self._cell_keys = {}

    # ── build ─────────────────────────────────────────────────────────────────

    def build(self, in_shape):
        input_size = in_shape[-1]

        for layer in range(self.num_layers):
            layer_input_size = (
                input_size if layer == 0 else self.out_size * self.num_directions
            )
            for d in range(self.num_directions):
                self._build_cell(layer, d, layer_input_size)

            if self.dropout_rate > 0 and layer < self.num_layers - 1:
                drop = self.add_layer(f"dropout_{layer}", Dropout(self.dropout_rate))
                self._dropouts.append(drop)
            else:
                self._dropouts.append(None)

    def _build_cell(self, layer, direction, input_size):
        H = self.hidden_size
        label = f"L{layer}_D{direction}"
        scale = 1.0 / np.sqrt(H)

        W_ih = np.random.uniform(-scale, scale, (4 * H, input_size))
        W_hh = np.random.uniform(-scale, scale, (4 * H, H))

        b_ih = np.zeros(4 * H)
        b_ih[H : 2 * H] = 1.0  # forget gate bias
        b_hh = np.zeros(4 * H)

        # register in _parameters exactly as before
        self.add_parameter(f"W_ih_{label}", Tensor(W_ih, requires_grad=True))
        self.add_parameter(f"W_hh_{label}", Tensor(W_hh, requires_grad=True))

        keys = {"W_ih": f"W_ih_{label}", "W_hh": f"W_hh_{label}"}

        if self.use_bias:
            self.add_parameter(f"b_ih_{label}", Tensor(b_ih, requires_grad=True))
            self.add_parameter(f"b_hh_{label}", Tensor(b_hh, requires_grad=True))
            keys["b_ih"] = f"b_ih_{label}"
            keys["b_hh"] = f"b_hh_{label}"

        if self.proj_size > 0:
            W_hr = np.random.uniform(
                -1 / np.sqrt(H), 1 / np.sqrt(H), (self.proj_size, H)
            )
            self.add_parameter(f"W_hr_{label}", Tensor(W_hr, requires_grad=True))
            keys["W_hr"] = f"W_hr_{label}"

        # store only the KEY NAMES — never the Tensor objects themselves
        self._cell_keys[(layer, direction)] = keys

    # ── live cell accessor — always reads from _parameters ───────────────────

    def _cell(self, layer, direction):
        """
        Returns a dict of live Tensor references by reading _parameters now.
        This means after set_parameters() updates _parameters, every subsequent
        _cell() call automatically sees the loaded weights.
        """
        keys = self._cell_keys[(layer, direction)]
        return {role: self._parameters[pkey] for role, pkey in keys.items()}

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x, states=None):
        """
        x      : Tensor (batch, seq_len, input_size)
        states : (h0, c0) each Tensor
                 (num_layers*num_directions, batch, hidden_or_proj_size)
                 or None → zeros
        """

        N, T, _ = x.shape
        num_dir = self.num_directions

        h0, c0 = self._init_states(states, N)

        layer_input = x
        final_h, final_c = [], []

        for layer in range(self.num_layers):
            fwd_out, h_fwd, c_fwd = self._run_direction(
                layer_input,
                layer,
                direction=0,
                h_init=h0[layer * num_dir],
                c_init=c0[layer * num_dir],
            )

            if self.bidirectional:
                bwd_out, h_bwd, c_bwd = self._run_direction(
                    layer_input,
                    layer,
                    direction=1,
                    h_init=h0[layer * num_dir + 1],
                    c_init=c0[layer * num_dir + 1],
                    reverse=True,
                )
                layer_output = Tensor.concat([fwd_out, bwd_out], axis=2)
                final_h.extend([h_fwd, h_bwd])
                final_c.extend([c_fwd, c_bwd])
            else:
                layer_output = fwd_out
                final_h.append(h_fwd)
                final_c.append(c_fwd)

            drop = self._dropouts[layer]
            if drop is not None and self.training:
                layer_output = drop(layer_output)

            layer_input = layer_output

        h_n = Tensor.stack(final_h, axis=0)
        c_n = Tensor.stack(final_c, axis=0)
        return layer_output, (h_n, c_n)

    # ── direction runner ──────────────────────────────────────────────────────

    def _run_direction(self, x, layer, direction, h_init, c_init, reverse=False):
        """
        Run one direction of one layer across the full sequence.

        Returns:
            outputs : Tensor (N, T, out_size)
            h_last  : Tensor (N, out_size)
            c_last  : Tensor (N, hidden_size)
        """

        N, T, _ = x.shape
        cell = self._cell(layer, direction)  # ← live lookup every call

        W_ih = cell["W_ih"]
        W_hh = cell["W_hh"]
        b = self._get_bias(cell)

        step_outputs = []
        seq = range(T - 1, -1, -1) if reverse else range(T)

        h_internal = h_init  # always (N, H)
        c = c_init

        for t in seq:
            x_t = x[:, t, :]
            packed = LSTMCellFunction.apply(x_t, h_internal, c, W_ih, W_hh, b)

            h_new_internal = packed[0]  # (N, H)
            c_new = packed[1]

            # projection (only for output)
            if self.proj_size > 0:
                W_hr = cell["W_hr"]
                h_out = h_new_internal @ W_hr.transpose()  # (N, P)
            else:
                h_out = h_new_internal

            h_internal = h_new_internal  # IMPORTANT
            c = c_new

            step_outputs.append(h_out)

        if reverse:
            step_outputs = step_outputs[::-1]

        return Tensor.stack(step_outputs, axis=1), h_internal, c

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_bias(self, cell):
        if self.use_bias:
            return cell["b_ih"] + cell["b_hh"]
        return Tensor.zeros((4 * self.hidden_size,))

    def _init_states(self, states, N):
        """
        Return (h0, c0) as lists of Tensors indexed by
        [layer * num_directions + direction].
        """
        total = self.num_layers * self.num_directions
        H = self.hidden_size
        P = self.proj_size if self.proj_size > 0 else H

        if states is None:
            h0 = [Tensor.zeros((N, N)) for _ in range(total)]
            c0 = [Tensor.zeros((N, H)) for _ in range(total)]
        else:
            h_tensor, c_tensor = states
            h0 = [h_tensor[i] for i in range(total)]
            c0 = [c_tensor[i] for i in range(total)]
        return h0, c0

    def reset_parameters(self):
        for layer, direction in self._cell_keys:
            cell = self._cell(layer, direction)
            H = self.hidden_size
            scale = 1.0 / np.sqrt(H)
            cell["W_ih"].data[:] = np.random.uniform(-scale, scale, cell["W_ih"].shape)
            cell["W_hh"].data[:] = np.random.uniform(-scale, scale, cell["W_hh"].shape)
            if self.use_bias:
                cell["b_ih"].data[:] = 0.0
                cell["b_ih"].data[H : 2 * H] = 1.0
                cell["b_hh"].data[:] = 0.0

    def __repr__(self):
        return (
            f"LSTM(hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, "
            f"bidirectional={self.bidirectional}, "
            f"proj_size={self.proj_size}, "
            f"dropout={self.dropout_rate})"
        )
