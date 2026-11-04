def make_grad(func, argnums=None, value=False):
    def grad_fn(*args):
        _argnums = argnums
        if not _argnums:
            _argnums = list(range(len(args)))

        assert len(args) > max(_argnums), (
            f"return grad indexes {_argnums} but {len(args)} argument(s) provided cannot access argument indexed {max(_argnums)}"
        )

        for i, input_tensor in enumerate(args):
            if i in _argnums:
                input_tensor.requires_grad_(True)

        output = func(*args)
        output.backward()

        grads = []
        for i in _argnums:
            for j, t in enumerate(args):
                if i == j:
                    grads.append(t.grad.copy()) 
                    t.zero_grad()

        if len(grads) == 1:
            grads = grads[0]
        else:
            grads = tuple(grads)

        if value:
            return grads, output
        return grads

    return grad_fn