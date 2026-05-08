class Global_grad:
    _grad_enabled_stack = [True]
    _global_grad_blockade = False

    @classmethod
    def stop_global_grad(cls):
        cls._global_grad_blockade = True

    @classmethod
    def release_global_grad(cls):
        cls._global_grad_blockade = False

    @classmethod
    def is_grad_enabled(cls):
        if cls._global_grad_blockade:
            return False
        return cls._grad_enabled_stack[-1]

    @classmethod
    def set_grad_enabled(cls, mode):
        if cls._global_grad_blockade:
            print(
                """
                   RuntimeWarning:
                   Global grad is currently Stopped 
                   use nerux.release_global_grad() 
                   to release it """
            )
        cls._grad_enabled_stack.append(mode)

    @classmethod
    def restore_grad_enabled(cls):
        cls._grad_enabled_stack.pop()


class No_grad:
    def __enter__(self):
        Global_grad.set_grad_enabled(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        Global_grad.restore_grad_enabled()
