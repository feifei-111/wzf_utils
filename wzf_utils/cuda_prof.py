
class CudaProfilerManager:

    class HookCreator:
        @classmethod
        def trigger_once_hook(cls):
            trigger = False

            def hook():
                nonlocal trigger
                if not trigger:
                    trigger = True
                    return True
                else:
                    return False

            return hook

        @classmethod
        def hasattr_hook(cls, mod, name):
            def hook():
                return hasattr(mod, name)

            return hook

        @classmethod
        def union_hooks(cls, hooks):
            def hook():
                return all(hk() for hk in hooks)

            return hook

    def __init__(self, hook=HookCreator.trigger_once_hook()):
        self.hook = hook
        self.under_profile = False

    def _start(self):
        from cuda import cudart

        cudart.cudaProfilerStart()

    def _stop(self):
        from cuda import cudart

        cudart.cudaProfilerStop()

    def __call__(self, func):
        def inner(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return inner

    def __enter__(self):
        if self.hook() and not self.under_profile:
            self._start()
            self.under_profile = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.under_profile:
            self._stop()
            self.under_profile = False

