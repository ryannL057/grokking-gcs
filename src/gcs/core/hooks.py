"""Adapt the MIT TransformerLens hooks to the research callback interface.

TransformerLens 2.16.1 implements registration, removal and execution. Its
copyright and license are retained in gcs/licenses/TransformerLens-LICENSE.
"""
from transformer_lens.hook_points import HookPoint as NativeHookPoint


def legacy_hook_name(name):
    return name.replace('hook_pattern', 'hook_attn').replace('hook_attn_scores', 'hook_attn_pre')


class HookPoint(NativeHookPoint):
    """MIT hook with callbacks accepting ``value, name=...``."""

    def give_name(self, name):
        self.name = name

    def add_hook(self, callback, dir='fwd', **kwargs):
        def invoke(value, hook):
            return callback((value,) if dir == 'bwd' else value, name=hook.name)
        return super().add_hook(invoke, dir=dir, **kwargs)


class HookView:
    """Expose checkpoint-era names and head/position axes for a native hook."""

    def __init__(self, point):
        self.native = point
        self.name = legacy_hook_name(point.name)

    def __call__(self, value):
        return self.native(value)

    def add_hook(self, callback, dir='fwd'):
        transpose = self.name.endswith(('hook_q', 'hook_k', 'hook_v', 'hook_z'))

        def invoke(value, hook):
            legacy = value.transpose(1, 2) if transpose else value
            result = callback((legacy,) if dir == 'bwd' else legacy, name=self.name)
            if result is None:
                return None
            if dir == 'bwd':
                result = result[0] if isinstance(result, tuple) else result
                return (result.transpose(1, 2) if transpose else result,)
            return result.transpose(1, 2) if transpose else result

        self.native.add_hook(invoke, dir=dir)

    def remove_hooks(self, dir='fwd'):
        # Architectural attention interpolation is a permanent native hook.
        self.native.remove_hooks(dir=dir, including_permanent=False)


class HookCacheMixin:
    """Shared activation-cache API for the authors' models."""

    def _name_hooks(self):
        for name, point in self.named_modules():
            if isinstance(point, NativeHookPoint):
                point.name = name

    def hook_points(self):
        return [point for point in self.modules() if isinstance(point, HookPoint)]

    @property
    def hook_dict(self):
        return {point.name: point for point in self.hook_points()}

    def remove_all_hooks(self):
        for point in self.hook_points():
            point.remove_hooks('both')

    def cache_all(self, cache, incl_bwd=False):
        def forward(value, name):
            cache[name] = value.detach()

        def backward(value, name):
            cache[name + '_grad'] = value[0].detach()

        for point in self.hook_points():
            point.add_hook(forward)
            if incl_bwd:
                point.add_hook(backward, dir='bwd')
