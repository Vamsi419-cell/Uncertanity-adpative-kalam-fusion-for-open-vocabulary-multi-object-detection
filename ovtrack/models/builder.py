try:
    from mmcv.cnn import build_model_from_cfg as build
    from mmcv.utils import Registry
except ImportError:
    class Registry:
        def __init__(self, name):
            self._name = name
            self._module_dict = dict()

        def register_module(self, name=None, force=False, module=None):
            def _register(cls):
                mod_name = cls.__name__ if name is None else name
                self._module_dict[mod_name] = cls
                return cls
            if module is not None:
                return _register(module)
            return _register

        def get(self, key):
            return self._module_dict.get(key, None)

    def build(cfg, registry, default_args=None):
        if not isinstance(cfg, dict):
            return cfg
        args = cfg.copy()
        obj_type = args.pop('type')
        cls = registry.get(obj_type)
        if cls is None:
            raise KeyError(f"{obj_type} is not in the {registry._name} registry")
        if default_args:
            for k, v in default_args.items():
                args.setdefault(k, v)
        return cls(**args)

MODELS = Registry("model")
TRACKERS = Registry("tracker")
MOTIONS = Registry("motion")
MOTION = MOTIONS

def build_tracker(cfg):
    """Build tracker."""
    return build(cfg, TRACKERS)

def build_motion(cfg):
    """Build motion."""
    return build(cfg, MOTIONS)

def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model."""
    return build(cfg, MODELS, dict(train_cfg=train_cfg, test_cfg=test_cfg))