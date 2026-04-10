from openocr.tools.engine.config import Config as OpenOCRConfig
from gch import RMFactory

rm_factory = RMFactory()
config_manager = rm_factory.config_manager


class Config(OpenOCRConfig):

    def __init__(self, config_path:str, BASE_KEY='_BASE_'):
        self.BASE_KEY = BASE_KEY
        if isinstance(config_path, str):
            self.cfg = self._load_config_with_base(config_path)
        else:
            self.cfg = config_path

    def _load_config_with_base(self, file_path):
        return config_manager.load_config(file_path)

    def save(self, p, cfg=None):
        import yaml
        with open(p, 'w') as f:
            yaml.dump(cfg, f)
        # config_manager.dict_accessor.dump_config(config=cfg, path=p)
