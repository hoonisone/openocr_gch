from cfgfw import ConfigHandler, ConfigManager, DefaultConfigManagerFactory

from cfgfw.handler import FileConfigReferHandler, TupleMergeHandler, BaseFlatHandler, FunctionHandler, ValueRemoveHandler, ValueReferHandler, EmptyValueCheckHandler

from typing import Optional, Hashable, List, Any
from pathlib import Path
from .db import DB

from cfgfw.context import RecursiveContext
from cfgfw.tool import DictTool


class DBReferHandler(ConfigHandler):
    # db id로 참조된 recode의 config file 경로를 찾아 @file_cif로 참조 변경 수행행

    def __init__(self,
            work_db:DB
        )->None:

        self.work_db = work_db
        self._config_manager:Optional[ConfigManager] = None

    def is_target(self, string:Optional[Hashable])->bool:
        return isinstance(string, str) and string.startswith("@db_cfg:")

    def get_cfg(self, string:str)->Any:
        # config 참조 string에서 config_file_path를 반환환
        tokens = string.split(":")
        type = tokens[0] 
        id = int(tokens[1])

        cfg_file_path = self.work_db.get_record(id).config_file_path 
        cfg = self.config_manager.load_config(cfg_file_path, handling=True, lazy_handling=False)
        
        if len(tokens) == 3:
            keys = tokens[2].split(".")

            cfg = DictTool(None).get(cfg, keys)
        
        return cfg
    
    @property
    def config_manager(self)->ConfigManager:
        if self._config_manager is None:
            raise ValueError("Config manager is not set, DBReferHandler must be initialized with a config manager")
        else:
            return self._config_manager

    def handle(self, config:dict)->Any:
        return RecursiveContext.replace(
            data = config,
            is_target = lambda v, k, idx: self.is_target(v),
            replacement = lambda v, k, idx: self.get_cfg(v)
        )


class CustomConfigManagerFactory(DefaultConfigManagerFactory):
    def __init__(self, db_refer_handler:DBReferHandler):
        self.db_refer_handler = db_refer_handler

    def make_config_handlers(self, config_manager:ConfigManager)->List[ConfigHandler]:
        self.db_refer_handler._config_manager = config_manager
        return [
            self.db_refer_handler,
            FileConfigReferHandler(config_manager=config_manager, dict_tool=self.dict_tool),
            TupleMergeHandler(dict_tool=self.dict_tool),
            BaseFlatHandler(dict_tool=self.dict_tool),
            FunctionHandler(),
        ]

    def make_laze_config_handlers(self, config_manager:ConfigManager)->List[ConfigHandler]:
        return [
            ValueReferHandler(dict_tool=self.dict_tool),
            ValueRemoveHandler(),
            TupleMergeHandler(dict_tool=self.dict_tool),
            EmptyValueCheckHandler(),
        ]
