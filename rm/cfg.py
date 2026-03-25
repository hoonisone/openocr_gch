from config_framework import ConfigHandler, ConfigManager, DefaultConfigManagerFactory

from config_framework.handler import FileConfigReferHandler, TupleMergeHandler, BaseFlatHandler, FunctionHandler, ValueRemoveHandler, ValueReferHandler

from typing import Optional, Hashable, List
from pathlib import Path
from .db import DB

from config_framework.context import RecursiveContext

class DBReferHandler(ConfigHandler):
    # db id로 참조된 recode의 config file 경로를 찾아 @file_cif로 참조 변경 수행행

    def __init__(self,
            work_db:DB
        )->None:

        self.work_db = work_db

    def is_target(self, string:Optional[Hashable])->bool:
        return isinstance(string, str) and string.startswith("@db_cfg:")

    def get_config_file_path(self, string:str)->str|Path:
        # config 참조 string에서 config_file_path를 반환환
        type, address = string.split(":")
        db_name:str = address.split("/")[0]
        id = int(address.split("/")[1])

        return self.work_db.get_record(id).config_file_path
    
    def handle(self, x:dict)->dict:
        return RecursiveContext.replace(
            data = x,
            is_target = lambda v, k, idx: self.is_target(v),
            replacement = lambda v, k, idx: f"{FileConfigReferHandler.MARK}{self.get_config_file_path(v)}"
        )


class CustomConfigManagerFactory(DefaultConfigManagerFactory):
    def __init__(self, db_refer_handler:DBReferHandler):
        self.db_refer_handler = db_refer_handler

    def make_config_handlers(self, config_manager:ConfigManager)->List[ConfigHandler]:
        return [
            self.db_refer_handler,
            FileConfigReferHandler(config_manager=config_manager),
            TupleMergeHandler(dict_tool=self.dict_tool),
            BaseFlatHandler(dict_tool=self.dict_tool),
            FunctionHandler(),
            ValueReferHandler(dict_tool=self.dict_tool),
            ValueRemoveHandler(),
            TupleMergeHandler(dict_tool=self.dict_tool),
        ]

