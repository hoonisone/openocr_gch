from .cfg import CustomConfigManagerFactory, DBReferHandler, ConfigManager
from .db import DB, WorkDBFactory, TaskDBFactory, DBFactory

__all__ = ['CustomConfigManagerFactory', 'DB', 'WorkDBFactory', 'TaskDBFactory']


from pathlib import Path
from fstdb import TreeDB, TreeNode, Record

from functools import cache, cached_property

class RMFactory:
    def __init__(self):
        pass

    @property
    def db_root_dir_path(self)->Path:
        return Path("/home/resource/3_rsc3")

    @cached_property
    def work_db_factory(self)->DBFactory:
        return WorkDBFactory(TreeNode, DB, Record)

    @cached_property
    def task_db_factory(self)->DBFactory:
        return TaskDBFactory(TreeNode, DB, Record)

    @cached_property
    def work_db(self)->DB:
        return self.work_db_factory.create_tree_db(self.db_root_dir_path)

    @cached_property
    def db_config_refer_handler(self)->DBReferHandler:
        return DBReferHandler(work_db=self.work_db)

    @cached_property
    def config_manager_factory(self)->CustomConfigManagerFactory:
        return CustomConfigManagerFactory(db_refer_handler=self.db_config_refer_handler)

    @cached_property
    def config_manager(self)->ConfigManager:
        return self.config_manager_factory.config_manager
    # @cache
    # def task_db(self)->DB:
    #     return self.make_task_db_factory().create_db(self.db_root_dir_path)