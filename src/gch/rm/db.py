from fstdb import DBFactory, TreeDB, TreeNode
from fstdb.db.record import Record as _Record
from fstdb.db.path_manager import RecordPath

from pathlib import Path

class Record(_Record):
    @property
    def config_file_path(self)->Path:
        if self.path.is_dir():
            return self.path/"config.py"
        else:
            return self.path

    @property
    def prop_file_path(self)->Path:
        if self.path.is_dir():
            return self.path/"prop.py"
        else:
            raise ValueError("")


class DB(TreeDB[TreeNode, Record]):

    pass
    # def get_sub_db(self, name: RecordPath) -> 'DB':
    #     """하위 데이터베이스를 가져옵니다."""
    #     # self.__class__를 사용하여 서브클래스도 올바르게 반환되도록 함
    #     tokens = self.record_context.path_manager.tokenize(name)
    #     node = self.tree_context.get_child(self.tree, tokens)
    #     return self.__class__(
    #         path=None,
    #         tree=node, 
    #         tree_db_context=self.record_context, 
    #         tree_context=self.tree_context, 
    #         NodeClass=self.NodeClass, 
    #         RecordClass=self.RecordClass
    #         )


WorkDBFactory = DBFactory[TreeNode, Record, DB]
TaskDBFactory = DBFactory[TreeNode, Record, DB]

# def get_db_factory()->WorkDBFactory:
#     return WorkDBFactory(TreeNode, DB, Record)