from types import ClassMethodDescriptorType
from .rm.cfg import ConfigManager
from .rm.db import DB, TaskDBFactory, TreeNode, Record as GCHRecord
from pathlib import Path
from typing import Optional, Dict, Any, List

from functools import cached_property
from textwrap import dedent
from .tools.command_manager import CommandManager

class DeepLearningContext:
    def __init__(self, 
            work_db:DB, 
            config_manager:ConfigManager, 
            task_db_factory:TaskDBFactory
        ):
        
        self.work_db = work_db
        self.config_manager = config_manager
        self.task_db_factory = task_db_factory

    def get_work(self, work_id)->GCHRecord:
        return self.work_db.get_record(work_id)

    def get_task_db(self, work_id)->DB:
        work = self.get_work(work_id)
        return self.task_db_factory.create_tree_db(work.path)
    
    def get_task(self, work_id, task_id)->GCHRecord:
        return self.get_task_db(work_id).get_record(task_id)

    def make_train_task(self, work_id):
        task_db = self.get_task_db(work_id)
        task_db.create_record("train/")

    def get_work_context(self, work_id)->'WorkContext':
        return WorkContext(
            task_db_factory=self.task_db_factory,
            work_record=self.get_work(work_id),
            config_manager=self.config_manager
        )

    def get_train_task_context(self, work_id, task_id)->'TrainTaskContext':
        work_context = self.get_work_context(work_id)
        task = self.get_task(work_id, task_id)
        return work_context.get_train_task_context(task)

    def get_train_task_config(self, work_id, task_id)->Any:
        work_context = self.get_work_context(work_id)
        task_context = work_context.get_train_task_context(self.get_task(work_id, task_id))
        return task_context.config
        
    def get_eval_task_context(self, work_id, task_id)->'EvalTaskContext':
        work_context = self.get_work_context(work_id)
        task = self.get_task(work_id, task_id)
        return work_context.get_eval_task_context(task)
    
    def get_eval_task_config(self, work_id, task_id)->Any:
        work_context = self.get_work_context(work_id)
        task_context = work_context.get_eval_task_context(self.get_task(work_id, task_id))
        return task_context.config

class WorkContext:
    TRAIN_TASK_ID = 1


    def __init__(self,
            task_db_factory:TaskDBFactory,
            work_record:GCHRecord,
            config_manager:ConfigManager,
        ):
        self.task_db_factory = task_db_factory
        self.work_record = work_record
        self.config_manager = config_manager

    

    @cached_property
    def path(self)->Path:
        return self.work_record.path
    
    @cached_property
    def task_db(self):
        return self.task_db_factory.create_tree_db(self.path)

    @cached_property
    def prop_path(self)->Path:
        return self.path / "prop.py"

    @cached_property
    def prop(self):
        return self.config_manager.load_config(self.prop_path, handling=True, lazy_handling=False)

    @cached_property
    def config_path(self)->Path:
        return self.work_record.config_file_path

    @property
    def config(self)->dict:
        return self.config_manager.load_config(self.config_path, handling=True, lazy_handling=False)
    



    def get_train_task_context(self, task_record)->'TrainTaskContext':
        
        return TrainTaskContext(
            work_record=self.work_record,
            task_record=task_record, 
            config_manager=self.config_manager
        )

    def get_eval_task_context(self, task_record)->'EvalTaskContext':
        
        return EvalTaskContext(
            work_record=self.work_record,
            task_record=task_record, 
            config_manager=self.config_manager
        )

    def get_train_task_record(self)->Optional[GCHRecord]:
        if self.TRAIN_TASK_ID in self.task_db:
            return self.task_db.get_record(self.TRAIN_TASK_ID)
        else:
            return None

    def make_train_task(self)->GCHRecord:
        train_task = self.get_train_task_record()
        if train_task is None:
            train_task = self.task_db.create_record("train")
            assert train_task.id == self.TRAIN_TASK_ID, "Train task id is not " + str(self.TRAIN_TASK_ID) + " This means that other tasks were created before the train task"
            task_context = self.get_train_task_context(train_task)


            config = dedent(f"""
                from gch import _path, _config, _prop
                from datetime import datetime as _datetime
                _base = (
                    _config({self.work_record.id}),
                    _prop({self.work_record.id})["train_dataset"]
                )
                Wandb = dict(
                    project="OpenOCR-GCH",
                    name="{self.work_record.id}",
                    sync_tensorboard=True,
                )
                """+"""
                _weights_dir = f"{__file__}/../weights"
                _output_dir = f"{__file__}/../{_datetime.now().strftime('%Y%m%d_%H%M%S')}"
                _save_res_path = f"{_output_dir }/output/res.txt"

                from pathlib import Path as _Path

                _pretrained_model = None

                _checkpoints = (_Path(_weights_dir)/"latest.pth").resolve()
                _checkpoints = _checkpoints.as_posix() if _checkpoints.exists() else None
            """)



            
            task_context.write_config(config)


            prop = dedent(f"""
                epoch = {None}
                type = "train"
            """)

            task_context.write_prop(prop)

            return train_task
        else:
            return train_task


    @cached_property
    def all_tasks(self)->Dict:
        tasks = {}
        for id in self.task_db.ids:
            record = self.task_db.get_record(id)
            context = self.get_train_task_context(record)
            task_type = context.prop['type']
            dataset_id = context.prop.get('dataset_id', None)
            epoch = context.prop.get('epoch', None)


            if task_type == "train":
                assert "train" not in tasks, "Train task already exists"
                tasks["train"] = context
            elif task_type == "test":
                assert epoch is not None, "Eval task must have epoch in prop, but it's not set"
                assert dataset_id is not None, "Eval task must have dataset id in prop, but it's not set"
                tasks.setdefault("test", {}).setdefault(dataset_id, {})[epoch] = context
            else:
                raise ValueError(f"Invalid task type: {context.prop['type']}")
        return tasks


    def make_eval_record_name(self, eval_dataset_id, epoch)->str:
        return f"test/dataset_{eval_dataset_id}/epoch_{epoch}"

    def get_eval_task_record(self, eval_dataset_id, epoch)->Optional[GCHRecord]:
        """
            task_record가 존재하면 반환
            없으면 None 반환
        """
        all_tasks = self.all_tasks
        if "test" not in all_tasks:
            return None
        if eval_dataset_id not in all_tasks["test"]:
            return None
        if epoch not in all_tasks["test"][eval_dataset_id]:
            return None
        return all_tasks["test"][eval_dataset_id][epoch]
        
    def make_eval_task(self, eval_dataset_id, epoch):
        """
            check_weight: epoch에 대한 weight 이 존재하는 경우에만 task 생성
        """
        record = self.get_eval_task_record(eval_dataset_id, epoch)
        if record is None:
            record_name = self.make_eval_record_name(eval_dataset_id, epoch)
            record = self.task_db.create_record(record_name)
            eval_context = self.get_eval_task_context(record)
            eval_context.initialize(eval_dataset_id, epoch)
    
        return record

    def make_all_eval_tasks(self)->None:
        prop = self.prop
        for dataset_id in prop["eval_dataset_ids"]:
            for id in prop["eval_epochs"]:
                record = self.make_eval_task(dataset_id, id)



    def get_all_eval_command(self, check_weight:bool=True):
        commands = []

        train_task_context = self.get_train_task_context(self.get_train_task_record())
        for id in self.task_db.ids:
            record = self.task_db.get_record(id)
            context = self.get_eval_task_context(record)
            prop = context.prop
            task_type = prop['type']
            epoch = prop['epoch']
            if task_type != "test":
                continue
            if context.is_evaluated():
                continue
            if check_weight and not train_task_context.does_weight_exists(epoch):
                continue
            commands.append(context.make_eval_command())
        return commands




class TaskContext:
    def __init__(self, 
        work_record:GCHRecord,
        task_record:GCHRecord,
        config_manager:ConfigManager
    ):
        self.work_record = work_record
        self.task_record = task_record
        self.config_manager = config_manager

    def get_task_path(self)->GCHRecord:
        return self.task_record

    @cached_property
    def config_path(self)->Path:
        return self.task_record.path / "config.py"
    
    @cached_property
    def prop_path(self)->Path:
        return self.task_record.path / "prop.py"

    @cached_property
    def config(self):
        return self.config_manager.load_config(self.config_path, handling=True, lazy_handling=True)

    @property
    def prop(self):
        return self.config_manager.load_config(self.prop_path, handling=False)

    def write_config(self, config:str)->None:
        with open(self.config_path, "w") as f:
            f.write(config)


    def write_prop(self, text:str)->None:
        with open(self.prop_path, "w") as f:
            f.write(text)


    def make_eval_command(self)->str:
        return f"python /home/src/gch/openocr/tools/eval_rec.py --work_id {self.work_record.id} --task_id {self.task_record.id}"




class TrainTaskContext(TaskContext):
    def make_train_command(self)->str:
        return f"python /home/src/gch/openocr/tools/train_rec_gch.py --work_id {self.work_record.id} --task_id {self.task_record.id}"
    

    def does_weight_exists(self, epoch:int)->bool:
        return self.weight_path(epoch).exists()


    @cached_property
    def weight_dir(self)->Path:
        return self.task_record.path / "weights"

    def weight_path(self, epoch:int)->Path:
        return self.weight_dir / f"epoch_{epoch}.pth"

    @cached_property
    def weight_epoches(self)->List[int]:
        paths = self.weight_dir.rglob("*.pth")
        return [int(path.stem.split("_")[-1]) for path in paths]
    


class EvalTaskContext(TaskContext):

    def initialize(self, eval_dataset_id, epoch)->None:

        config = dedent(f"""
                from gch import _path, _config, _prop
                from datetime import datetime as _datetime

                _base = (
                    _config({self.work_record.id}),
                    _prop({self.work_record.id})["eval_datasets"][{eval_dataset_id}]
                )

                _weights_dir = _path({self.work_record.id})/"train___id_1/weights"
                _pretrained_model = _weights_dir/"epoch_{epoch}.pth"
                """+"""
                _output_dir = f"{__file__}/../{_datetime.now().strftime('%Y%m%d_%H%M%S')}"
                _save_res_path = f"{_output_dir }/output/res.txt"

                _train_data_dir_list = None
                _train_label_file_list = None
            """)


        self.write_config(config)

        prop = dedent(f"""
            type = "test"
            epoch = {epoch}
            dataset_id = {eval_dataset_id}
        """)

        self.write_prop(prop)


    def make_train_command(self)->str:
        return f"python /home/src/gch/openocr/tools/train_rec_gch.py --work_id {self.work_record.id} --task_id {self.task_record.id}"
    
    @cached_property
    def eval_result_path(self)->Path:
        return self.task_record.path / "eval_result.yml"

    def eval_result(self)->Optional[str]:
        import yaml

        path = self.eval_result_path
        
        with open(path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)


    def save_eval_result(self, result:dict)->None:
        import yaml
        path = self.eval_result_path
        with open(path, "w") as f:
            yaml.dump(result, f)

    def is_evaluated(self)->bool:
        if not self.eval_result_path.exists():
            return False
        if self.eval_result() is None:
            return False
        if len(self.eval_result()) == 0:
            return False
        return True