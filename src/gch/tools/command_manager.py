from pathlib import Path
from typing import List

class CommandManager:
    def __init__(self, dir_path:Path):
        self.dir_path = dir_path




    def _make_file_path(self, task_type:str, gpu_id:int)->Path:
        assert task_type in ["train", "eval", "test", "infer"]
        return self.dir_path / f"{task_type}{gpu_id}.sh"


    def _split_commands(self, commands:List[str], num_splits:int):
        return [commands[i::num_splits] for i in range(num_splits)]

    def write_commands(self, task_type:str, gpu_ids:int|List[int], commands:List[str], accumalate:bool=False):
        
        if isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]

        command_splits = self._split_commands(commands, len(gpu_ids))


        for to_gpu, commands in zip(gpu_ids, command_splits):

            file_option = "a" if accumalate else "w"

            path = self._make_file_path(task_type, to_gpu)
            with open(path, file_option) as f:
                for command in commands:
                    f.write(command + "\n")


    def clear(self, task_type:str, gpu_id:int):
        path = self._make_file_path(task_type, gpu_id)
        if path.exists():
            path.unlink()
            with open(path, "w") as f:
                f.write("")
        else:
            print(f"Command file {path} does not exist, So it's not cleared, This is not error, Just warning")

    def get_count(self, task_type:str, gpu_id:int)->int:
        path = self._make_file_path(task_type, gpu_id)
        if not path.exists():
            return 0
        else:
            return len(path.read_text().split("\n"))