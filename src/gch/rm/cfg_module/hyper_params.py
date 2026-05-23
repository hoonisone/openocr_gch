from dataclasses import dataclass as _dataclass
from dataclasses import asdict as _asdict
from dataclasses import field as _field
from cfgfw.empty_tag import EMPTY_TAG as _EMPTY_TAG
from cfgfw.empty_tag import EmptyTag as _EmptyTag

@_dataclass
class STRHyperParams:
    commit_hash:str|_EmptyTag = _field(default_factory=lambda: _EMPTY_TAG, )
    epoch_num:int|_EmptyTag = _field(default_factory=lambda: _EMPTY_TAG, )
    log_step:int|_EmptyTag = _field(default_factory=lambda: _EMPTY_TAG, )
    max_text_length:int|_EmptyTag = _field(default_factory=lambda: _EMPTY_TAG, )

    character_dict_path:str|_EmptyTag = _field(default_factory=lambda: _EMPTY_TAG, )
    train_batch_size:int|_EmptyTag = _field(default_factory=lambda: _EMPTY_TAG, )
    eval_batch_size:int|_EmptyTag = _field(default_factory=lambda: _EMPTY_TAG, )
    
    train_num_workers:int|_EmptyTag = _field(default_factory=lambda: _EMPTY_TAG, )
    eval_num_workers:int|_EmptyTag = _field(default_factory=lambda: _EMPTY_TAG, )
    warmup_epoch:float|_EmptyTag = _field(default_factory=lambda: _EMPTY_TAG, )