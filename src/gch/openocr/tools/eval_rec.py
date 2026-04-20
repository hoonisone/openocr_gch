# from openocr.tools.engine.config import Config
from numbers import Real

from gch.openocr.tools.engine.config import Config
from gch.openocr.tools.engine.trainer import GCHTrainer as Trainer
from gch.openocr.tools.utility import ArgsParser
from gch import RMFactory

def _flatten_dict(data, parent_key="", sep="."):
    flat = {}
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, parent_key=new_key, sep=sep))
        else:
            flat[new_key] = value
    return flat


def _round_nested_metric(data, ndigits=4):
    if isinstance(data, dict):
        return {k: _round_nested_metric(v, ndigits=ndigits) for k, v in data.items()}
    if isinstance(data, list):
        return [_round_nested_metric(v, ndigits=ndigits) for v in data]
    if isinstance(data, tuple):
        return tuple(_round_nested_metric(v, ndigits=ndigits) for v in data)
    if isinstance(data, Real) and not isinstance(data, bool):
        return round(data, ndigits)
    return data


def _expand_dotted_keys(data, sep="."):
    def _merge_dict(dst, src):
        for k, v in src.items():
            if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
                _merge_dict(dst[k], v)
            else:
                dst[k] = v

    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            value = _expand_dotted_keys(value, sep=sep)
            key_parts = str(key).split(sep) if isinstance(key, str) and sep in key else [key]

            cursor = result
            for part in key_parts[:-1]:
                if part not in cursor or not isinstance(cursor[part], dict):
                    cursor[part] = {}
                cursor = cursor[part]

            leaf = key_parts[-1]
            if leaf in cursor and isinstance(cursor[leaf], dict) and isinstance(value, dict):
                _merge_dict(cursor[leaf], value)
            else:
                cursor[leaf] = value
        return result

    if isinstance(data, list):
        return [_expand_dotted_keys(v, sep=sep) for v in data]
    if isinstance(data, tuple):
        return tuple(_expand_dotted_keys(v, sep=sep) for v in data)
    return data

def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        '--work_id',
        type=int,
        required=True,
        help='Work ID',
    )
    parser.add_argument(
        '--task_id',
        type=int,
        required=True,
        help='Task ID',
    )
    args = parser.parse_args()
    return args


def main():
    FLAGS = parse_args()
    FLAGS = vars(FLAGS)
    work_id = FLAGS.pop("work_id")
    task_id = FLAGS.pop("task_id")
    config_path = FLAGS.pop("config")
    opt = FLAGS.pop('opt')



    context = RMFactory().get_deep_learning_context()
    
    task_context = context.get_eval_task_context(work_id, task_id)
    if task_context.is_evaluated():
        print(f"(work {work_id}, task {task_id}) is already evaluated")
        return
    
    config = context.get_eval_task_config(work_id, task_id)

    cfg = Config(config)
    
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    trainer = Trainer(cfg, mode='eval')

    best_model_dict = trainer.status.get('metrics', {})
    trainer.logger.info('metric in ckpt ***************')
    for k, v in best_model_dict.items():
        trainer.logger.info('{}:{}'.format(k, v))

    metric = trainer.eval()\
    
    metric = _flatten_dict(metric)

    trainer.logger.info('metric eval ***************')
    for k, v in metric.items():
        trainer.logger.info('{}:{}'.format(k, v))




    metric = _round_nested_metric(metric, ndigits=4)
    metric = _expand_dotted_keys(metric, sep=".")

    
        
    context.get_eval_task_context(work_id, task_id).save_eval_result(metric)

if __name__ == '__main__':
    main()
