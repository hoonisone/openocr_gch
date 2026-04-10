# from openocr.tools.engine.config import Config
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




    for k, v in metric.items():
        metric[k] = round(v, 4)
        
    context.get_eval_task_context(work_id, task_id).save_eval_result(metric)

if __name__ == '__main__':
    main()
