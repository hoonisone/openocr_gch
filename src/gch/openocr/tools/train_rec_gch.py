import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))


from gch.openocr.tools.engine.config import Config
from gch.openocr.tools.engine.trainer import GCHTrainer as Trainer
from gch.openocr.tools.utility import ArgsParser

from gch import RMFactory

def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        '--eval',
        action='store_true',
        default=True,
        help='Whether to perform evaluation in train',
    )
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
    config = context.get_train_task_config(work_id, task_id)
    cfg = Config(config)
    
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)



    trainer =  Trainer(cfg,
                      mode='train_eval' if FLAGS['eval'] else 'train',
                      task='rec')
    trainer.train()


if __name__ == '__main__':
    main()
