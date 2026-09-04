import os
import importlib
import inspect
from torch.optim.lr_scheduler import _LRScheduler

def get_all_schedulers():
    return [
        scheduler.split('.')[0]
        for scheduler in os.listdir('scheduler')
        if scheduler.endswith('.py') and not scheduler.startswith('__')
    ]

scheduler_names = {}
for scheduler in get_all_schedulers():
    try:
        mod = importlib.import_module(f"scheduler.{scheduler}")
        scheduler_class = None
        # 查找当前模块定义的 scheduler 类
        local_classes = []
        for name, obj in inspect.getmembers(mod, inspect.isclass):

            # 必须是当前文件定义的类
            if obj.__module__ != mod.__name__:
                continue

            # 必须继承 PyTorch scheduler
            if not issubclass(obj, _LRScheduler):
                continue

            local_classes.append((name, obj))

        # 注册 scheduler
        for name, cls in local_classes:
            scheduler_name = getattr(cls, "NAME", None)
            if scheduler_name is None:
                print(
                    f"Warning: Scheduler class {name} "
                    f"in {scheduler}.py has no NAME attribute"
                )
                continue
            scheduler_names[scheduler_name] = cls
            print(
                f"Registered scheduler {scheduler_name}: "
                f"{cls.__module__}.{cls.__name__}"
            )
    except Exception as e:
        print(
            f"Warning: Could not import scheduler {scheduler}: {e}"
        )


def get_scheduler(scheduler_name, optimizer, args):
    """
    Build scheduler from NAME.
    Args:
        scheduler_name:
            scheduler NAME
        optimizer:
            torch optimizer
        args:
            experiment arguments

    Returns:
        scheduler instance
    """
    try:
        if scheduler_name not in scheduler_names:
            raise ValueError(
                f"Unknown scheduler: {scheduler_name}. "
                f"Available schedulers: {list(scheduler_names.keys())}"
            )

        return scheduler_names[scheduler_name](
            optimizer,
            args
        )

    except Exception:
        return None