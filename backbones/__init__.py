import os
import importlib

import math 
import torch
import torch.nn as nn

import inspect


def get_all_backbones():
    return [backbone.split('.')[0] for backbone in os.listdir('backbones')
            if not backbone.find('__') > -1 and 'py' in backbone]

names = {}
for backbone in get_all_backbones():
    try:
        mod = importlib.import_module(f"backbones.{backbone}")
        backbone_normalized = backbone.replace("_", "").lower()

        class_name = None

        # 只检查当前模块中定义的类，忽略从 torchvision 等地方导入的类
        local_classes = []

        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if not issubclass(obj, nn.Module):
                continue

            # 关键：只保留定义在当前模块里的类
            if obj.__module__ != mod.__name__:
                continue

            local_classes.append((name, obj))

        # 优先按规范化名称精确匹配
        for name, obj in local_classes:
            name_normalized = name.replace("_", "").lower()

            if name_normalized == backbone_normalized:
                class_name = name
                break

        # 没有精确匹配时，仅在当前模块只定义了一个模型类时使用它
        if class_name is None and len(local_classes) == 1:
            class_name = local_classes[0][0]

        if class_name is not None:
            names[backbone] = getattr(mod, class_name)
            print(
                f"Registered {backbone}: "
                f"{names[backbone].__module__}.{names[backbone].__name__}"
            )
        else:
            candidates = [name for name, _ in local_classes]
            print(
                f"Warning: Could not uniquely find backbone class "
                f"in {backbone}.py. Candidates: {candidates}"
            )
    except Exception as e:
        print(f"Warning: Could not import {backbone}: {e}")

#    try:
#        mod = importlib.import_module('backbones.' + backbone)
#        print(mod)
#        # Try to find class with name matching backbone (case insensitive)
#        backbone_lower = backbone.replace('_', '').lower()
#        class_name = None
#        
#        # First try exact match (case insensitive)
#        for x in mod.__dir__():
#            if x.lower() == backbone_lower:
#                class_name = x
#                break
#        
#        
#        # If not found, try to find any class that looks like a backbone
#        if class_name is None:
#            for x in mod.__dir__():
#                obj = getattr(mod, x)
#                if isinstance(obj, type) and issubclass(obj, nn.Module) and x != 'nn.Module':
#                    class_name = x
#                    break
#        
#
#        if class_name is not None:
#            names[backbone] = getattr(mod, class_name)
#        else:
#            print(f"Warning: Could not find backbone class in {backbone}.py")
#    except Exception as e:
#        print(f"Warning: Could not import {backbone}: {e}")

def get_backbone(backbone_name, indim, hiddim, outdim, args):
    """Get the network architectures for encoder, predictor, discriminator."""
    return names[backbone_name](indim, hiddim, outdim, args)
