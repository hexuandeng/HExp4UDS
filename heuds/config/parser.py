# Adopt from fairseq https://github.com/facebookresearch/fairseq

from heuds.constant import TASK_REGISTRY, MODEL_REGISTRY
from heuds.config.base_config import BaseConfig
from dataclasses import dataclass, MISSING
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
from enum import Enum
import argparse
import inspect
import ast
import re


def eval_str_list(x, x_type=float):
    if x is None:
        return None
    if isinstance(x, str):
        if len(x) == 0:
            return []
        x = ast.literal_eval(x)
    try:
        return list(map(x_type, x))
    except TypeError:
        return [x_type(x)]

def map_to_str(x):
    if isinstance(x, list) or isinstance(x, tuple):
        tmp = []
        for i in x:
            if isinstance(i, str):
                tmp.append("'" + i + "'")
            else:
                tmp.append(str(i))
        return tmp
    else:
        return [str(x)]

def interpret_dc_type(field_type):
    if isinstance(field_type, str):
        raise RuntimeError("field should be a type")

    if field_type == Any:
        return str

    typestring = str(field_type)
    if re.match(
        r"(typing.|^)Union\[(.*), NoneType\]$", typestring
    ) or typestring.startswith("typing.Optional"):
        return field_type.__args__[0]
    return field_type


class PytorchParser(object):
    def __init__(self, default_task = None, default_model = None) -> None:
        self.task_name = default_task
        self.model_name = default_model
        self.parser = argparse.ArgumentParser(allow_abbrev=False)
        dataclass = self.get_training_parser()
        self.args, _ = self.parse_args_and_arch()

        self.model = self.gen_dataclass_from_parser(MODEL_REGISTRY[self.model_name][1]())
        self.task = self.gen_dataclass_from_parser(dataclass.task)
        self.dataset = self.gen_dataclass_from_parser(dataclass.dataset)
        self.iterator = self.gen_dataclass_from_parser(dataclass.iterator)
        self.optimization = self.gen_dataclass_from_parser(dataclass.optimization)
        self.checkpoint = self.gen_dataclass_from_parser(dataclass.checkpoint)
        # self.generation = self.gen_dataclass_from_parser(dataclass.generation)

    def get_training_parser(self):
        task_group = self.parser.add_argument_group("Task configuration")
        task_group.add_argument("mission")
        task_group.add_argument('--task', metavar='TASK',
                                help='task architecture')
        model_group = self.parser.add_argument_group("Model configuration")
        model_group.add_argument('--arch', metavar='ARCH',
                                 help='model architecture')

        args, _ = self.parser.parse_known_args()
        self.mission = args.mission
        self.task_name = args.task
        self.model_name = args.arch

        dataclass = TASK_REGISTRY[self.task_name][1]()
        self.gen_parser_from_dataclass(model_group, MODEL_REGISTRY[self.model_name][1]())
        self.gen_parser_from_dataclass(task_group, dataclass.task)
        
        group = self.parser.add_argument_group("dataset configuration")
        self.gen_parser_from_dataclass(group, dataclass.dataset)
        group = self.parser.add_argument_group("iterator configuration")
        self.gen_parser_from_dataclass(group, dataclass.iterator)
        group = self.parser.add_argument_group("optimization configuration")
        self.gen_parser_from_dataclass(group, dataclass.optimization)
        group = self.parser.add_argument_group("checkpoint configuration")
        self.gen_parser_from_dataclass(group, dataclass.checkpoint)
        # group = self.parser.add_argument_group("generation configuration")
        # self.gen_parser_from_dataclass(group, dataclass.generation)

        return dataclass

    def gen_parser_from_dataclass(self,
                                  parser,
                                  dataclass_instance: BaseConfig,
                                  delete_default: bool = False,
                                  with_prefix: Optional[str] = ''
                                  ) -> None:
        """
        convert a dataclass instance to tailing parser arguments.

        If `with_prefix` is provided, prefix all the keys in the resulting parser with it. It means that we are
        building a flat namespace from a structured dataclass (see transformer_config.py for example).
        """
        def argparse_name(name: str):
            if name == "data" and (with_prefix is None or with_prefix == ""):
                # normally data is positional args, so we don't add the -- nor the prefix
                return name
            if name == "_name":
                # private member, skip
                return None
            full_name = "--" + name.replace("_", "-")
            if with_prefix is not None and with_prefix != "":
                # if a prefix is specified, construct the prefixed arg name
                full_name = with_prefix + "-" + \
                    full_name[2:]  # strip -- when composing
            return full_name

        def get_kwargs_from_dc(
            dataclass_instance: BaseConfig, k: str
        ) -> Dict[str, Any]:
            """k: dataclass attributes"""
            kwargs = {}

            field_type = dataclass_instance._get_type(k)
            inter_type = interpret_dc_type(field_type)

            field_default = dataclass_instance._get_default(k)

            if isinstance(inter_type, type) and issubclass(inter_type, Enum):
                field_choices = [t.value for t in list(inter_type)]
            else:
                field_choices = None

            field_help = dataclass_instance._get_help(k)
            field_const = dataclass_instance._get_argparse_const(k)

            if isinstance(field_default, str) and field_default.startswith("${"):
                kwargs["default"] = field_default
            else:
                if field_default is MISSING:
                    kwargs["required"] = True
                if field_choices is not None:
                    kwargs["choices"] = field_choices
                if (
                    isinstance(inter_type, type)
                    and (issubclass(inter_type, List) or issubclass(inter_type, Tuple))
                ) or ("List" in str(inter_type) or "Tuple" in str(inter_type)):
                    if "int" in str(inter_type):
                        kwargs["type"] = lambda x: eval_str_list(x, int)
                    elif "float" in str(inter_type):
                        kwargs["type"] = lambda x: eval_str_list(x, float)
                    elif "str" in str(inter_type):
                        kwargs["type"] = lambda x: eval_str_list(x, str)
                    else:
                        raise NotImplementedError(
                            "parsing of type " +
                            str(inter_type) + " is not implemented"
                        )
                    if field_default is not MISSING:
                        kwargs["default"] = (
                            ",".join(map_to_str(field_default))
                            if field_default is not None
                            else None
                        )
                elif (
                    isinstance(inter_type, type) and issubclass(
                        inter_type, Enum)
                ) or "Enum" in str(inter_type):
                    kwargs["type"] = str
                    if field_default is not MISSING:
                        if isinstance(field_default, Enum):
                            kwargs["default"] = field_default.value
                        else:
                            kwargs["default"] = field_default
                elif inter_type is bool:
                    kwargs["action"] = (
                        "store_false" if field_default is True else "store_true"
                    )
                    kwargs["default"] = field_default
                else:
                    kwargs["type"] = inter_type
                    if field_default is not MISSING:
                        kwargs["default"] = field_default

            # build the help with the hierarchical prefix
            if with_prefix is not None and with_prefix != "" and field_help is not None:
                field_help = with_prefix[2:] + ": " + field_help

            kwargs["help"] = field_help
            if field_const is not None:
                kwargs["const"] = field_const
                kwargs["nargs"] = "?"

            return kwargs

        for k in dataclass_instance._get_all_attributes():
            field_name = argparse_name(dataclass_instance._get_name(k))
            field_type = dataclass_instance._get_type(k)
            if field_name is None:
                continue
            elif inspect.isclass(field_type) and issubclass(field_type, BaseConfig):
                # for fields that are of type BaseDataclass, we can recursively
                # add their fields to the namespace (so we add the args from model, task, etc. to the root namespace)
                prefix = None
                if with_prefix is not None:
                    # if a prefix is specified, then we don't want to copy the subfields directly to the root namespace
                    # but we prefix them with the name of the current field.
                    prefix = field_name
                self.gen_parser_from_dataclass(
                    parser, field_type(), delete_default, prefix)
                continue

            kwargs = get_kwargs_from_dc(dataclass_instance, k)

            if "default" in kwargs:
                if isinstance(kwargs["default"], str) and kwargs["default"].startswith(
                    "${"
                ):
                    if kwargs["help"] is None:
                        # this is a field with a name that will be added elsewhere
                        continue
                    else:
                        del kwargs["default"]
                if delete_default and "default" in kwargs:
                    del kwargs["default"]
            try:
                parser.add_argument(field_name, **kwargs)
                # parser.add_argument("--" + k.replace("_", "-"), **kwargs)
            except:
                pass

    def parse_args_and_arch(self, input_args: List[str] = None):
        """
        Args:
            parser (ArgumentParser): the parser
            input_args (List[str]): strings to parse, defaults to sys.argv
            parse_known (bool): only parse known arguments, similar to
                `ArgumentParser.parse_known_args`
            suppress_defaults (bool): parse while ignoring all default values
            modify_parser (Optional[Callable[[ArgumentParser], None]]):
                function to modify the parser, e.g., to set default values
        """
        # Parse a second time.
        args, extra = self.parser.parse_known_args(input_args)
        '''
        # Post-process args.
        if (
            hasattr(args, "batch_size_valid") and args.batch_size_valid is None
        ) or not hasattr(args, "batch_size_valid"):
            args.batch_size_valid = args.batch_size
        if hasattr(args, "max_tokens_valid") and args.max_tokens_valid is None:
            args.max_tokens_valid = args.max_tokens
        if getattr(args, "memory_efficient_fp16", False):
            args.fp16 = True
        if getattr(args, "memory_efficient_bf16", False):
            args.bf16 = True
            
        if getattr(args, "seed", None) is None:
            args.seed = 1  # default seed for training
            args.no_seed_provided = True
        else:
            args.no_seed_provided = False

        if getattr(args, "update_epoch_batch_itr", None) is None:
            if hasattr(args, "grouped_shuffling"):
                args.update_epoch_batch_itr = args.grouped_shuffling
            else:
                args.grouped_shuffling = False
                args.update_epoch_batch_itr = False
        '''
        
        logger.info('All arguments:')
        for k, v in sorted(vars(args).items()):
            logger.info(f'{k}:' + ''.join([' '] * max(50 - len(k), 1)) + f'{v}')
        logger.info(f'Unused arguments: {extra}')
        return args, extra
        
    def gen_dataclass_from_parser(self, dataclass_instance, with_prefix: Optional[str] = ''):
        def argparse_name(name: str):
            if name == "data" and (with_prefix is None or with_prefix == ""):
                # normally data is positional args, so we don't add the -- nor the prefix
                return name
            if name == "_name":
                # private member, skip
                return None
            if with_prefix is not None and with_prefix != "":
                # if a prefix is specified, construct the prefixed arg name
                name = with_prefix + "_" + name
            return name

        args = vars(self.args)
        for k in dataclass_instance._get_all_attributes():
            full_name = k
            field_name = argparse_name(dataclass_instance._get_name(k))
            field_type = dataclass_instance._get_type(k)
            if field_name is None:
                continue
            elif inspect.isclass(field_type) and issubclass(field_type, BaseConfig):
                prefix = None
                if with_prefix is not None:
                    prefix = field_name
                setattr(dataclass_instance, k,
                        self.gen_dataclass_from_parser(field_type(), prefix))
            elif field_name in args and args[field_name] is not None:
                setattr(dataclass_instance, k, args[field_name])
            # elif full_name in args and args[full_name]:
            #     setattr(dataclass_instance, k, args[full_name])

        return dataclass_instance

    def __getattribute__(self, attr):
        if attr == 'task':
            return super().__getattribute__(attr)
        try:
            return self.task.__getattribute__(attr)
        except:
            return super().__getattribute__(attr)
        