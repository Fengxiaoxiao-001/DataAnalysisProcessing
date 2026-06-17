# logger.py
from __future__ import annotations

import functools
import inspect
import json
import logging
import shutil
import sys
import threading
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# 项目根目录查找
# =============================================================================

def find_project_root(
        start_path: Optional[Union[str, Path]] = None,
        markers: Optional[List[str]] = None,
) -> Path:
    """
    查找项目根目录。

    默认会从当前文件所在目录或传入目录开始，逐级向上寻找项目标识文件。

    常见项目标识包括：
    - pyproject.toml
    - setup.py
    - setup.cfg
    - requirements.txt
    - .git

    如果没有找到，则返回当前工作目录。

    Parameters
    ----------
    start_path:
        开始查找的路径。
        如果为 None，则从当前文件所在目录开始查找。

    markers:
        项目标识文件或目录名称列表。

    Returns
    -------
    Path
        项目根目录路径。
    """
    if markers is None:
        markers = [
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
            ".git",
        ]

    if start_path is None:
        current = Path(__file__).resolve().parent
    else:
        current = Path(start_path).resolve()

    if current.is_file():
        current = current.parent

    for parent in [current, *current.parents]:
        for marker in markers:
            if (parent / marker).exists():
                return parent

    return Path.cwd().resolve()


# 默认日志缓存目录
DEFAULT_PATH = find_project_root() / "cache" / "log"

# normal 文件用于记录普通日志、info、error、函数运行状态等
NORMAL_LOG_FILE = DEFAULT_PATH / "normal"

# meta 文件用于记录函数输入、函数输出、特殊结果等结构化信息
META_LOG_FILE = DEFAULT_PATH / "meta"


# =============================================================================
# 工具函数
# =============================================================================

def _utc_now_iso() -> str:
    """
    获取当前 UTC 时间字符串。

    使用 datetime.now(UTC)，避免 datetime.utcnow() 在新版 Python 中的弃用警告。

    Returns
    -------
    str
        ISO 格式 UTC 时间字符串。
    """
    return datetime.now(UTC).isoformat()


def _safe_json_dumps(obj: Any) -> str:
    """
    安全 JSON 序列化。

    避免普通 json.dumps 在遇到 DataFrame、Path、datetime、自定义类时直接报错。

    Parameters
    ----------
    obj:
        任意 Python 对象。

    Returns
    -------
    str
        JSON 字符串。
    """
    return json.dumps(
        obj,
        ensure_ascii=False,
        default=_json_default,
    )


def _json_default(obj: Any) -> Any:
    """
    json.dumps 的 default 回调。

    用于处理 JSON 默认不支持的对象类型。

    Parameters
    ----------
    obj:
        任意 Python 对象。

    Returns
    -------
    Any
        可 JSON 序列化的对象。
    """
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, datetime):
        return obj.isoformat()

    if is_dataclass(obj):
        try:
            return asdict(obj)
        except Exception:
            return repr(obj)

    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return obj.to_dict()
        except Exception:
            return repr(obj)

    return repr(obj)


def _is_tabular_data_like(obj: Any) -> bool:
    """
    判断对象是否像 TabularData。

    这里不直接 import TabularData，是为了避免 logger 模块和数据模块互相循环导入。

    满足以下条件之一即可认为是 TabularData-like：
    1. 类名为 TabularData
    2. 对象有 get_info() 方法，并且模块名或类名中包含 tabular

    Parameters
    ----------
    obj:
        任意对象。

    Returns
    -------
    bool
        是否疑似 TabularData 对象。
    """

    cls = obj.__class__
    cls_name = cls.__name__.lower()
    module_name = getattr(cls, "__module__", "").lower()

    if cls_name == "tabulardata":
        return True

    if hasattr(obj, "get_info") and callable(obj.get_info):
        if "tabular" in cls_name or "tabular" in module_name:
            return True

    return False


def _summarize_value(value: Any, max_string_length: int = 500) -> Any:
    """
    对日志中的值进行安全摘要。

    目的：
    - 避免大型 DataFrame、大型列表、大型对象直接写入日志
    - 对 TabularData 自动调用 get_info()
    - 保留基础类型的可读性

    Parameters
    ----------
    value:
        任意对象。

    max_string_length:
        字符串最大保留长度。

    Returns
    -------
    Any
        适合写入日志的摘要数据。
    """
    # 针对 TabularData 特殊处理
    if _is_tabular_data_like(value):
        try:
            return {
                "__type__": value.__class__.__name__,
                "info": value.get_info(),
            }
        except Exception as exc:
            return {
                "__type__": value.__class__.__name__,
                "get_info_error": repr(exc),
                "repr": repr(value),
            }

    # None、布尔值、数字可以直接记录
    if value is None or isinstance(value, (bool, int, float)):
        return value

    # 字符串过长时截断
    if isinstance(value, str):
        if len(value) > max_string_length:
            return value[:max_string_length] + f"... <truncated length={len(value)}>"
        return value

    # Path 转字符串
    if isinstance(value, Path):
        return str(value)

    # datetime 转 ISO
    if isinstance(value, datetime):
        return value.isoformat()

    # 字典递归摘要
    if isinstance(value, dict):
        return {
            str(k): _summarize_value(v, max_string_length=max_string_length)
            for k, v in value.items()
        }

    # list / tuple / set 控制最大记录数量
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        max_items = 20
        summarized = [
            _summarize_value(v, max_string_length=max_string_length)
            for v in seq[:max_items]
        ]

        if len(seq) > max_items:
            summarized.append(f"... <truncated items={len(seq) - max_items}>")

        return {
            "__type__": value.__class__.__name__,
            "length": len(seq),
            "items": summarized,
        }

    # pandas DataFrame / Series 的轻量处理，不强制依赖 pandas
    cls_name = value.__class__.__name__

    if cls_name == "DataFrame":
        try:
            return {
                "__type__": "DataFrame",
                "shape": value.shape,
                "columns": list(value.columns),
                "dtypes": {
                    str(k): str(v)
                    for k, v in value.dtypes.to_dict().items()
                },
            }
        except Exception:
            return repr(value)

    if cls_name == "Series":
        try:
            return {
                "__type__": "Series",
                "name": value.name,
                "length": len(value),
                "dtype": str(value.dtype),
            }
        except Exception:
            return repr(value)

    # dataclass 转 dict
    if is_dataclass(value):
        try:
            return asdict(value)
        except Exception:
            return repr(value)

    # 有 to_dict 方法的对象
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            result = value.to_dict()
            return _summarize_value(result, max_string_length=max_string_length)
        except Exception:
            return repr(value)

    # 兜底
    text = repr(value)
    if len(text) > max_string_length:
        return text[:max_string_length] + f"... <truncated length={len(text)}>"
    return text


def _get_callable_location(func: Callable[..., Any]) -> Dict[str, Any]:
    """
    获取函数位置信息。

    Parameters
    ----------
    func:
        被装饰的函数。

    Returns
    -------
    Dict[str, Any]
        函数所在模块、函数名、限定名、文件路径、起始行号。
    """
    try:
        source_file = inspect.getsourcefile(func)
    except Exception:
        source_file = None

    try:
        line_no = inspect.getsourcelines(func)[1]
    except Exception:
        line_no = None

    return {
        "module": getattr(func, "__module__", None),
        "name": getattr(func, "__name__", None),
        "qualname": getattr(func, "__qualname__", None),
        "file": source_file,
        "line": line_no,
    }


def _bind_arguments(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Dict[str, Any]:
    """
    将函数 args / kwargs 绑定为参数名到参数值的映射。

    如果绑定失败，则退化为 args / kwargs 结构。

    Parameters
    ----------
    func:
        目标函数。

    args:
        位置参数。

    kwargs:
        关键字参数。

    Returns
    -------
    Dict[str, Any]
        参数名和值的映射。
    """
    try:
        signature = inspect.signature(func)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except Exception:
        return {
            "args": args,
            "kwargs": kwargs,
        }


# =============================================================================
# 全局日志单例类
# =============================================================================

class GlobalLogger:
    """
    全局单例日志类。

    设计目标
    --------
    1. 全局只有一个日志实例，避免重复创建 handler 导致日志重复输出。
    2. 每次程序运行时自动清理旧日志缓存。
    3. normal 文件记录普通日志和异常。
    4. meta 文件记录输入、输出、结果等结构化信息。
    5. 装饰器支持记录函数运行状态、输入、输出、异常和耗时。
    6. 对 TabularData 类型输入自动调用 get_info()，避免记录过大的表格数据。
    """

    _instance: Optional["GlobalLogger"] = None
    _instance_lock = threading.Lock()

    def __new__(
            cls,
            log_dir: Union[str, Path] = DEFAULT_PATH,
            clear_on_start: bool = True,
            default_console: bool = True,
    ) -> "GlobalLogger":
        """
        创建或获取单例实例。

        Parameters
        ----------
        log_dir:
            日志目录。

        clear_on_start:
            是否在首次初始化时清理旧日志。

        default_console:
            默认是否输出到控制台。

        Returns
        -------
        GlobalLogger
            全局单例对象。
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False

        return cls._instance

    def __init__(
            self,
            log_dir: Union[str, Path] = DEFAULT_PATH,
            clear_on_start: bool = True,
            default_console: bool = True,
    ) -> None:
        """
        初始化日志器。

        注意：
        单例模式下 __init__ 可能被多次调用，因此内部用 _initialized 防止重复初始化。
        """
        if getattr(self, "_initialized", False):
            return

        self.log_dir = Path(log_dir).resolve()
        self.normal_log_file = self.log_dir / "normal"
        self.meta_log_file = self.log_dir / "meta"
        self.default_console = default_console

        # 内存日志缓存
        # normal_records 用于保存普通日志记录
        # meta_records 用于保存结构化元信息记录
        self.normal_records: List[Dict[str, Any]] = []
        self.meta_records: List[Dict[str, Any]] = []

        self._write_lock = threading.RLock()

        self._prepare_log_dir(clear_on_start=clear_on_start)
        self._normal_logger = self._create_file_logger(
            name="datamineana.normal",
            file_path=self.normal_log_file,
        )
        self._meta_logger = self._create_file_logger(
            name="datamineana.meta",
            file_path=self.meta_log_file,
        )

        self._initialized = True

    @classmethod
    def get_instance(
            cls,
            log_dir: Union[str, Path] = DEFAULT_PATH,
            clear_on_start: bool = True,
            default_console: bool = True,
    ) -> "GlobalLogger":
        """
        获取全局日志单例。

        Parameters
        ----------
        log_dir:
            日志目录。

        clear_on_start:
            首次初始化时是否清理旧缓存。

        default_console:
            默认是否输出到控制台。

        Returns
        -------
        GlobalLogger
            全局日志器。
        """
        return cls(
            log_dir=log_dir,
            clear_on_start=clear_on_start,
            default_console=default_console,
        )

    def _prepare_log_dir(self, clear_on_start: bool) -> None:
        """
        准备日志目录。

        Parameters
        ----------
        clear_on_start:
            是否清理旧日志目录。
        """
        if clear_on_start and self.log_dir.exists():
            shutil.rmtree(self.log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 创建 normal 和 meta 文件
        self.normal_log_file.touch(exist_ok=True)
        self.meta_log_file.touch(exist_ok=True)

    def _create_file_logger(self, name: str, file_path: Path) -> logging.Logger:
        """
        创建仅写文件的 logger。

        这里不直接添加 StreamHandler。
        原因是不同装饰器需要单独控制是否输出控制台。
        如果给 logger 添加全局 StreamHandler，就无法做到每条日志单独控制 console。

        Parameters
        ----------
        name:
            logger 名称。

        file_path:
            日志文件路径。

        Returns
        -------
        logging.Logger
            Python logger 对象。
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # 防止重复添加 handler
        logger.handlers.clear()

        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(message)s")
        )

        logger.addHandler(file_handler)

        return logger

    # -------------------------------------------------------------------------
    # 基础日志写入
    # -------------------------------------------------------------------------

    def _emit(
            self,
            *,
            target: str,
            level: str,
            event: str,
            message: str,
            data: Optional[Dict[str, Any]] = None,
            console: Optional[bool] = None,
    ) -> None:
        """
        写入一条日志。

        Parameters
        ----------
        target:
            日志目标。
            可选：
            - "normal"
            - "meta"

        level:
            日志级别。
            例如：
            - DEBUG
            - INFO
            - WARNING
            - ERROR
            - EXCEPTION
            - META

        event:
            事件名称。

        message:
            日志消息。

        data:
            附加结构化数据。

        console:
            是否输出到控制台。
            如果为 None，则使用 self.default_console。
        """
        if console is None:
            console = self.default_console

        record = {
            "time": _utc_now_iso(),
            "target": target,
            "level": level.upper(),
            "event": event,
            "message": message,
            "data": data or {},
        }

        line = _safe_json_dumps(record)

        with self._write_lock:
            if target == "meta":
                self.meta_records.append(record)
                self._meta_logger.info(line)
            else:
                self.normal_records.append(record)
                self._normal_logger.info(line)

        if console:
            self._print_console(record)

    def _print_console(self, record: Dict[str, Any]) -> None:
        """
        控制台打印日志。

        Parameters
        ----------
        record:
            日志记录。
        """
        level = record.get("level", "INFO")
        event = record.get("event", "")
        message = record.get("message", "")
        time_text = record.get("time", "")

        text = f"[{time_text}] [{level}] [{event}] {message}"

        if level in {"ERROR", "EXCEPTION"}:
            print(text, file=sys.stderr, flush=True)
        else:
            print(text, file=sys.stdout, flush=True)

    def debug(self, message: str, data: Optional[Dict[str, Any]] = None, console: Optional[bool] = None) -> None:
        """记录 DEBUG 日志。"""
        self._emit(
            target="normal",
            level="DEBUG",
            event="debug",
            message=message,
            data=data,
            console=console,
        )

    def info(self, message: str, data: Optional[Dict[str, Any]] = None, console: Optional[bool] = None) -> None:
        """记录 INFO 日志。"""
        self._emit(
            target="normal",
            level="INFO",
            event="info",
            message=message,
            data=data,
            console=console,
        )

    def warning(self, message: str, data: Optional[Dict[str, Any]] = None, console: Optional[bool] = None) -> None:
        """记录 WARNING 日志。"""
        self._emit(
            target="normal",
            level="WARNING",
            event="warning",
            message=message,
            data=data,
            console=console,
        )

    def error(self, message: str, data: Optional[Dict[str, Any]] = None, console: Optional[bool] = None) -> None:
        """记录 ERROR 日志。"""
        self._emit(
            target="normal",
            level="ERROR",
            event="error",
            message=message,
            data=data,
            console=console,
        )

    def meta(self, message: str, data: Optional[Dict[str, Any]] = None, console: bool = False) -> None:
        """
        记录 META 日志。

        meta 默认不输出到控制台。
        """
        self._emit(
            target="meta",
            level="META",
            event="meta",
            message=message,
            data=data,
            console=console,
        )

    def exception(
            self,
            message: str,
            exc: BaseException,
            data: Optional[Dict[str, Any]] = None,
            console: Optional[bool] = None,
    ) -> None:
        """
        记录异常日志。

        Parameters
        ----------
        message:
            异常说明。

        exc:
            捕获到的异常对象。

        data:
            附加数据。

        console:
            是否输出控制台。
        """
        exception_data = {
            "exception_type": exc.__class__.__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }

        if data:
            exception_data.update(data)

        self._emit(
            target="normal",
            level="EXCEPTION",
            event="exception",
            message=message,
            data=exception_data,
            console=console,
        )

    # -------------------------------------------------------------------------
    # 缓存读取与清理
    # -------------------------------------------------------------------------

    def get_normal_cache(self) -> List[Dict[str, Any]]:
        """
        获取内存中的 normal 日志缓存。

        Returns
        -------
        List[Dict[str, Any]]
            normal 日志记录列表。
        """
        return list(self.normal_records)

    def get_meta_cache(self) -> List[Dict[str, Any]]:
        """
        获取内存中的 meta 日志缓存。

        Returns
        -------
        List[Dict[str, Any]]
            meta 日志记录列表。
        """
        return list(self.meta_records)

    def clear_memory_cache(self) -> None:
        """
        清理内存日志缓存。

        注意：
        该方法只清理内存中的缓存，不清理文件。
        """
        with self._write_lock:
            self.normal_records.clear()
            self.meta_records.clear()

    # -------------------------------------------------------------------------
    # 装饰器：普通函数运行日志
    # -------------------------------------------------------------------------

    def log_call(
            self,
            *,
            level: str = "INFO",
            console: Optional[bool] = True,
            log_args_on_error: bool = True,
            reraise: bool = True,
    ) -> Callable[[F], F]:
        """
        装饰器：记录函数开始、结束和异常。

        这是最常用的装饰器。

        功能：
        - 函数开始时记录：正在运行哪个模块的哪个函数
        - 函数结束时记录：运行完成以及耗时
        - 函数异常时记录：异常类型、异常信息、traceback
        - 默认控制台显示，因此可以实时看到运行到哪个函数

        Parameters
        ----------
        level:
            正常日志级别。

        console:
            是否输出到控制台。

        log_args_on_error:
            报错时是否记录函数输入参数摘要。

        reraise:
            捕获异常后是否重新抛出。
            一般建议 True，否则错误会被吞掉。

        Returns
        -------
        Callable
            装饰器。
        """

        def decorator(func: F) -> F:
            location = _get_callable_location(func)

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()

                self._emit(
                    target="normal",
                    level=level,
                    event="function_start",
                    message=f"Start running {location.get('module')}.{location.get('qualname')}",
                    data={
                        "function": location,
                    },
                    console=console,
                )

                try:
                    result = func(*args, **kwargs)
                    elapsed = time.perf_counter() - start

                    self._emit(
                        target="normal",
                        level=level,
                        event="function_end",
                        message=f"Finished {location.get('module')}.{location.get('qualname')} in {elapsed:.6f}s",
                        data={
                            "function": location,
                            "elapsed_seconds": elapsed,
                        },
                        console=console,
                    )

                    return result

                except Exception as exc:
                    elapsed = time.perf_counter() - start

                    extra_data: Dict[str, Any] = {
                        "function": location,
                        "elapsed_seconds": elapsed,
                    }

                    if log_args_on_error:
                        bound_args = _bind_arguments(func, args, kwargs)
                        extra_data["arguments"] = _summarize_value(bound_args)

                    self.exception(
                        message=f"Error in {location.get('module')}.{location.get('qualname')}",
                        exc=exc,
                        data=extra_data,
                        console=console,
                    )

                    if reraise:
                        raise

                    return None

            return cast(F, wrapper)

        return decorator

    def log_info(
            self,
            *,
            message: Optional[str] = None,
            console: Optional[bool] = True,
    ) -> Callable[[F], F]:
        """
        装饰器：以 INFO 级别记录函数运行。

        Parameters
        ----------
        message:
            自定义日志消息。

        console:
            是否输出控制台。

        Returns
        -------
        Callable
            装饰器。
        """

        def decorator(func: F) -> F:
            location = _get_callable_location(func)

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                msg = message or f"Running {location.get('module')}.{location.get('qualname')}"
                self.info(
                    msg,
                    data={
                        "function": location,
                    },
                    console=console,
                )
                return func(*args, **kwargs)

            return cast(F, wrapper)

        return decorator

    def log_error(
            self,
            *,
            console: Optional[bool] = True,
            reraise: bool = True,
    ) -> Callable[[F], F]:
        """
        装饰器：专门记录函数异常。

        与 log_call 不同：
        - log_error 不记录函数开始和结束
        - 只在报错时记录异常

        Parameters
        ----------
        console:
            是否输出控制台。

        reraise:
            是否重新抛出异常。

        Returns
        -------
        Callable
            装饰器。
        """

        def decorator(func: F) -> F:
            location = _get_callable_location(func)

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    bound_args = _bind_arguments(func, args, kwargs)

                    self.exception(
                        message=f"Error in {location.get('module')}.{location.get('qualname')}",
                        exc=exc,
                        data={
                            "function": location,
                            "arguments": _summarize_value(bound_args),
                        },
                        console=console,
                    )

                    if reraise:
                        raise

                    return None

            return cast(F, wrapper)

        return decorator

    # -------------------------------------------------------------------------
    # 装饰器：meta 信息记录
    # -------------------------------------------------------------------------

    def log_inputs(
            self,
            *,
            console: bool = False,
            max_string_length: int = 500,
    ) -> Callable[[F], F]:
        """
        装饰器：记录函数输入参数到 meta 文件。

        meta 默认不输出控制台。

        特殊处理：
        - 如果输入参数中含有 TabularData，会自动调用 TabularData.get_info()
        - 避免把大型表格完整写入日志

        Parameters
        ----------
        console:
            是否输出控制台。

        max_string_length:
            字符串最大记录长度。

        Returns
        -------
        Callable
            装饰器。
        """

        def decorator(func: F) -> F:
            location = _get_callable_location(func)

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                bound_args = _bind_arguments(func, args, kwargs)
                summarized_args = {
                    key: _summarize_value(value, max_string_length=max_string_length)
                    for key, value in bound_args.items()
                }

                self._emit(
                    target="meta",
                    level="META",
                    event="function_inputs",
                    message=f"Inputs of {location.get('module')}.{location.get('qualname')}",
                    data={
                        "function": location,
                        "inputs": summarized_args,
                    },
                    console=console,
                )

                return func(*args, **kwargs)

            return cast(F, wrapper)

        return decorator

    def log_result(
            self,
            *,
            console: bool = False,
            max_string_length: int = 500,
    ) -> Callable[[F], F]:
        """
        装饰器：记录函数返回值到 meta 文件。

        meta 默认不输出控制台。

        Parameters
        ----------
        console:
            是否输出控制台。

        max_string_length:
            字符串最大记录长度。

        Returns
        -------
        Callable
            装饰器。
        """

        def decorator(func: F) -> F:
            location = _get_callable_location(func)

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = func(*args, **kwargs)

                self._emit(
                    target="meta",
                    level="META",
                    event="function_result",
                    message=f"Result of {location.get('module')}.{location.get('qualname')}",
                    data={
                        "function": location,
                        "result": _summarize_value(result, max_string_length=max_string_length),
                    },
                    console=console,
                )

                return result

            return cast(F, wrapper)

        return decorator

    def log_io(
            self,
            *,
            console: bool = False,
            max_string_length: int = 500,
    ) -> Callable[[F], F]:
        """
        装饰器：同时记录函数输入和输出到 meta 文件。

        Parameters
        ----------
        console:
            是否输出控制台。

        max_string_length:
            字符串最大记录长度。

        Returns
        -------
        Callable
            装饰器。
        """

        def decorator(func: F) -> F:
            location = _get_callable_location(func)

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                bound_args = _bind_arguments(func, args, kwargs)
                summarized_args = {
                    key: _summarize_value(value, max_string_length=max_string_length)
                    for key, value in bound_args.items()
                }

                self._emit(
                    target="meta",
                    level="META",
                    event="function_inputs",
                    message=f"Inputs of {location.get('module')}.{location.get('qualname')}",
                    data={
                        "function": location,
                        "inputs": summarized_args,
                    },
                    console=console,
                )

                result = func(*args, **kwargs)

                self._emit(
                    target="meta",
                    level="META",
                    event="function_result",
                    message=f"Result of {location.get('module')}.{location.get('qualname')}",
                    data={
                        "function": location,
                        "result": _summarize_value(result, max_string_length=max_string_length),
                    },
                    console=console,
                )

                return result

            return cast(F, wrapper)

        return decorator

    def log_time(
            self,
            *,
            console: Optional[bool] = True,
    ) -> Callable[[F], F]:
        """
        装饰器：只记录函数耗时。

        Parameters
        ----------
        console:
            是否输出控制台。

        Returns
        -------
        Callable
            装饰器。
        """

        def decorator(func: F) -> F:
            location = _get_callable_location(func)

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    elapsed = time.perf_counter() - start
                    self._emit(
                        target="normal",
                        level="INFO",
                        event="function_time",
                        message=f"{location.get('module')}.{location.get('qualname')} cost {elapsed:.6f}s",
                        data={
                            "function": location,
                            "elapsed_seconds": elapsed,
                        },
                        console=console,
                    )

            return cast(F, wrapper)

        return decorator

    # -------------------------------------------------------------------------
    # 装饰器：类对象 run() 结果记录
    # -------------------------------------------------------------------------

    def log_run_result(
            self,
            *,
            console: bool = False,
            max_string_length: int = 500,
            log_run_call: bool = True,
    ) -> Callable[[type], type]:
        """
        类装饰器：包装类的 run 方法。

        功能：
        - 当对象的 run() 方法执行完成后
        - 自动调用对象的 get_result() 方法
        - 将 get_result() 的返回值写入 meta 文件

        使用要求：
        被装饰的类应该具有：
        - run(self, *args, **kwargs)
        - get_result(self)

        示例
        ----
        @logger.log_run_result()
        class MyPipeline:
            def run(self):
                ...

            def get_result(self):
                return {"status": "ok"}

        Parameters
        ----------
        console:
            meta 信息是否输出控制台。
            默认 False。

        max_string_length:
            字符串最大记录长度。

        log_run_call:
            是否同时把 run 方法的开始和结束写入 normal。
            默认 True，方便实时显示运行到哪里。

        Returns
        -------
        Callable[[type], type]
            类装饰器。
        """

        def class_decorator(cls: type) -> type:
            original_run = getattr(cls, "run", None)

            if original_run is None or not callable(original_run):
                raise AttributeError(
                    f"Class {cls.__name__} must define a callable run method."
                )

            @functools.wraps(original_run)
            def wrapped_run(instance: Any, *args: Any, **kwargs: Any) -> Any:
                cls_location = {
                    "module": getattr(cls, "__module__", None),
                    "class": getattr(cls, "__name__", None),
                    "method": "run",
                    "qualname": f"{getattr(cls, '__name__', None)}.run",
                }

                if log_run_call:
                    self._emit(
                        target="normal",
                        level="INFO",
                        event="object_run_start",
                        message=f"Start running {cls_location.get('module')}.{cls_location.get('qualname')}",
                        data={
                            "class": cls_location,
                        },
                        console=True,
                    )

                start = time.perf_counter()

                try:
                    run_return = original_run(instance, *args, **kwargs)
                except Exception as exc:
                    elapsed = time.perf_counter() - start
                    self.exception(
                        message=f"Error in {cls_location.get('module')}.{cls_location.get('qualname')}",
                        exc=exc,
                        data={
                            "class": cls_location,
                            "elapsed_seconds": elapsed,
                        },
                        console=True,
                    )
                    raise

                elapsed = time.perf_counter() - start

                if log_run_call:
                    self._emit(
                        target="normal",
                        level="INFO",
                        event="object_run_end",
                        message=f"Finished {cls_location.get('module')}.{cls_location.get('qualname')} in {elapsed:.6f}s",
                        data={
                            "class": cls_location,
                            "elapsed_seconds": elapsed,
                        },
                        console=True,
                    )

                if not hasattr(instance, "get_result") or not callable(instance.get_result):
                    self.warning(
                        message=f"{cls.__name__} has no callable get_result method.",
                        data={
                            "class": cls_location,
                        },
                        console=True,
                    )
                    return run_return

                try:
                    result = instance.get_result()
                    self._emit(
                        target="meta",
                        level="META",
                        event="object_run_result",
                        message=f"Result of {cls_location.get('module')}.{cls.__name__}.get_result() after run()",
                        data={
                            "class": cls_location,
                            "result": _summarize_value(
                                result,
                                max_string_length=max_string_length,
                            ),
                        },
                        console=console,
                    )
                except Exception as exc:
                    self.exception(
                        message=f"Error when calling {cls.__name__}.get_result()",
                        exc=exc,
                        data={
                            "class": cls_location,
                        },
                        console=True,
                    )
                    raise

                return run_return

            setattr(cls, "run", wrapped_run)
            return cls

        return class_decorator


# =============================================================================
# 全局默认实例
# =============================================================================

logger = GlobalLogger.get_instance(
    log_dir=DEFAULT_PATH,
    clear_on_start=True,
    default_console=True,
)


# =============================================================================
# 模块级快捷函数和快捷装饰器
# =============================================================================
# 这样外部既可以：
# from datamineana.logger import logger
#
# 也可以：
# from datamineana.logger import log_call, log_inputs


def debug(message: str, data: Optional[Dict[str, Any]] = None, console: Optional[bool] = None) -> None:
    """快捷 DEBUG 日志。"""
    logger.debug(message, data=data, console=console)


def info(message: str, data: Optional[Dict[str, Any]] = None, console: Optional[bool] = None) -> None:
    """快捷 INFO 日志。"""
    logger.info(message, data=data, console=console)


def warning(message: str, data: Optional[Dict[str, Any]] = None, console: Optional[bool] = None) -> None:
    """快捷 WARNING 日志。"""
    logger.warning(message, data=data, console=console)


def error(message: str, data: Optional[Dict[str, Any]] = None, console: Optional[bool] = None) -> None:
    """快捷 ERROR 日志。"""
    logger.error(message, data=data, console=console)


def meta(message: str, data: Optional[Dict[str, Any]] = None, console: bool = False) -> None:
    """快捷 META 日志。"""
    logger.meta(message, data=data, console=console)


def log_call(
        *,
        level: str = "INFO",
        console: Optional[bool] = True,
        log_args_on_error: bool = True,
        reraise: bool = True,
) -> Callable[[F], F]:
    """快捷装饰器：记录函数开始、结束和异常。"""
    return logger.log_call(
        level=level,
        console=console,
        log_args_on_error=log_args_on_error,
        reraise=reraise,
    )


def log_info(
        *,
        message: Optional[str] = None,
        console: Optional[bool] = True,
) -> Callable[[F], F]:
    """快捷装饰器：记录 INFO 日志。"""
    return logger.log_info(
        message=message,
        console=console,
    )


def log_error(
        *,
        console: Optional[bool] = True,
        reraise: bool = True,
) -> Callable[[F], F]:
    """快捷装饰器：记录异常。"""
    return logger.log_error(
        console=console,
        reraise=reraise,
    )


def log_inputs(
        *,
        console: bool = False,
        max_string_length: int = 500,
) -> Callable[[F], F]:
    """快捷装饰器：记录函数输入到 meta 文件。"""
    return logger.log_inputs(
        console=console,
        max_string_length=max_string_length,
    )


def log_result(
        *,
        console: bool = False,
        max_string_length: int = 500,
) -> Callable[[F], F]:
    """快捷装饰器：记录函数返回值到 meta 文件。"""
    return logger.log_result(
        console=console,
        max_string_length=max_string_length,
    )


def log_io(
        *,
        console: bool = False,
        max_string_length: int = 500,
) -> Callable[[F], F]:
    """快捷装饰器：同时记录函数输入和输出到 meta 文件。"""
    return logger.log_io(
        console=console,
        max_string_length=max_string_length,
    )


def log_time(
        *,
        console: Optional[bool] = True,
) -> Callable[[F], F]:
    """快捷装饰器：记录函数耗时。"""
    return logger.log_time(
        console=console,
    )


def log_run_result(
        *,
        console: bool = False,
        max_string_length: int = 500,
        log_run_call: bool = True,
) -> Callable[[type], type]:
    """快捷类装饰器：run 执行后调用 get_result 并记录结果。"""
    return logger.log_run_result(
        console=console,
        max_string_length=max_string_length,
        log_run_call=log_run_call,
    )


__all__ = [
    "DEFAULT_PATH",
    "NORMAL_LOG_FILE",
    "META_LOG_FILE",
    "GlobalLogger",
    "logger",
    "find_project_root",
    "debug",
    "info",
    "warning",
    "error",
    "meta",
    "log_call",
    "log_info",
    "log_error",
    "log_inputs",
    "log_result",
    "log_io",
    "log_time",
    "log_run_result",
]
