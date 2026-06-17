# datamineana 全局日志模块 使用文档

本文档覆盖模块所有对外暴露的接口、使用方法与特殊约定。

---

## 一、快速导入
支持两种使用方式，功能完全等价：
```python
# 方式1：导入全局单例实例，通过实例调用所有方法
from datamineana.logger import logger

# 方式2：直接导入快捷函数/装饰器（日常开发推荐）
from datamineana.logger import (
    debug, info, warning, error, meta,
    log_call, log_info, log_error,
    log_inputs, log_result, log_io, log_time,
    log_run_result,
    find_project_root,
    GlobalLogger
)
```

---

## 二、快捷日志函数（手动打点）
所有函数均写入 `normal` 日志文件（`meta` 函数除外，写入 `meta` 文件），支持附加结构化数据与单独控制控制台输出。

| 函数 | 功能说明 | 核心参数 | 使用示例 |
|------|----------|----------|----------|
| `debug(message, data=None, console=None)` | 调试级日志，用于开发排查细节 | `message`: 日志文本<br>`data`: 附加结构化字典<br>`console`: 是否打印控制台，默认跟随全局配置 | `debug("变量校验通过", data={"val": 123})` |
| `info(message, data=None, console=None)` | 信息级日志，记录正常流程节点 | 同上 | `info("数据加载完成", data={"rows": 1000})` |
| `warning(message, data=None, console=None)` | 警告级日志，记录非阻断异常 | 同上 | `warning("配置项缺失，使用默认值")` |
| `error(message, data=None, console=None)` | 错误级日志，记录业务失败 | 同上 | `error("文件读取失败", data={"path": "./data.csv"})` |
| `meta(message, data=None, console=False)` | 结构化元数据日志，写入 `meta` 文件 | 同上，`console` 默认 `False` | `meta("模型参数", data={"lr": 0.01, "epoch": 100})` |

> 补充：模块级无 `exception` 快捷函数，需记录完整异常堆栈时，使用全局实例调用：
> ```python
> try:
>     risky_operation()
> except Exception as e:
>     logger.exception("操作执行失败", e)
> ```

---

## 三、装饰器
所有装饰器均为关键字参数调用，标准写法 `@装饰器()`。

### 3.1 函数装饰器
#### `log_call()`
- **功能**：全流程埋点，记录函数开始、执行结束、运行耗时、异常堆栈，报错时自动记录入参摘要。最常用装饰器。
- **参数**：
  - `level: str = "INFO"`：正常日志级别
  - `console: bool = True`：是否打印控制台
  - `log_args_on_error: bool = True`：报错时是否记录函数入参
  - `reraise: bool = True`：捕获异常后是否向上重抛
- **示例**：
  ```python
  @log_call()
  def load_file(path: str):
      return open(path).read()
  ```

#### `log_info()`
- **功能**：轻量化，仅在函数执行前打印一条 INFO 日志，不记录结束与耗时。
- **参数**：
  - `message: str = None`：自定义日志内容，不填则自动使用函数名
  - `console: bool = True`
- **示例**：
  ```python
  @log_info(message="开始执行数据清洗")
  def clean_data(df):
      return df.dropna()
  ```

#### `log_error()`
- **功能**：仅捕获并记录异常，不记录函数启停，轻量化错误埋点。
- **参数**：
  - `console: bool = True`
  - `reraise: bool = True`
- **示例**：
  ```python
  @log_error()
  def risky_calc(x, y):
      return x / y
  ```

#### `log_inputs()`
- **功能**：记录函数所有入参摘要，写入 `meta` 文件，默认不打印控制台。自动对大对象做截断处理。
- **参数**：
  - `console: bool = False`
  - `max_string_length: int = 500`：字符串截断长度
- **示例**：
  ```python
  @log_inputs()
  def train_model(data, lr=0.01, epoch=100):
      pass
  ```

#### `log_result()`
- **功能**：记录函数返回值摘要，写入 `meta` 文件，默认不打印控制台。
- **参数**：同 `log_inputs()`
- **示例**：
  ```python
  @log_result()
  def get_config():
      return {"version": "v1.0", "env": "prod"}
  ```

#### `log_io()`
- **功能**：同时记录函数入参与返回值，等价于 `log_inputs()` + `log_result()`，写入 `meta` 文件。
- **参数**：同 `log_inputs()`
- **示例**：
  ```python
  @log_io()
  def add(a, b):
      return a + b
  ```

#### `log_time()`
- **功能**：仅统计函数执行耗时，不记录其他信息。
- **参数**：`console: bool = True`
- **示例**：
  ```python
  @log_time()
  def heavy_compute():
      import time
      time.sleep(1)
  ```

### 3.2 类装饰器
#### `log_run_result()`
- **功能**：装饰业务流水线类，自动监听 `run()` 方法执行；`run()` 正常结束后，自动调用 `get_result()` 并将结果写入 `meta` 文件。
- **特殊要求（必须满足）**：
  1. 被装饰类**必须实现 `run(self, *args, **kwargs)` 方法**，作为主执行入口，缺失会直接抛出 `AttributeError`
  2. 被装饰类**建议实现 `get_result(self)` 方法**，用于返回执行结果；缺失会打印 WARNING 警告，不中断程序
- **参数**：
  - `console: bool = False`：结果日志是否打印控制台
  - `max_string_length: int = 500`：结果字符串截断长度
  - `log_run_call: bool = True`：是否将 `run()` 的启停、耗时写入 `normal` 日志
- **示例**：
  ```python
  @log_run_result()
  class DataPipeline:
      def __init__(self):
          self.result = None

      def run(self):
          self.result = {"status": "success", "count": 200}

      def get_result(self):
          return self.result
  ```

---

## 四、GlobalLogger 全局日志类
线程安全的单例日志类，全局唯一实例。

### 获取实例
```python
# 方式1：使用默认配置（推荐，直接用模块导出的 logger 即可）
from datamineana.logger import logger

# 方式2：自定义配置初始化（仅需在程序入口调用一次）
from datamineana.logger import GlobalLogger
logger = GlobalLogger.get_instance(
    log_dir="./custom_log",    # 日志存储目录
    clear_on_start=False,      # 关闭启动自动清空旧日志
    default_console=False      # 关闭默认控制台输出
)
```
> 注意：单例模式下，首次初始化后再次调用 `get_instance()` 不会修改配置。

### 实例方法
除上述快捷函数对应的方法外，还支持以下能力：
| 方法 | 功能说明 |
|------|----------|
| `logger.exception(message, exc, data=None, console=None)` | 记录完整异常堆栈，需传入捕获到的异常对象 |
| `logger.get_normal_cache()` | 获取内存中 `normal` 日志缓存列表 |
| `logger.get_meta_cache()` | 获取内存中 `meta` 日志缓存列表 |
| `logger.clear_memory_cache()` | 清空内存日志缓存，不影响本地文件 |

---

## 五、工具函数
### `find_project_root()`
- **功能**：从指定路径开始，逐级向上查找项目根目录，通过项目标识文件识别。
- **参数**：
  - `start_path: str | Path = None`：起始路径，不填则从当前文件所在目录开始
  - `markers: List[str] = None`：项目标识文件列表，默认包含 `pyproject.toml`、`setup.py`、`.git`、`requirements.txt` 等
- **返回值**：`Path` 对象，未找到标识文件时返回当前工作目录
- **示例**：
  ```python
  root = find_project_root()
  print(root / "config.yaml")
  ```

---

## 六、导出常量
| 常量 | 说明 |
|------|------|
| `DEFAULT_PATH` | 默认日志存储目录路径，即 `项目根目录/cache/log` |
| `NORMAL_LOG_FILE` | normal 日志文件完整路径 |
| `META_LOG_FILE` | meta 日志文件完整路径 |

---

## 七、特殊约定与注意事项
1. **日志格式**：所有日志文件均为 JSON Lines 格式，每行一条 JSON 记录，包含时间、级别、事件、消息、结构化数据。
2. **时区统一**：所有日志时间均为 UTC 时区 ISO 格式，避免时区混乱。
3. **大对象自动摘要**：
   - `TabularData`：自动调用 `get_info()` 记录结构摘要，不打印全量数据
   - `DataFrame/Series`：仅记录 shape、列名、字段类型
   - 长字符串、长列表：自动截断并标注总长度
   - `dataclass`、实现 `to_dict()` 的类：自动转字典序列化
4. **单例特性**：全局只有一个日志实例，不会重复创建 handler，避免日志重复输出。
5. **启动清理**：默认首次初始化时自动清空旧日志目录，如需保留历史日志请手动设置 `clear_on_start=False`。