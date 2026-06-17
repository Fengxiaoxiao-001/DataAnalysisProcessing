# DataMineAna 数据模块说明文档

## 一、模块概述

本模块是 DataMineAna
项目的核心数据基础设施，封装了表格数据从加载、封装、处理到存储的全链路能力，支持多格式文件读取、大数据集懒加载分块、标准化数据容器、安全序列化导出，同时提供存储、渲染等扩展能力，统一项目内数据流转规范。

## 二、真实目录结构

```
datamineana/
├── __init__.py               # 包入口
├── logger.py                 # 日志模块
├── dataobject/               # 核心数据对象层
│   ├── __init__.py
│   ├── DataSelf.py           # 核心表格数据容器（TabularData 等核心类）
│   ├── Dtypes.py             # 基础类型定义（FileType 枚举、类型别名）
│   └── special_data.py       # 特殊数据与处理后数据封装
├── dataloader/               # 数据加载层
│   ├── __init__.py
│   ├── base.py               # 加载器抽象基类
│   ├── csv_loader.py         # CSV 文件加载器
│   ├── excel_loader.py       # Excel 文件加载器
│   └── universal_loader.py   # 通用调度加载器（自动识别文件格式）
├── savers/                   # 数据存储层
│   ├── __init__.py
│   ├── base.py               # 存储抽象基类
│   ├── file_saver.py         # 文件存储实现
│   └── database_saver.py     # 数据库存储实现
├── html/                     # 数据渲染层
│   ├── __init__.py
│   └── renderer.py           # HTML 格式数据渲染
├── utils/                    # 通用工具层
│   ├── __init__.py
│   └── path_function.py      # 路径处理工具
└── test/                     # 测试辅助模块
    ├── __init__.py
    └── create_csv.py         # 测试数据生成工具
```

## 三、核心模块说明

### 1. 数据对象层（dataobject）

全项目数据流转的标准载体，是模块的核心。

- **DataSelf.py**：实现 `TabularData` 通用表格容器，支持 DataFrame、分块迭代器等多类数据源，提供列/行筛选、数据预览、结构视图、JSON/字典安全导出、懒加载优化等完整能力。
- **Dtypes.py**：定义 `FileType` 文件类型枚举、`PathLike` 等类型别名，统一全项目类型规范。
- **special_data.py**：定义 `SpecialData` 通用业务数据容器、`ProcessedTabularData` 处理后表格容器，用于封装模型输出、统计结果、流程元数据等非原生表格数据。

### 2. 数据加载层（dataloader）

多格式文件统一读取入口，所有加载器均遵循统一基类规范，输出标准 `TabularData` 对象。

- `universal_loader`：自动根据文件后缀调度对应加载器，一键读取 CSV/Excel 等格式
- `csv_loader` / `excel_loader`：单格式专用加载器，支持懒加载分块、列筛选、类型指定等读取参数

### 3. 数据存储层（savers）

统一数据落盘接口，支持文件存储与数据库存储两种模式，对接标准 `TabularData` 对象，屏蔽底层存储差异。

### 4. 扩展能力

- **HTML 渲染**：将表格数据渲染为可交互 HTML 页面，用于数据预览与报告生成
- **工具集**：路径规范化、临时目录管理等通用工具函数
- **测试辅助**：快速生成标准化测试数据集，支撑单元测试

## 四、核心设计原则

1. **懒加载优先**：大文件原生支持分块迭代，避免全量加载导致内存溢出
2. **序列化安全**：内置 NaN、时间、路径等特殊类型的 JSON 兼容处理
3. **接口统一**：所有加载器、存储器均遵循统一基类，扩展新格式成本低
4. **元数据可追溯**：所有数据容器携带来源、处理链路、运行参数等元信息，便于问题排查

## 五、标准使用流程

1. 通过 `UniversalLoader` 加载文件，得到标准 `TabularData` 对象
2. 业务处理：小数据全量处理调用 `to_dataframe()`，大数据集通过 `iter_chunks()` 分块处理
3. 处理结果封装为 `ProcessedTabularData` 或 `SpecialData`
4. 通过 `savers` 模块落盘/写入数据库，或通过 `html` 模块渲染预览

```