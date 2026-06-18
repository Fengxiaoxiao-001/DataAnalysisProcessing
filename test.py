import unittest
import pandas as pd
from datamineana.dataloader import ExcelLoader
from datamineana.test import TestTempCSV
from datamineana.processors import DataPreprocessor
from datamineana.dataobject import TabularData

class TestExcelLoaderChunkLazy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """全局只执行一次：创建自定义目录下的临时csv测试文件"""
        # 构造带缺失值、重复行、异常值、字符串、时间字段的测试数据集
        test_data = pd.DataFrame({
            "id": [1, 2, 2, 3, 4, 5, None, 7, 8, 9],
            "age": [18, 25, 25, 30, 150, 22, 27, None, 33, 45],
            "gender": [" M ", "female", "M", "F", "female", " m ", None, "F", "M", "female"],
            "register_time": pd.date_range("2024-01-01", periods=10, freq="D"),
            "score": [88.5, 92.0, 92.0, 77.2, 66.8, 95.3, None, 82.1, 79.5, 85.6]
        })
        # 生成临时CSV文件
        cls.temp_csv = TestTempCSV(data=test_data)
        cls.temp_path = cls.temp_csv.get_path()

    @classmethod
    def tearDownClass(cls):
        """测试结束自动清理临时文件"""
        cls.temp_csv.clean()

    def setUp(self):
        """每个用例执行前初始化加载器"""
        # 每个用例初始化：加载数据 + 初始化预处理入口
        self.loader = ExcelLoader()
        self.raw_tabular = self.loader.load(self.temp_path, lazy=False)
        print(self.raw_tabular.to_dataframe())
        self.preprocessor = DataPreprocessor()

    def test_preprocessor_init(self):
        """测试预处理入口类正常实例化"""
        self.assertIsInstance(self.preprocessor, DataPreprocessor)
        self.assertEqual(len(self.preprocessor.reports), 0)

    def test_handle_missing(self):
        """测试缺失值填充（均值填充数值列）"""
        data = self.preprocessor.handle_missing(
            self.raw_tabular,
            strategy="mean",
            columns=["age", "score"]
        )
        df = data.to_dataframe()
        # 验证缺失值已填充
        self.assertEqual(df["age"].isna().sum(), 0)
        self.assertEqual(df["score"].isna().sum(), 0)
        # 验证报告收集成功
        self.assertEqual(len(self.preprocessor.reports), 1)

    def test_drop_duplicates(self):
        """测试重复行删除"""
        data = self.preprocessor.drop_duplicates(self.raw_tabular, subset=["id", "age"])
        df = data.to_dataframe()
        # 原始10行，重复1行，处理后9行
        self.assertEqual(df.shape[0], 9)

    def test_handle_outliers(self):
        """测试IQR异常值截尾处理"""
        data = self.preprocessor.handle_outliers(
            self.raw_tabular,
            method="iqr",
            action="cap",
            columns=["age"]
        )
        df = data.to_dataframe()
        # 异常值150被截断，不再大于合理上界
        self.assertLess(df["age"].max(), 150)

    def test_normalize_text(self):
        """测试字符串清洗：去空格、小写"""
        data = self.preprocessor.normalize_text(
            self.raw_tabular,
            columns=["gender"],
            strip=True,
            lower=True
        )
        df = data.to_dataframe()
        self.assertEqual(df.loc[0, "gender"], "m")
        self.assertEqual(df.loc[1, "gender"], "female")

    def test_convert_types(self):
        """测试字段类型转换"""
        data = self.preprocessor.convert_types(
            self.raw_tabular,
            dtype_mapping={
                "id": "Int64",
                "gender": "category",
                "register_time": "datetime"
            }
        )
        df = data.to_dataframe()
        self.assertEqual(str(df["id"].dtype), "Int64")
        self.assertEqual(str(df["gender"].dtype), "category")

    def test_train_valid_test_split(self):
        """测试数据集随机划分，返回三元TabularData"""
        train, valid, test = self.preprocessor.train_valid_test_split(
            self.raw_tabular,
            train_size=0.7,
            valid_size=0.15,
            test_size=0.15,
            random_state=42
        )
        self.assertIsInstance(train, TabularData)
        self.assertIsInstance(valid, TabularData)
        self.assertIsInstance(test, TabularData)

    def test_run_pipeline(self):
        """测试流水线批量执行多个预处理步骤"""
        steps = [
            {"method": "handle_missing", "params": {"strategy": "median", "columns": ["age", "score"]}},
            {"method": "normalize_text", "params": {"columns": ["gender"], "strip": True, "lower": True}},
            {"method": "drop_duplicates", "params": {"subset": ["id"]}}
        ]
        result = self.preprocessor.run_pipeline(self.raw_tabular, steps=steps)
        self.assertIsInstance(result, TabularData)
        self.assertEqual(len(self.preprocessor.reports), 3)
        # 获取最后一步报告
        last_report = self.preprocessor.last_report()
        self.assertIsNotNone(last_report)

    def test_report_collection(self):
        """测试全流程报告收集、清空能力"""
        self.preprocessor.handle_missing(self.raw_tabular, strategy="mean")
        self.preprocessor.drop_duplicates(self.raw_tabular)
        all_reports = self.preprocessor.get_reports()
        print(all_reports)
        self.assertEqual(len(all_reports), 2)
        # 清空报告
        self.preprocessor.clear_reports()
        self.assertEqual(len(self.preprocessor.reports), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
