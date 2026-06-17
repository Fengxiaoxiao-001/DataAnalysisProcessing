import unittest
import pandas as pd
from datamineana.dataloader import ExcelLoader
from datamineana.test import TestTempCSV


class TestExcelLoaderChunkLazy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """全局只执行一次：创建自定义目录下的临时csv测试文件"""
        # 生成测试数据：共25000行
        test_df = pd.DataFrame({
            "col1": list(range(25000)),
            "col2": [f"test_{i}" for i in range(25000)]
        })
        # 使用你封装的临时文件类，默认存放在 项目根目录/cache/test_data
        cls.temp_csv = TestTempCSV(data=test_df)
        cls.temp_path = cls.temp_csv.get_path()

    @classmethod
    def tearDownClass(cls):
        """测试结束自动清理临时文件"""
        cls.temp_csv.clean()

    def setUp(self):
        """每个用例执行前初始化加载器"""
        self.loader = ExcelLoader()

    def test_loader_init(self):
        """测试1：加载器可以正常实例化"""
        self.assertIsInstance(self.loader, ExcelLoader)

    def test_lazy_chunk_load(self):
        """测试2：懒加载+分块读取正常"""
        data = self.loader.load(
            path=self.temp_path,
            lazy=True,
            chunksize=10000
        )
        self.assertIsNotNone(data)

    def test_view_to_json(self):
        """测试3：view方法可以正常转JSON不抛异常"""
        data = self.loader.load(path=self.temp_path, lazy=True, chunksize=10000)
        json_str = data.view_info().to_json()
        print(json_str)
        self.assertIsInstance(json_str, str)
        self.assertTrue(len(json_str) > 0)

    def test_iter_chunks(self):
        """测试4：迭代分块，每个块行数≤10000，总共3块"""
        data = self.loader.load(path=self.temp_path, lazy=True, chunksize=10000)
        chunk_list = []
        for chunk in data.iter_chunks():
            chunk_list.append(chunk)
            rows, _ = chunk.shape
            self.assertLessEqual(rows, 10000)
            self.assertIsInstance(chunk, pd.DataFrame)

        self.assertEqual(len(chunk_list), 3)

    def test_file_not_found_exception(self):
        """测试5：传入不存在路径需要抛出异常"""
        with self.assertRaises(FileNotFoundError):
            self.loader.load(
                path="not_exist_file.csv",
                lazy=True,
                chunksize=10000
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)
