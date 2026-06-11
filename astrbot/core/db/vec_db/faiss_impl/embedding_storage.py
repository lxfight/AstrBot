import asyncio
import os

import numpy as np


class EmbeddingStorage:
    def __init__(self, dimension: int, path: str | None = None) -> None:
        try:
            import faiss
        except ModuleNotFoundError as e:
            raise ImportError(
                "faiss 未安装。请使用 'pip install faiss-cpu' 或 'pip install faiss-gpu' 安装。",
            ) from e
        self._faiss = faiss
        self.dimension = dimension
        self.path = path
        self.index = None
        if path and os.path.exists(path):
            self.index = faiss.read_index(path)
        else:
            base_index = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIDMap(base_index)

    async def insert(self, vector: np.ndarray, id: int) -> None:
        """插入向量

        Args:
            vector (np.ndarray): 要插入的向量
            id (int): 向量的ID
        Raises:
            ValueError: 如果向量的维度与存储的维度不匹配

        """
        assert self.index is not None, "FAISS index is not initialized."
        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"向量维度不匹配, 期望: {self.dimension}, 实际: {vector.shape[0]}",
            )
        self.index.add_with_ids(vector.reshape(1, -1), np.array([id]))
        await self.save_index()

    async def insert_batch(self, vectors: np.ndarray, ids: list[int]) -> None:
        """批量插入向量

        Args:
            vectors (np.ndarray): 要插入的向量数组
            ids (list[int]): 向量的ID列表
        Raises:
            ValueError: 如果向量的维度与存储的维度不匹配

        """
        assert self.index is not None, "FAISS index is not initialized."
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"向量维度不匹配, 期望: {self.dimension}, 实际: {vectors.shape[1]}",
            )
        self.index.add_with_ids(vectors, np.array(ids))
        await self.save_index()

    async def search(self, vector: np.ndarray, k: int) -> tuple:
        """搜索最相似的向量

        Args:
            vector (np.ndarray): 查询向量
            k (int): 返回的最相似向量的数量
        Returns:
            tuple: (距离, 索引)

        """
        assert self.index is not None, "FAISS index is not initialized."
        self._faiss.normalize_L2(vector)
        distances, indices = self.index.search(vector, k)
        return distances, indices

    async def delete(self, ids: list[int]) -> None:
        """删除向量

        Args:
            ids (list[int]): 要删除的向量ID列表

        """
        assert self.index is not None, "FAISS index is not initialized."
        id_array = np.array(ids, dtype=np.int64)
        self.index.remove_ids(id_array)
        await self.save_index()

    async def save_index(self) -> None:
        """原子性保存索引到磁盘

        使用临时文件 + os.replace() 确保原子性，防止进程崩溃导致索引损坏。
        """
        if self.index is None:
            return
        if not self.path:
            return

        # 原子性保存：先写临时文件，成功后再替换
        temp_path = f"{self.path}.tmp.{os.getpid()}"
        try:
            await asyncio.to_thread(self._faiss.write_index, self.index, temp_path)
            # 使用 os.replace 确保原子性（POSIX 保证）
            await asyncio.to_thread(os.replace, temp_path, self.path)
        except Exception as exc:
            # 清理临时文件
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise RuntimeError(
                f"保存 FAISS 索引失败: {exc}。索引未更新，保持原有状态。"
            ) from exc
