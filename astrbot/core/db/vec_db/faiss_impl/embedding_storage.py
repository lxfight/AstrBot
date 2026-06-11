import asyncio
import os

import numpy as np


def _safe_normalize_l2(vectors: np.ndarray) -> None:
    """L2 归一化，对零向量抛出明确错误"""
    import faiss

    if vectors.ndim == 2:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        zero_count = int((norms < 1e-12).sum())
        if zero_count > 0:
            raise ValueError(f"检测到 {zero_count} 个零向量。Embedding Provider 返回了异常数据。")
    elif vectors.ndim == 1:
        if np.linalg.norm(vectors) < 1e-12:
            raise ValueError("检测到零向量。Embedding Provider 返回了异常数据。")
    faiss.normalize_L2(vectors)


class EmbeddingStorage:
    def __init__(self, dimension: int, path: str | None = None, index_type: str = "flat") -> None:
        try:
            import faiss
        except ModuleNotFoundError as e:
            raise ImportError("faiss 未安装。请使用 'pip install faiss-cpu' 或 'pip install faiss-gpu' 安装。") from e
        self._faiss = faiss
        self.dimension = dimension
        self.path = path
        self.index = None
        self.index_type = index_type
        self._write_lock = asyncio.Lock()

        if path and os.path.exists(path):
            self.index = faiss.read_index(path)
            loaded_dim = self.index.d
            if loaded_dim != self.dimension:
                raise ValueError(f"索引维度不匹配: 磁盘={loaded_dim}, 当前={self.dimension}")
            self._migrate_l2_to_ip_if_needed()
        else:
            self.index = self._create_index()

    def _create_index(self):
        faiss = self._faiss
        if self.index_type == "hnsw":
            base_index = faiss.index_factory(self.dimension, "HNSW32", faiss.METRIC_INNER_PRODUCT)
            return faiss.IndexIDMap(base_index)
        return faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))

    def _migrate_l2_to_ip_if_needed(self) -> None:
        faiss = self._faiss
        assert self.index is not None
        base_index = self.index.index if hasattr(self.index, "index") else self.index
        if not isinstance(base_index, faiss.IndexFlatL2):
            return

        import warnings
        ntotal = self.index.ntotal
        if ntotal == 0:
            warnings.warn("检测到空的旧版 L2 索引，将重建为 IP 索引。")
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))
            return

        warnings.warn(f"检测到旧版 L2 索引 (含 {ntotal} 个向量)，正在自动迁移到 IP 索引...")
        try:
            vectors = np.zeros((ntotal, self.dimension), dtype=np.float32)
            for i in range(ntotal):
                vectors[i] = self.index.reconstruct(i)
        except RuntimeError:
            warnings.warn("无法从旧索引重建向量，将重建空 IP 索引。")
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))
            faiss.write_index(self.index, self.path)
            return

        _safe_normalize_l2(vectors)
        new_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))
        new_index.add_with_ids(vectors, np.arange(ntotal, dtype=np.int64))
        self.index = new_index
        faiss.write_index(self.index, self.path)

    async def insert(self, vector: np.ndarray, id: int) -> None:
        async with self._write_lock:
            assert self.index is not None
            if vector.shape[0] != self.dimension:
                raise ValueError(f"向量维度不匹配, 期望: {self.dimension}, 实际: {vector.shape[0]}")
            vec = vector.reshape(1, -1).copy()
            _safe_normalize_l2(vec)
            self.index.add_with_ids(vec, np.array([id]))
            await self._save_index_locked()

    async def insert_batch(self, vectors: np.ndarray, ids: list[int]) -> None:
        async with self._write_lock:
            assert self.index is not None
            if vectors.shape[1] != self.dimension:
                raise ValueError(f"向量维度不匹配, 期望: {self.dimension}, 实际: {vectors.shape[1]}")
            vecs = vectors.copy()
            _safe_normalize_l2(vecs)
            self.index.add_with_ids(vecs, np.array(ids))
            await self._save_index_locked()

    async def search(self, vector: np.ndarray, k: int) -> tuple:
        assert self.index is not None
        vec = vector.copy()
        _safe_normalize_l2(vec)
        distances, indices = self.index.search(vec, k)
        distances = (distances + 1.0) / 2.0
        return distances, indices

    async def delete(self, ids: list[int]) -> None:
        async with self._write_lock:
            assert self.index is not None
            id_array = np.array(ids, dtype=np.int64)
            self.index.remove_ids(id_array)
            await self._save_index_locked()

    async def _save_index_locked(self) -> None:
        if self.index is None:
            return
        if not self.path:
            raise RuntimeError("无法保存 FAISS 索引：索引文件路径未设置。")

        temp_path = f"{self.path}.tmp.{os.getpid()}"
        try:
            await asyncio.to_thread(self._faiss.write_index, self.index, temp_path)
            await asyncio.to_thread(os.replace, temp_path, self.path)
        except Exception as exc:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise RuntimeError(f"保存 FAISS 索引失败: {exc}") from exc

    async def save_index(self) -> None:
        async with self._write_lock:
            await self._save_index_locked()
