#!/bin/bash
# 自动创建 6 个知识库 PR 的脚本

set -e

echo "=== 提交并推送 PR1: feat/kb-batch-delete-documents ==="
git checkout feat/kb-batch-delete-documents
git add -A
git commit -m "feat(kb): add batch delete documents API

- Add delete_documents_by_ids in KBSQLiteDatabase
- Add delete_documents in KBHelper
- Add batch-delete API endpoint (max 100 docs)
- Parallel vector deletion with best-effort handling
- Add comprehensive tests" || echo "Already committed"
git push -u origin feat/kb-batch-delete-documents

echo ""
echo "=== PR1 完成，GitHub URL: ==="
echo "https://github.com/lxfight/AstrBot/compare/master...feat/kb-batch-delete-documents"
echo ""
echo "按回车继续创建 PR2..."
read

# TODO: 创建 PR2-PR6 的命令将在下面添加
