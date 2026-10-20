# 评测目录

`questions.csv` 与 `questions.json` 包含同一组20题、标准答案和出处

`results/bm25.csv` 与 `results/vector.csv` 各20条
包含原始最终回答、标准答案、耗时和最终评分

`results/bm25.json` 与 `results/vector.json` 各20条
包含对应运行的完整检索和模型输出记录

`run-20261008-102913/predictions.jsonl` 是按实际执行顺序保存的40条合并原始记录
`human_review.csv` 是最终评分表
`summary.json` 是统计结果
`environment.json` 是实际环境快照
`scoring-provenance.json` 说明作者抽查及授权 AI 复核的评分来源

结果按最终交付答案评分
关键条件遗漏、错误结论、误拒答或格式拦截记错
无答案题中的最终交付编造单独计入陷阱题幻觉率

`auto_hit_all` 仅看 Top-3
`context_hit_all` 包含追加上下文
两个指标不能混用

当前资料文件名为 `AHGNN论文.pdf`
历史运行记录保留旧文件名 `HAGNN论文.pdf`
当前题集只更正来源编号，问题、标准答案与原评分不变
文件名与编号映射见 `document-rename.json`
