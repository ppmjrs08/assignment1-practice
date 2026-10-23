# 本次本地运行的实际结果

日期：2026-10-22。来自学习者亲自执行的checkpoints/quickstart运行。

- training_summary.json：配置、来源哈希、损失、PPL、耗时和训练量。
- training_metrics.jsonl：实际逐阶段日志；训练loss是最近日志区间均值，验证loss/PPL使用固定采样窗口。
- loss.png：对应曲线。
- generated.txt：未修正文法和情节的原始生成文本。
- generation.json：权重哈希、提示、采样参数及训练步数；本机绝对路径已替换为项目相对路径，数值和生成内容保留。

训练使用本地TinyStories验证文本重新划分的教学训练/验证集，并非原版官方标准数据划分。checkpoint和语料未随仓库上传；按项目README重新准备数据和训练即可运行生成。
