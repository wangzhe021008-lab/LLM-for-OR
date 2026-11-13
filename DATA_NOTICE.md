# 算例来源与许可

Marcella Papini (2026), *Benchmark and Adapted Instances for the Capacitated Location-Routing Problem Study Based on Prins Instances*, Mendeley Data, Version 1. DOI 10.17632/dzntyc2r28.1。作者说明原始基准出自 C. Prodhon (2006)，并在此基础上扩展了绿色选址与路径算例。[数据集发布页](https://data.mendeley.com/datasets/dzntyc2r28/1)

该数据集按 [Creative Commons Attribution 4.0 International（CC BY 4.0）](https://creativecommons.org/licenses/by/4.0/) 发布。许可适用于此处引用的数据集。转发或改编算例时请保留署名、来源和许可，并注明所作修改。

`data/raw/` 保留发布的原始 Excel，文件内容未修改，SHA-256 为 `fc683bba6ab31294e0ecdefd4b7ee1cdc98a3e966d9b2b14892224496c24e6c3`。`data/processed/` 为本项目的派生数据，包括保留公式和缓存值的单元格快照、按容量拆分的 JSON 算例及清单。转换逻辑在 `orllm/data.py` 中，可用 `python main.py prepare` 重现。
