# 范式识别流程

## 主决策
1. read `/mnt/skills/ethovision-paradigm-knowledge/SKILL.md` 拿 EV19 模板决策树
2. 综合证据(不 read raw txt): 用户文字 + 文件名 + 已知 EV19 模板列表
3. 决策:
   - 唯一高置信 → set_experiment_paradigm
   - 多候选 → ask_clarification 带证据反问
   - 完全不知道 → ask_clarification 让用户从范式列表选

## 反问最多 1 次
如用户答"不知道":告诉用户"我需要范式信息才能选指标,请联系实验设计者确认",不要默认猜测。

## 已知 paradigm key
`epm` / `oft` / `fst` / `tst` / `ldb` / `zero_maze` / `shoaling`
