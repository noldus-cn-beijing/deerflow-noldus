# 5月21日产品反馈

## 悬尾新检测方法
由于悬尾的检测噪声相对强迫游泳更多（主要是悬挂摆动），增加了在使用Activity测量时的新`JavaScript状态`因变量脚本和计算方法。其中包含：
- 算法说明文档 — /tstYoyo/tst-pendulum-algorithm.md
  - 问题背景与核心思路
  - 6 个 Phase 的完整流程图和逐阶段说明
  - 10 个参数的表格（默认值、含义、调参建议）
  - 数据格式要求、输入输出规格、固有限制  
- Python 示例脚本 — /tstYoyo/tst_pendulum_example.py
  - 可独立运行，包含完整的 EthoVision XT 数据读取 + 检测算法 + 结果汇总 + CSV 导出
  - 命令行用法：python tst_pendulum_example.py <导出文件.txt> [--output result.csv]
  - 已通过两个实验数据验证
- EthoVision XT JS 代码 — tstYoyo/TST_PendulumDetect.js
  - 实时逐帧检测版本，直接粘贴到 EthoVision XT 的 Analysis Profile 中使用
  - 变量类型选 "State"，输出数量设为 1
  - 包含帧率自适应（通过 GetSampleTime() 自动检测）
- 相关指标指标和计算方法说明 — /tstYoyo/manual/

## 范式-图表对应关系更新
> “需要目前的5种实验范式所经常性生成的图对齐。需要填写skill。每种范式出的图按照置信度排名。完全不用，没有必要用的图可以不写。大概就是每种实验必出的图，可能用到的图，很少有人用到的图”

对MVP阶段6种范式该出什么图按照使用频率进行限定。参考 /范式-图表对应关系.md
