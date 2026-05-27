# -*- coding: utf-8 -*-
"""
TST 钟摆运动检测 — Python 示例脚本
TST Pendulum Motion Detection — Python Example Script

用法 / Usage:
    python tst_pendulum_example.py <ethovision_export.txt> [--output result.csv]

对 EthoVision XT 导出的悬尾实验数据进行钟摆检测，
逐帧判定"静止"或"挣扎"状态。

Runs pendulum detection on EthoVision XT exported TST data,
classifying each frame as "Still" or "Struggling".
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd


# ============================================================
# ===== 可配置参数 / Configurable Parameters =====
# ============================================================

# 以下参数基于 25fps（1帧=0.04秒）设定。
# 如果你的采样率不同，请按比例调整帧数参数。
# Parameters below are set for 25fps (1 frame = 0.04s).
# Adjust frame counts proportionally for other sample rates.

SMOOTH_WINDOW = 1             # 预处理平滑窗口（帧）/ Pre-smoothing window (frames)
ANALYSIS_WINDOW = 25           # 自相关分析窗口（帧）/ Autocorrelation window (frames)
PERIOD_MIN = 4                 # 最短搜索周期（帧）/ Min pendulum period (frames)
PERIOD_MAX = 12                # 最长搜索周期（帧）/ Max pendulum period (frames)
PERIODICITY_THRESHOLD = 0.55   # 周期性强度阈值 / Periodicity strength threshold
ACTIVITY_STRUGGLE_THRESHOLD = 2.0   # 高 Activity 挣扎阈值 / High activity threshold
MIN_STILL_ACTIVITY = 0.3       # 极低 Activity 静止阈值 / Very low activity threshold
MODERATE_ACTIVITY_THRESHOLD = 1.0   # 中等 Activity 挣扎阈值 / Moderate activity threshold
MIN_STATE_DURATION = 25        # 状态最短持续帧数 / Min state duration (frames)
PENDULUM_GRACE_PERIOD = 20     # 钟摆宽容期帧数 / Pendulum grace period (frames)


# ============================================================
# ===== 数据读取 / Data Loading =====
# ============================================================

def load_ethovision_data(filepath):
    """
    读取 EthoVision XT 导出文件。
    Read EthoVision XT export file (UTF-16 LE, semicolon-separated).

    Returns
    -------
    df : pd.DataFrame
    dt : float — 采样间隔（秒）/ sample interval (seconds)
    time_col : str — 时间列名 / time column name
    """
    with open(filepath, 'r', encoding='utf-16') as f:
        lines = f.readlines()

    # 第 36 行（索引 35）为列名，第 37 行为单位，第 38 行起为数据
    # Line 36 (index 35) = headers, line 37 = units, line 38+ = data
    header_line = lines[35].strip()
    headers = [h.strip('"') for h in header_line.split(';')]

    df = pd.read_csv(filepath, encoding='utf-16', sep=';',
                     skiprows=37, names=headers)
    df = df.dropna(axis=1, how='all')

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    time_col = df.columns[0]
    df = df.dropna(subset=[time_col]).reset_index(drop=True)

    dt = df[time_col].diff().median()

    return df, dt, time_col


# ============================================================
# ===== 检测算法 / Detection Algorithm =====
# ============================================================

def detect_pendulum(activity, dt=0.04):
    """
    对 Activity 序列运行钟摆检测算法。
    Run pendulum detection on an Activity time series.

    Parameters
    ----------
    activity : array-like
        Activity 百分比值序列 (Activity % time series, 0~100)
    dt : float
        采样间隔秒数 (sampling interval in seconds)

    Returns
    -------
    results : list of dict
        每帧的检测结果 / Per-frame detection results:
        - state: 0=静止(Still), 1=挣扎(Struggling)
        - periodicity: 周期性强度 (0~1)
        - is_pendulum: 是否为钟摆运动
    """
    # 自适应帧率 / Adapt frame params to sample rate
    scale = 0.04 / dt if dt > 0 and dt != 0.04 else 1.0
    min_dur = max(1, round(MIN_STATE_DURATION * scale))
    grace_max = max(0, round(PENDULUM_GRACE_PERIOD * scale))

    ring_buffer = [0.0] * ANALYSIS_WINDOW
    ring_idx = 0
    smooth_buffer = []
    output_state = 0  # 0=still, 1=struggling
    pending_state = -1
    pending_count = 0
    grace_counter = 0
    results = []

    for i in range(len(activity)):
        raw = activity[i]

        if np.isnan(raw):
            results.append({'state': 0, 'periodicity': 0, 'is_pendulum': False})
            continue

        # Phase 1: 预处理平滑 / Pre-smoothing
        smooth_buffer.append(raw)
        if len(smooth_buffer) > SMOOTH_WINDOW:
            smooth_buffer.pop(0)
        smoothed = sum(smooth_buffer) / len(smooth_buffer)

        # Phase 2: 环形缓冲区 / Ring buffer
        ring_buffer[ring_idx % ANALYSIS_WINDOW] = smoothed
        ring_idx += 1

        if ring_idx < ANALYSIS_WINDOW:
            results.append({'state': 1, 'periodicity': 0, 'is_pendulum': False})
            continue

        # 计算窗口均值 / Compute window mean
        n = ANALYSIS_WINDOW
        mean_act = sum(ring_buffer[(ring_idx - n + j) % n] for j in range(n)) / n

        # Phase 3: 自相关周期性检测 / Autocorrelation periodicity
        norm_data = [ring_buffer[(ring_idx - n + j) % n] - mean_act for j in range(n)]
        energy = sum(v * v for v in norm_data)

        max_ac = 0.0
        if energy > 1e-10:
            max_lag = min(PERIOD_MAX + 1, n // 2)
            for lag in range(PERIOD_MIN, max_lag):
                ac = sum(norm_data[j] * norm_data[j + lag] for j in range(n - lag))
                ac /= energy
                if ac > max_ac:
                    max_ac = ac

        periodicity = max(0.0, min(1.0, max_ac))

        # Phase 4: 宽容期更新 / Grace period update
        if periodicity > PERIODICITY_THRESHOLD:
            grace_counter = grace_max
        elif grace_counter > 0:
            grace_counter -= 1

        # Phase 5: 状态判定 / State decision
        recent_pendulum = grace_counter > 0

        if mean_act < MIN_STILL_ACTIVITY:
            state = 0
        elif periodicity > PERIODICITY_THRESHOLD:
            state = 0
        elif mean_act > ACTIVITY_STRUGGLE_THRESHOLD:
            state = 1
        elif mean_act > MODERATE_ACTIVITY_THRESHOLD:
            state = 1
        elif recent_pendulum:
            state = 0
        else:
            state = 1

        is_pendulum = periodicity > PERIODICITY_THRESHOLD and mean_act >= MIN_STILL_ACTIVITY

        # Phase 6: 持续时间过滤 / Duration filter
        if state == output_state:
            pending_state = -1
            pending_count = 0
        elif state == pending_state:
            pending_count += 1
            if pending_count >= min_dur:
                output_state = state
                pending_state = -1
                pending_count = 0
        else:
            pending_state = state
            pending_count = 1

        results.append({
            'state': output_state,
            'periodicity': periodicity,
            'is_pendulum': is_pendulum,
        })

    return results


# ============================================================
# ===== 结果输出 / Result Output =====
# ============================================================

def summarize_results(results, dt):
    """
    汇总检测结果 / Summarize detection results.
    """
    total = len(results)
    struggling = sum(1 for r in results if r['state'] == 1)
    still = sum(1 for r in results if r['state'] == 0)
    pendulum = sum(1 for r in results if r['is_pendulum'])

    print(f'\n--- 检测结果汇总 / Detection Summary ---')
    print(f'总帧数 / Total frames:        {total}')
    print(f'挣扎帧 / Struggling frames:   {struggling} ({struggling/total*100:.1f}%)')
    print(f'静止帧 / Still frames:         {still} ({still/total*100:.1f}%)')
    print(f'其中钟摆 / Pendulum frames:    {pendulum} ({pendulum/total*100:.1f}%)')
    print(f'挣扎时长 / Struggling duration: {struggling * dt:.1f}s')
    print(f'静止时长 / Still duration:      {still * dt:.1f}s')

    # 找出静止段 / Find still segments
    segments = []
    in_still = False
    start = 0
    for i in range(len(results)):
        if results[i]['state'] == 0 and not in_still:
            start = i
            in_still = True
        elif results[i]['state'] != 0 and in_still:
            segments.append((start, i - 1))
            in_still = False
    if in_still:
        segments.append((start, len(results) - 1))

    print(f'\n静止段数 / Still segments: {len(segments)}')
    for idx, (s, e) in enumerate(segments):
        dur = (e - s + 1) * dt
        t_start = s * dt
        t_end = e * dt
        has_pendulum = any(results[j]['is_pendulum'] for j in range(s, min(e + 1, len(results))))
        label = '钟摆/Pendulum' if has_pendulum else '真静止/True still'
        print(f'  #{idx+1}: {t_start:.1f}s - {t_end:.1f}s ({dur:.1f}s) [{label}]')


def export_csv(results, dt, output_path):
    """
    导出逐帧结果为 CSV / Export per-frame results to CSV.
    """
    rows = []
    for i, r in enumerate(results):
        rows.append({
            'frame': i,
            'time_sec': round(i * dt, 3),
            'state': 'Still' if r['state'] == 0 else 'Struggling',
            'periodicity': round(r['periodicity'], 4),
            'is_pendulum': r['is_pendulum'],
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'\n结果已导出 / Results exported to: {output_path}')


# ============================================================
# ===== 主函数 / Main =====
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='TST 钟摆运动检测 / TST Pendulum Motion Detection')
    parser.add_argument('input', help='EthoVision XT 导出文件路径 / Path to EthoVision XT export file')
    parser.add_argument('--output', '-o', help='输出 CSV 路径 / Output CSV path')
    parser.add_argument('--column', '-c', default=None,
                        help='Activity 列名（自动检测如果省略）/ Activity column name (auto-detect if omitted)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'文件不存在 / File not found: {args.input}')
        sys.exit(1)

    # 读取数据 / Load data
    df, dt, time_col = load_ethovision_data(args.input)
    print(f'数据文件 / Data file: {args.input}')
    print(f'帧数 / Frames: {len(df)}, 采样间隔 / dt: {dt:.4f}s ({1/dt:.1f} fps)')

    # 找到 Activity 列 / Find Activity column
    activity_col = args.column
    if activity_col is None:
        for col in df.columns:
            if 'activity' in col.lower():
                activity_col = col
                break

    if activity_col is None:
        print(f'未找到 Activity 列 / Activity column not found')
        print(f'可用列 / Available columns: {list(df.columns)}')
        sys.exit(1)

    print(f'Activity 列 / Activity column: {activity_col}')

    # 运行检测 / Run detection
    activity = df[activity_col].fillna(0).values
    results = detect_pendulum(activity, dt)

    # 输出结果 / Output results
    summarize_results(results, dt)

    if args.output:
        export_csv(results, dt, args.output)


if __name__ == '__main__':
    main()
