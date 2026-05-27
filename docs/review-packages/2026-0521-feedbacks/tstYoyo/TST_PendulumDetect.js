// -*- coding: utf-8 -*-
// TST Pendulum Motion Detection - EthoVision XT JavaScript
// 悬尾实验钟摆运动检测 - EthoVision XT JavaScript 脚本
//
// 使用自相关周期性检测区分悬尾实验中的钟摆式摆动与真实挣扎。
// 输出为 State 变量：1 = 静止 (Still，含钟摆)，0 = 挣扎 (Struggling)。
//
// Uses autocorrelation periodicity detection to distinguish pendulum
// oscillation from real struggling in Tail Suspension Test.
// Output is a State variable: 1 = Still (incl. pendulum), 0 = Struggling.
//
// 在 EthoVision XT 的 Analysis Profile 中添加此脚本，
// 变量类型选择 "State"，输出数量设为 1。
// Add this script to an Analysis Profile in EthoVision XT,
// variable type: "State", output count: 1.
//
// ★ 关于帧数：这里的"帧"指 EthoVision XT 的采样间隔。
//   如果视频为25fps且每帧都采样，则1帧=0.04秒。
//   如果2帧采样一次，则1帧=0.08秒。
//   "Frame" here = EthoVision XT sample interval.
//   25fps every frame → 1 frame = 0.04s.
//   25fps every 2nd frame → 1 frame = 0.08s.
//
// ★ 实时检测固有的延迟：算法存在约 ANALYSIS_WINDOW/2 + MIN_STATE_DURATION
//   帧的检测延迟（约1.2秒），这是滑动窗口方法不可避免的。
//   Inherent real-time delay: ~ANALYSIS_WINDOW/2 + MIN_STATE_DURATION frames (~1.2s).
//   This is unavoidable with the sliding window approach.


// ============================================================
// ===== 可配置参数 / Configurable Parameters =====
// ============================================================
// 请根据你的实验设置调整以下参数。
// Please adjust the following parameters according to your experiment.
// ============================================================

// --- 预处理平滑窗口 / Pre-smoothing window ---
// 作用：对原始Activity做移动平均，减少单帧噪点干扰。
// 填法：填整数帧数。数值越大越平滑，但会损失时间分辨率。
//       建议范围 1~10。填1表示不平滑。
//       ★ 重要：过度平滑会模糊钟摆的周期性特征，建议不超过3。
var SMOOTH_WINDOW = 1;

// --- 自相关分析窗口 / Autocorrelation analysis window ---
// 作用：回溯观察多长时间的数据来判断是否存在钟摆运动。
//       窗口越长判定越稳，但响应越慢。
// 填法：填整数帧数。建议覆盖3~5个完整钟摆周期。
//       例如钟摆周期约0.3秒(约8帧@25fps)，则填24~40。
//       ★ 重要：窗口过大会稀释短钟摆段落的周期性特征。
//       ★ 重要：窗口越大，检测延迟越大（约等于窗口长度的一半）。
//       ★ 建议在25fps下使用25~30，在更低采样率下适当缩小。
var ANALYSIS_WINDOW = 25;

// --- 钟摆周期搜索范围 / Pendulum period search range ---
// 作用：算法在这个范围内自动寻找钟摆的摆动周期。
// 填法：PERIOD_MIN = 最短可能的周期（帧数），
//       PERIOD_MAX = 最长可能的周期（帧数）。
//       不同大小/品系的鼠摆动频率不同，范围宁宽勿窄。
//       建议范围覆盖0.15秒~0.6秒所对应的帧数。
//       ★ 重要：最短周期不要设得太小，否则会检测到帧间噪声。
//       ★ 重要：PERIOD_MAX 不得超过 ANALYSIS_WINDOW/2。
var PERIOD_MIN = 4;
var PERIOD_MAX = 12;

// --- 周期性强度阈值 / Periodicity strength threshold ---
// 作用：自相关检测结果越接近1，说明周期性越强。
//       低于此阈值则不认为是钟摆运动。
// 填法：0~1之间的小数。越低越容易检出钟摆，越高越严格。
//       钟摆通常在0.5以上，挣扎通常在0.3以下。
//       建议0.40~0.60之间调试。
var PERIODICITY_THRESHOLD = 0.55;

// --- 挣扎Activity阈值 / Struggling activity threshold ---
// 作用：当Activity均值明显高于钟摆水平时，直接判定为挣扎。
// 填法：Activity百分比值（0~100）。
//       请观察你的数据中挣扎时的Activity范围来设定。
var ACTIVITY_STRUGGLE_THRESHOLD = 2.0;

// --- 极低Activity阈值 / Very low activity threshold ---
// 作用：Activity均值低于此值时直接判定为静止（真静止）。
// 填法：Activity百分比值（0~100）。建议0.2~0.5。
var MIN_STILL_ACTIVITY = 0.3;

// --- 中等Activity挣扎阈值 / Moderate activity struggling threshold ---
// 作用：当Activity均值高于此值但低于挣扎阈值，且未检测到周期性时，
//       判定为挣扎。用于防止将中等强度的挣扎误判为静止。
// 填法：Activity百分比值（0~100）。建议0.8~1.5。
//       值越低越容易将中等Activity判为挣扎（减少假静止）。
var MODERATE_ACTIVITY_THRESHOLD = 1.0;

// --- 状态最短持续帧数 / Minimum state duration ---
// 作用：避免状态在挣扎/静止之间频繁跳变。
//       新状态必须连续出现超过此帧数才会被确认输出。
// 填法：整数帧数（基于25fps换算，其他帧率自动适配）。
//       越小响应越快但越容易跳变。
var MIN_STATE_DURATION = 25;

// --- 钟摆宽容期帧数 / Pendulum grace period ---
// 作用：当周期性信号在阈值附近波动时，保持"近期检测到钟摆"的上下文。
//       在宽容期内，低Activity无周期性的帧仍判为静止（钟摆过渡区域）。
//       超出宽容期后，低Activity无周期性 → 判为挣扎（挣扎暂停）。
// 填法：整数帧数（基于25fps换算，其他帧率自动适配）。
//       填0禁用此功能（退回原始行为）。
var PENDULUM_GRACE_PERIOD = 20;


// ============================================================
// ===== 内部变量 / Internal Variables =====
// ============================================================

var g_ringBuffer;      // 环形缓冲区 / Ring buffer
var g_ringIdx;         // 环形缓冲区写入位置 / Ring buffer write index
var g_smoothBuffer;    // 平滑缓冲区 / Smoothing buffer
var g_outputState;     // 当前输出状态 / Current output state
var g_pendingState;    // 待确认状态 / Pending state (not yet confirmed)
var g_pendingCount;    // 待确认计数 / Pending state consecutive count
var g_graceCounter;    // 钟摆宽容期倒计时 / Pendulum grace countdown
var g_prevTime;        // 上一帧时间 / Previous sample time
var g_minStateDur;     // 自适应最短持续帧数 / Adapted min state duration
var g_graceMax;        // 自适应宽容期帧数 / Adapted grace period


// ============================================================
// ===== 生命周期函数 / Lifecycle Functions =====
// ============================================================

function Start() {
    // 试验开始时初始化 / Initialize at trial start
    g_ringBuffer = new Array(ANALYSIS_WINDOW);
    for (var i = 0; i < ANALYSIS_WINDOW; i++) g_ringBuffer[i] = 0;
    g_ringIdx = 0;
    g_smoothBuffer = [];
    g_outputState = 0;    // 默认静止 / Default still
    g_pendingState = -1;  // 无待确认 / No pending
    g_pendingCount = 0;
    g_graceCounter = 0;   // 无宽容期 / No grace
    g_prevTime = -1;
    g_minStateDur = MIN_STATE_DURATION;
    g_graceMax = PENDULUM_GRACE_PERIOD;
}

function Stop() {
    // 试验结束时清理 / Cleanup at trial end
}

function Process() {
    // 每个采样点调用一次 / Called once per sample
    var pixelChange = GetPixelChange();

    // --- 采样率自适应 / Sample rate adaptation ---
    // 参数基于25fps填写，检测到实际帧率后自动换算。
    // Parameters are authored at 25fps; auto-scale to actual sample rate.
    var sampleTime = GetSampleTime();
    if (g_prevTime >= 0 && g_minStateDur === MIN_STATE_DURATION) {
        var dt = sampleTime - g_prevTime;
        if (dt > 0 && dt !== 0.04) {
            var scale = 0.04 / dt;
            g_minStateDur = Math.max(1, Math.round(MIN_STATE_DURATION * scale));
            g_graceMax = Math.max(0, Math.round(PENDULUM_GRACE_PERIOD * scale));
        }
    }
    g_prevTime = sampleTime;

    if (pixelChange === null) {
        SetOutput(1 - g_outputState);
        return;
    }

    // Activity % = 像素变化比例 × 100
    // Activity % = pixel change ratio × 100
    var activity = pixelChange * 100;

    // --- Phase 1: 预处理平滑 / Pre-smoothing ---
    var smoothed = _smooth(activity);

    // --- Phase 2: 环形缓冲区 / Ring buffer ---
    g_ringBuffer[g_ringIdx % ANALYSIS_WINDOW] = smoothed;
    g_ringIdx++;

    if (g_ringIdx < ANALYSIS_WINDOW) {
        // 窗口未满，输出挣扎（实验初期不太可能放弃挣扎）
        // Window not full, output struggling (unlikely to give up early in trial)
        SetOutput(0);
        return;
    }

    // 计算窗口内均值（用于状态判定）
    // Compute window mean (for state decision)
    var n = ANALYSIS_WINDOW;
    var meanAct = 0;
    for (var j = 0; j < n; j++) {
        meanAct += g_ringBuffer[(g_ringIdx - n + j) % n];
    }
    meanAct /= n;

    // --- Phase 3: 自相关周期性检测 / Autocorrelation periodicity ---
    var periodicity = _detectPeriodicity(meanAct);

    // --- Phase 4: 钟摆宽容期更新 / Pendulum grace period update ---
    if (periodicity > PERIODICITY_THRESHOLD) {
        g_graceCounter = g_graceMax;
    } else if (g_graceCounter > 0) {
        g_graceCounter--;
    }

    // --- Phase 5: 状态判定 / State decision ---
    // 优先级：极低Activity → 周期性强(钟摆) → 高Activity(挣扎)
    //         → 中等Activity无周期性(挣扎) → 低Activity视宽容期判定
    // Priority: very low Activity → strong periodicity → high Activity
    //           → moderate Activity without periodicity
    //           → low Activity: grace period → still, no grace → struggling
    var state;
    var recentPendulum = g_graceCounter > 0;

    if (meanAct < MIN_STILL_ACTIVITY) {
        // Activity极低 → 静止（真静止）
        state = 0;
    } else if (periodicity > PERIODICITY_THRESHOLD) {
        // 周期性强 → 钟摆，归为静止
        state = 0;
    } else if (meanAct > ACTIVITY_STRUGGLE_THRESHOLD) {
        // 无周期性 + 高Activity → 挣扎
        state = 1;
    } else if (meanAct > MODERATE_ACTIVITY_THRESHOLD) {
        // 无周期性 + 中等Activity → 挣扎
        state = 1;
    } else if (recentPendulum) {
        // 低Activity + 近期有周期性 → 钟摆过渡区，归为静止
        state = 0;
    } else {
        // 低Activity + 无近期周期性 → 挣扎暂停，归为挣扎
        state = 1;
    }

    // --- Phase 6: 状态持续时间过滤 / State duration filter ---
    g_outputState = _filterDuration(state, g_minStateDur);

    SetOutput(1 - g_outputState);
}


// ============================================================
// ===== 辅助函数 / Helper Functions =====
// ============================================================

function _smooth(value) {
    // 移动平均平滑 / Moving average smoothing
    g_smoothBuffer.push(value);
    if (g_smoothBuffer.length > SMOOTH_WINDOW) {
        g_smoothBuffer.shift();
    }
    var sum = 0;
    for (var i = 0; i < g_smoothBuffer.length; i++) {
        sum += g_smoothBuffer[i];
    }
    return sum / g_smoothBuffer.length;
}


function _detectPeriodicity(mean) {
    // 自相关周期性检测 / Autocorrelation periodicity detection
    // 返回周期性强度 (0~1) / Returns periodicity strength (0~1)

    var n = ANALYSIS_WINDOW;
    var normData = new Array(n);
    var energy = 0;

    // 减均值 / Subtract mean
    for (var j = 0; j < n; j++) {
        var v = g_ringBuffer[(g_ringIdx - n + j) % n] - mean;
        normData[j] = v;
        energy += v * v;
    }

    if (energy < 1e-10) return 0;

    // 在搜索范围内找最大自相关 / Find max autocorrelation in search range
    var maxAC = 0;
    var maxLag = Math.min(PERIOD_MAX + 1, Math.floor(n / 2));

    for (var lag = PERIOD_MIN; lag < maxLag; lag++) {
        var ac = 0;
        for (var j = 0; j < n - lag; j++) {
            ac += normData[j] * normData[j + lag];
        }
        ac /= energy;
        if (ac > maxAC) {
            maxAC = ac;
        }
    }

    return Math.max(0, Math.min(1, maxAC));
}


function _filterDuration(newState, minDur) {
    // 状态持续时间过滤：新状态必须连续出现 minDur 帧才确认切换。
    // State duration filter: new state must appear for minDur
    // consecutive frames before being committed.

    if (newState === g_outputState) {
        // 与当前输出一致，无需切换 / Same as current, no change needed
        g_pendingState = -1;
        g_pendingCount = 0;
        return g_outputState;
    }

    if (newState === g_pendingState) {
        // 继续累积待确认状态 / Accumulate pending state
        g_pendingCount++;
        if (g_pendingCount >= minDur) {
            // 达到最短持续时间，确认切换 / Met min duration, commit change
            g_pendingState = -1;
            g_pendingCount = 0;
            return newState;
        }
    } else {
        // 新的待确认状态 / New pending state
        g_pendingState = newState;
        g_pendingCount = 1;
    }

    // 尚未确认，保持当前输出 / Not yet confirmed, keep current output
    return g_outputState;
}
