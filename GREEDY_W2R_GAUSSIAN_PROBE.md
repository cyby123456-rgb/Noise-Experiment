# Greedy 错误回答的多位置单点加噪 W2R 实验

## 一句话定义

先生成一条完整的 greedy 错误回答，根据回答长度均匀选择若干 token 位置。每次实验只固定其中一个位置，只对这个 token 在指定层的 hidden state 加一次噪声，然后继续 greedy 生成，观察最终答案是否从错误翻转为正确。

这不是对 prompt 加噪，也不是在整条回答的每一步持续加噪。

## 实验过程

设 clean greedy 回答的 token 是：

\[
r_1,r_2,\ldots,r_L.
\]

1. 从完整 prompt 开始运行一次 greedy generation。
2. 用 verifier 检查最终回答，只保留错误回答。
3. 根据 clean 回答长度选定若干位置 \(t_1,\ldots,t_K\)。
4. 对每个位置 \(t\) 分别执行后续步骤；不同位置不会在同一条 rollout 中同时加噪。
5. 重放固定前缀 \(\text{prompt}+r_1+\cdots+r_t\)。
6. 在指定 decoder layer 取得 \(r_t\) 的 hidden state \(h_t\)。
7. 对每个 noise seed 独立采样：

   \[
   \epsilon_s\sim\mathcal N(0,\sigma^2I),
   \]

   并且只执行一次：

   \[
   h_t\leftarrow h_t+\epsilon_s.
   \]

8. 从 \(r_{t+1}\) 开始继续 greedy generation，得到新的完整回答并重新评分。

示意图：

```text
prompt → r1 → r2 → ... → rt → rt+1 → ... → answer
                         ↑
                 只在这里加一次噪声
```

回答前缀 \(r_1,\ldots,r_t\) 在所有试验中完全相同；噪声只能改变 \(r_{t+1}\) 及其后续生成。

## 如何选择位置

脚本支持三种互斥方式：

- `--response-position 1024`：使用从 1 开始计数的第 1024 个回答 token；
- `--response-position-fraction 0.5`：使用 clean 回答的中间位置。
- `--num-response-positions 10`：根据 clean 回答长度均匀选择 10 个内部位置。

对于长度为 \(L\) 的 clean 回答，10 个目标位置近似为：

\[
t_j=\operatorname{round}\left(\frac{jL}{11}\right),\qquad j=1,\ldots,10.
\]

也就是大约位于回答的 9%、18%、……、91%。这样覆盖前、中、后段，同时避免直接选择回答终点。短回答因取整产生重复时会自动去重，所以实际位置数可能少于 10。

启动器默认使用：

```bash
RESPONSE_POSITION=""
RESPONSE_POSITION_FRACTION=""
NUM_RESPONSE_POSITIONS=10
```

所以第一版默认对每道 clean greedy 错误回答选择约 10 个位置。每个位置都是一组独立实验，每条 noisy rollout 仍然只在一个位置加一次噪声。

## 为什么需要 clean replay

得到完整错误回答以后，脚本会在不加噪声的情况下重放到位置 \(t\)，再继续 greedy 生成。

重放得到的 token 序列必须和原始 clean greedy 回答完全相同。若不同，脚本立即报错，不会采集 noisy 数据。这保证后续实验确实从同一个回答状态开始。

## Seed 的含义

启动器默认：

```bash
NUM_NOISE_SEEDS=32
BASE_NOISE_SEED=20260827
```

实际 seed 是：

```text
20260827, 20260828, ..., 20260858
```

因此每个位置的 noisy 组使用 32 个不同噪声。10 个位置最多产生 \(10\times32=320\) 条 noisy rollout。decode 始终是 greedy；同一位置内部，试验之间唯一主动改变的变量是该位置上的噪声。

不同位置复用同一组 32 个 seed，所以相同 seed 在各位置对应同一个标准 Gaussian 基向量 (z_s)。默认 RMS 相对缩放时，各位置复用相同随机方向，但会根据该位置的 clean hidden RMS 调整绝对幅度。这形成 matched-position 对照，便于比较“同一个方向加在回答早期或晚期”的差异；每个位置内部的 32 个噪声仍然彼此不同。

不同噪声仍可能得到相同回答，这表示这些噪声没有改变后续 greedy token 路径，并不表示 seed 被重复使用。

## 噪声为什么默认使用 RMS 相对缩放

默认配置为：

```bash
NOISE_SCALE_MODE="relative_rms"
NOISE_STD=0.1
```

对每个固定 clean state，脚本采样 (z_s\sim\mathcal N(0,I))，然后使用：

\[
\epsilon_s=\alpha\,\operatorname{RMS}(h_t)\,z_s.
\]

因此 `NOISE_STD=0.1` 中的 0.1 是相对比例：噪声的每维 RMS 约为该 token hidden RMS 的 10%。这样跨回答早、中、晚 10 个位置比较时，不会因为不同位置的 hidden 绝对尺度不同而混淆结果。条件于某个固定 (h_t)，它仍然是各向同性 Gaussian。

兼容旧实验时可以设置 `NOISE_SCALE_MODE="absolute"`。这时 (epsilon_s\sim\mathcal N(0,\sigma^2I))，`NOISE_STD` 直接使用 hidden unit；但它不适合作为跨位置比较的默认主结果。

## Zero-noise placebo 对照

每个 response 位置在运行 noisy 组之前，还会运行 32 条匹配的 zero-noise rollout：

\[
\epsilon=0.
\]

它们使用和 noisy 组完全相同的：

- clean response prefix；
- 注入 hook 代码路径；
- batch size 和调用次数；
- greedy generation 参数；
- continuation token budget。

因此每道题默认还会产生 \(10\times32=320\) 条 zero-noise control rollout。这个对照不是为了估计一个应当非零的“自然翻转率”；在固定模型、固定前缀和 greedy decode 下，它的预期翻转率严格为 0。

只要任意 zero-noise rollout 的 token 序列发生变化，或者以下任一 hidden-state 差异不为 0，脚本就立即终止，不继续把 noisy 结果解释为噪声效应：

```text
clean_hidden_batch_max_abs_spread
clean_final_batch_max_abs_spread
zero_noise_max_pre_hidden_vs_clean_abs_diff
zero_noise_max_applied_abs
zero_noise_max_final_hidden_vs_clean_abs_diff
max_pre_noise_hidden_vs_clean_abs_diff
```

## 保存的数据

每个“题目 × response 位置”保存一个 `.pt` 文件，主要字段包括：

- `prompt_token_ids`；
- `clean_response_token_ids`；
- `fixed_response_prefix_token_ids`；
- `clean_hidden_state`：加噪前的 \(h_t\)；
- `standard_normal_noise`：由 seed 决定的 FP32 (z_s\sim\mathcal N(0,I))；
- `sampled_noise`：经过 RMS 或 absolute 缩放后的目标 FP32 Gaussian noise；
- `applied_noise`：考虑 bf16/fp16 舍入后实际作用的噪声；
- `clean_final_hidden_state`：\(h_t\) 不加噪时进入 LM head 前的状态；
- `noisy_final_hidden_state`：\(h_t+\epsilon_s\) 经过 suffix decoder 后进入 LM head 前的状态；
- `baseline_response`、`noisy_responses`；
- `baseline_score`、`noisy_scores`、`is_w2r`；
- `zero_noise_control`：control 数量、token 一致性和零翻转率；
- `response_position`、layer、seed 和一致性诊断。

每个 seed 对应的核心四元组为：

\[
(h_t,\epsilon_s,D(h_t),D(h_t+\epsilon_s)).
\]

分析实际模型收到的扰动时使用 `applied_noise`；检查标准高斯采样时使用 `standard_normal_noise`；检查缩放后的目标扰动时使用 `sampled_noise`。元数据同时保存 `clean_hidden_rms` 与 `effective_noise_std_hidden_units`。

## 运行

编辑配置后运行：

```bash
bash noise_experiments/probe/run_greedy_wrong_gaussian_probe.sh
```

Python 参数说明：

```bash
python noise_experiments/probe/run_greedy_wrong_gaussian_probe.py --help
```
