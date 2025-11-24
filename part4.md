节点预留（Node reservation）： 由于 SmolLM3 是在由 Slurm 管理的集群上训练的，我们为整个训练过程预订了固定的 48 个节点。这种设置让我们能够持续追踪同一批节点的健康与性能，也解决了前文提到的数据存储问题。我们还预留了一个备用节点（就像汽车的备胎），一旦某个节点故障，可立即替换，无需等待维修。空闲时，该备用节点会运行评估任务或开发实验。

持续监控（Continuous monitoring）： 训练期间，我们实时跟踪所有节点的关键指标，包括 GPU 温度、内存用量、计算利用率与吞吐波动。我们使用 [Prometheus](https://prometheus.io/) 收集所有 GPU 的 [DCGM](https://github.com/NVIDIA/DCGM) 指标，并在 [Grafana](https://grafana.com/) 仪表板中可视化，实现实时监控。如需在 AWS 基础设施上部署 Prometheus 与 Grafana 进行 GPU 监控的详细步骤，请参考[此示例设置指南](https://github.com/aws-samples/awsome-distributed-training/tree/3ae961d022399021cc4053c3ba19b182ca6b8dc8/4.validation_and_observability/4.prometheus-grafana)。Slack 机器人会在任何节点出现异常行为时发出告警，使我们能在硬件彻底崩溃前主动更换。

[访问仪表板](https://huggingfacetb-smol-training-playbook.hf.space/screencapture-grafana-huggingface.pdf) 这种多层策略让硬件问题变成了可控的中断。

热现实检验：当 GPU 降速时

营销规格假设完美散热，但现实更复杂。GPU 在过热时会自动降低时钟频率，即使系统设计良好，性能也会低于理论最大值。

![图 21：图片](https://huggingfacetb-smol-training-playbook.hf.space/_astro/image_27d1384e-bcac-80b1-9ffb-ec29d0021ccc.D54wWyJ9_2jmnNO.webp)

这个 Grafana 面板展示了我们整个 GPU 集群的热节流（thermal throttling）事件。底部面板中的条形图表示 GPU 因过热而自动降低时钟频率的时刻。

我们通过监控来自 [NVIDIA DCGM](https://github.com/NVIDIA/DCGM/tree/master) 的指标 `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` 来检测热节流。当该指标出现非零值时，说明 GPU 正因过热而自动降频。上面的面板展示了这些节流事件在实际运行中的表现。

热节流不仅会影响受影响的 GPU，还会在整个分布式训练环境中产生连锁反应。在我们的测试中，我们观察到单个发生节流的节点会显著拖慢集体通信（collective communication）性能。

在我们压力测试过程中，跨节点的 AllReduce 带宽出现下降。当节点数超过 14 个后，带宽从 350 GB/s 骤降至 100 GB/s，其根本原因就是一台 GPU 发生了热节流，这说明了单个慢节点就能成为整个分布式训练管道的瓶颈。

上图展示了随着节点数从 1 扩展到 16，AllReduce 带宽的退化情况。注意在 14 个节点之后出现的急剧下降：带宽从 350 GB/s 掉到 100 GB/s，而我们原本预期带宽应保持在 300 GB/s 以上（此前已观测到）。这并不是网络问题：单个发生热节流的节点成了瓶颈，在梯度同步阶段迫使所有其他节点等待。在分布式训练中，整体速度取决于最慢的那个节点。

👉 关键教训： 在启动长时间训练之前，务必先用前文提到的工具对硬件进行压力测试，以发现散热和供电瓶颈。训练过程中应持续使用 DCGM 遥测监控温度，并为实际的热设计极限做好预案。同时，建议确认 GPU 时钟已锁定在最高性能档位。若想深入了解为何 GPU 会因功耗限制而无法持续达到标称性能，请参阅这篇关于功耗降频（power throttling）的[精彩分析](https://www.thonking.ai/p/strangely-matrix-multiplications)。

#### [Checkpoint Management（检查点管理）](https://huggingfacetb-smol-training-playbook.hf.space/#checkpoint-management)

检查点（checkpoint）是我们在长时间训练过程中的安全网。我们定期保存它们，出于三个实际原因：从故障中恢复、通过评估监控训练进度，以及与社区共享中间模型以供研究。恢复方面最为重要。如果我们的运行失败，我们希望从最新保存的检查点重新启动，这样如果我们立即恢复，最多只会丢失保存间隔的时间（例如，如果我们每 4 小时保存一次，则最多丢失 4 小时的训练）。

尽量自动化你的恢复过程。例如，在 Slurm 上，你可以使用 `SBATCH --requeue`，这样作业会从最新的检查点自动重启。这样，你可以避免浪费时间等待有人注意到故障并手动重启。

在实现恢复机制时，有两个重要细节需要牢记：

*   检查点保存应在后台进行，不影响训练吞吐量（throughput）。
*   注意你的存储空间，在一个 24 天的运行中，每 4 小时保存一次意味着大约 144 个检查点。对于大型模型和优化器状态（optimizer states），这会迅速累积。在我们的案例中，我们一次只存储一个本地检查点（最新保存的），其余的卸载到 S3，以避免填满集群存储。

过去的一个惨痛教训：

在我们第一次大规模运行（StarCoder 15B）期间，训练在多次重启中顺利进行。在最后一天，我们发现整个检查点文件夹被脚本末尾遗留的 `rm -rf $CHECKPOINT_PATH` 命令删除了，这个命令来自旧的吞吐量测试。这个破坏性命令只有在 Slurm 作业真正完成时才会触发，而之前的重启中作业从未真正完成过。

幸运的是，我们保存了前一天的 checkpoint（检查点），因此只损失了一天的重训时间。教训很明确：永远不要把破坏性命令留在生产脚本中，并且在保存后立即自动化 checkpoint 备份，而不是依赖人工干预。

在我们的 nanotron 训练中，我们每 2 小时在本地保存一次 checkpoint，随后立即将其上传到 S3，一旦备份确认就删除本地副本。恢复时，如果最新的 checkpoint 在本地不可用，就从 S3 拉取。这种方法既节省存储，又确保备份，还能实现快速恢复。

#### [自动化评估](https://huggingfacetb-smol-training-playbook.hf.space/#automated-evaluations)

手动运行评估很快就会成为瓶颈。看起来简单，但一旦需要反复执行，跑基准、追踪并绘制每次实验的结果，开销就会迅速累积。解决之道？一开始就全部自动化。

对于 SmolLM3，我们使用 [LightEval](https://github.com/huggingface/lighteval) 在 nanotron 检查点上运行评估。每保存一个检查点，集群就会自动触发一次评估任务。结果直接推送到 Weights & Biases 或 [Trackio](https://github.com/gradio-app/trackio)，我们只需打开仪表板，就能看到曲线实时变化。这为我们节省了大量时间，并确保整个训练过程中评估追踪的一致性。

如果你的训练流程只能自动化一件事，那就把评估自动化。

最后，让我们看看如何优化训练布局（training layout），也就是模型在可用 GPU 上的分布方式，以最大化吞吐量。

### [优化训练吞吐量](https://huggingfacetb-smol-training-playbook.hf.space/#optimizing-training-throughput)

#### [我们需要多少块 GPU？](https://huggingfacetb-smol-training-playbook.hf.space/#how-many-gpus-do-we-need)

好问题！聊了这么多规格和基准，你还得解决一个实际问题：到底该租或买多少块 GPU？

确定合适的 GPU 数量需要在训练时间、成本和扩展效率之间取得平衡。以下是我们采用的框架：

基础规模估算公式：

GPU 数量 = 总 FLOPs 需求 / (单 GPU 吞吐量 × 目标训练时间)

这个公式把问题拆成三个关键部分：

*   总 FLOPs 需求（Total FLOPs Required）：训练模型所需的计算量（取决于模型大小、训练 token 数和架构）
*   单 GPU 吞吐量（Per-GPU Throughput）：每块 GPU 实际能提供的 FLOPs/s（不是理论峰值！）
*   目标训练时间（Target Training Time）：你愿意等待训练完成的时间

关键洞察：你需要估算实际吞吐量（realistic throughput），而非峰值规格。这意味着要考虑 Model FLOPs Utilization（MFU，模型 FLOPs 利用率）：你在实践中能达到的理论峰值性能百分比。

对于 SmolLM3，我们的计算如下：

*   模型大小：30 亿参数（3B parameters）
*   训练 token 数：11 万亿 token
*   目标训练时间：约 4 周
*   预期 MFU：30%（基于同规模实验）

首先，用标准 transformer 近似——每 token 6N FLOPs（N = 参数数）——估算总 FLOPs 需求：

总 FLOPs = 6 × 3×10⁹ 参数 × 11×10¹² token = 1.98×10²³ FLOPs

在 30% 的预期 MFU 下，每块 GPU 的有效吞吐量变为：

有效吞吐量 = 720×10¹² FLOPs/sec × 0.30 = 216×10¹² FLOPs/sec  
$$\text{Effective Throughput} = 720 \times 10^{12} \text{ FLOPs/sec} \times 0.30 = 216 \times 10^{12} \text{ FLOPs/sec}$$

现在代入我们的规模估算公式：

GPU 数量 = 1.98×10²³ FLOPs / (216×10¹² FLOPs/sec × 4 weeks × 604,800 sec/week)  
= 1.98×10²³ / 5.23×10²⁰ ≈ 379 GPUs  
$$\text{GPU Count} = \frac{1.98 \times 10^{23} \text{ FLOPs}}{216 \times 10^{12} \text{ FLOPs/sec} \times 4 \text{ weeks} \times 604,800 \text{ sec/week}}  
= \frac{1.98 \times 10^{23}}{5.23 \times 10^{20}} \approx 379 \text{ GPUs}$$

这一计算指向 375–400 张 H100，我们最终拿到了 384 张 H100，这个数字与我们的并行策略非常契合，并给出了一个现实可行的 4 周时间表，同时为节点故障和重启等意外情况留出了缓冲。

---

为什么更多 GPU 并不总是更好：阿姆达尔定律（Amdahl’s Law）在起作用

这里有一个反直觉的事实：增加 GPU 实际上可能让你的训练变慢。这就是 [阿姆达尔定律](https://en.wikipedia.org/wiki/Amdahl%27s_law) 登场的地方。

阿姆达尔定律指出，并行化带来的加速从根本上受限于工作负载中串行（不可并行）部分的比例。在 LLM 训练中，这部分“串行”主要是通信开销：在 GPU 之间同步梯度/权重/激活所花费的时间，这部分无法通过并行化消除（更多阅读见[此处](https://acenet-arc.github.io/ACENET_Summer_School_General/05-performance/index.html)）。

公式为： 

$$\text{最大加速比} = \frac{1}{\text{串行比例} + \frac{\text{并行比例}}{\text{处理器数量}}}$$

对于 SmolLM3 的 3B 模型，如果通信占用每一步 10 % 的时间，那么无论你增加多少 GPU，都无法获得超过 10 倍的加速。更糟的是，随着 GPU 数量增加，通信占比往往还会上升，因为：

* 更多 GPU = 更多 AllReduce 参与者 = 更长的同步时间  
* 网络延迟/带宽成为瓶颈  
* 小模型无法把通信隐藏在计算背后

对于 SmolLM3，我们采用了弱扩展（weak scaling）原则：全局批次大小（global batch size）随 GPU 数量线性扩展，保持每块 GPU 约 8K 个 token。这样既能维持通信与计算的合理比例，又能最大化吞吐量。

#### [寻找最优并行配置](https://huggingfacetb-smol-training-playbook.hf.space/#finding-the-optimal-parallelism-configuration)

一旦你搞定了 GPU，下一个挑战就是把它们配置得能真正高效训练。此时，并行策略（parallelism strategy）就成了关键。

我们借鉴了 [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=step_1%3A_fitting_a_training_step_in_memory)[寻找最优训练配置的方法](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=step_1%3A_fitting_a_training_step_in_memory)。该手册把问题拆成三步：先确保模型能放进内存，再达到目标批次大小（batch size），最后最大化吞吐。下面看看我们如何把这套流程用在 SmolLM3 上。

#### [第 1 步：让训练步装进内存](https://huggingfacetb-smol-training-playbook.hf.space/#step-1-fitting-a-training-step-in-memory)

第一个问题很简单：我们的 SmolLM3 3B 模型能不能塞进单张 H100 的 80 GB 内存？为了回答它，我们使用 [nanotron 的](https://huggingface.co/spaces/nanotron/predict_memory)[`predict_memory`](https://huggingface.co/spaces/nanotron/predict_memory)[工具](https://huggingface.co/spaces/nanotron/predict_memory)，它能估算模型参数、优化器状态（optimizer states）、梯度（gradients）和激活值（activations）的内存占用。

结果显示我们已逼近 80 GB 上限。这意味着必须采用某种并行方式，降低每张 GPU 的内存占用——无论是 Tensor Parallelism（张量并行，把模型层拆到多张 GPU）、Pipeline Parallelism（流水线并行，把模型深度拆到多张 GPU），还是 ZeRO 优化器分片（ZeRO optimizer sharding，把优化器状态分布出去）。如果不用其中至少一种策略，我们就无法高效训练，甚至根本无法训练。

#### [步骤 2：达到目标全局批次大小](https://huggingfacetb-smol-training-playbook.hf.space/#step-2-achieving-the-target-global-batch-size)

既然我们已经通过某种形式的并行（parallelism）确认模型可以放进显存，接下来就要确定如何把全局批次大小（Global Batch Size，GBS）做到约 200 万个 token。这一约束给出了第一个等式：

$$\text{GBS} = \text{DP} \times \text{MBS} \times \text{GRAD\_ACC} \times \text{SEQLEN} \approx 2\text{M tokens}$$

其中：

*   DP（Data Parallelism，数据并行）：数据并行副本的数量  
*   MBS（Micro Batch Size，微批次大小）：每个 GPU 在每个微批次中处理的 token 数  
*   GRAD_ACC（Gradient Accumulation，梯度累积）：在优化器更新前执行的 forward-backward 次数  
*   SEQLEN（Sequence Length，序列长度）：每条序列的 token 数（第一阶段预训练为 4096）

我们还受到 384 张 H100 的硬件约束：

$$\text{DP} \times \text{TP} \times \text{PP} = 384 = 2^7 \times 3$$

其中：

*   TP（Tensor Parallelism，张量并行）：每个模型层所用的 GPU 数（拆分权重矩阵）  
*   PP（Pipeline Parallelism，流水线并行）：模型深度方向上的 GPU 数（纵向拆分层）

这两个等式共同定义了我们的搜索空间。我们需要在满足双重约束的同时，找到能最大化训练吞吐量的取值。

#### [步骤 3：优化训练吞吐量](https://huggingfacetb-smol-training-playbook.hf.space/#step-3-optimizing-training-throughput)

在确定了约束条件后，我们需要找到能够最大化训练吞吐量的并行配置。搜索空间由我们的硬件拓扑和模型架构共同定义。

如上一节所述，我们的硬件环境提供两种截然不同的互连方式：用于节点内通信的 NVLink（900 GB/s）和用于节点间通信的 EFA（~50 GB/s）。这种拓扑天然提示我们至少应采用两种并行形式，以匹配网络特性。这两种互连带宽的巨大差异将极大影响哪些并行策略表现最佳。

从模型角度看，SmolLM3 的架构限制了可选方案。由于我们未采用混合专家（Mixture-of-Experts，MoE）架构，因此无需专家并行（Expert Parallelism）。同样，第一阶段以 4096 的序列长度训练，也意味着无需上下文并行（Context Parallelism）。这给我们留下了三个主要并行维度可供探索：数据并行（Data Parallelism，DP）、张量并行（Tensor Parallelism，TP）和流水线并行（Pipeline Parallelism，PP）。

鉴于步骤 2 的约束，我们需要在以下参数范围内进行搜索：

*   带 ZeRO 变体的 DP（ZeRO-0、ZeRO-1、ZeRO-3）：取值 1 到 384，且需为 2 和/或 3 的倍数  
*   TP（1、2、3、4、6、8）：限制在单节点内，以充分利用 NVLink 的高带宽  
*   PP（1..48）：将模型深度拆分到多张 GPU  
*   MBS（2、3、4、5）：根据并行带来的内存节省，可增大 MBS 以更好地利用 Tensor Core  
*   激活检查点（Activation checkpointing）（无、选择性、完整）：用额外计算换取内存与通信的减少  
*   内核优化（Kernel optimizations）：在可用处启用 CUDA Graph 与优化内核

尽管组合数量看似庞大，一个实用的做法是先独立测试每个维度，然后剔除那些明显拖慢吞吐量的配置。关键洞见在于：并非所有并行策略都生而平等。有些策略引入的通信开销远超其收益，尤其在我们这种规模下。

在我们的实验中，Pipeline Parallelism（流水线并行，PP） 表现出较差的性能特征。PP 需要在节点间频繁进行 pipeline bubble（流水线气泡）同步，而对我们仅有 3B 参数的较小模型而言，通信开销盖过了任何潜在收益。此外，我们也没有拿到能彻底消除流水线气泡的高效 PP 调度方案，这进一步削弱了 PP 的可行性。同样，ZeRO 等级高于 0 时会引入大量 all-gather 与 reduce-scatter 操作，对吞吐量的损害超过了其在内存上的帮助。这些早期基准测试让我们大幅缩小了搜索空间，专注于将 Data Parallelism（数据并行，DP） 与适度的 Tensor Parallelism（张量并行，TP） 相结合的配置。

👉 为评估每种配置，我们运行 5 次迭代基准测试，并记录 tokens per second per GPU (tok/s/gpu)——这最终是我们关心的指标。我们使用 Weights & Biases 和 Trackio 来记录吞吐量与配置，方便比较不同并行策略。

在系统地测试了 nanotron 中的可用选项后，我们最终确定 DP = 192，利用节点间 EFA 带宽进行数据并行梯度同步。这意味着 192 个独立的模型副本，各自处理不同的数据批次。对于张量并行，我们选择 TP = 2，将张量并行通信限制在单节点内，以充分利用 NVLink 的高带宽。这样每层权重矩阵被拆分到两块 GPU 上，在前后向传播时需要高速通信。

我们的 Micro Batch Size = 3（微批次大小 = 3） 在内存占用与计算效率之间取得了平衡。更大的批次规模能更好地利用 Tensor Cores（张量核心），但我们已接近内存上限。最终，我们选择了 ZeRO-0，即不对优化器状态做分片。虽然 ZeRO-1 或 ZeRO-3 可以进一步降低内存占用，但在 384 块 GPU 上跨节点收集与分发优化器状态所带来的通信开销，会显著拖慢整体吞吐。

该配置将全局批次规模控制在约 200 万 token（192 × 3 × 1 × 4096 ≈ 2.3M），同时在我们 384 张 H100 集群上实现了最大吞吐。完整训练配置见 [stage1_8T.yaml](https://github.com/huggingface/smollm/blob/main/text/pretraining/smollm3/stage1_8T.yaml)。

[Conclusion](https://huggingfacetb-smol-training-playbook.hf.space/#conclusion)
-------------------------------------------------------------------------------

我们最初只问了一个简单的问题：到 2025 年，训练一台高性能 LLM（大语言模型）到底需要什么？在走完从预训练到后训练的完整流程后，我们不仅展示了具体技术，更分享了让这些技术真正落地的整套方法论。

Pretraining at scale（规模化预训练）。 我们介绍了 Training Compass（训练罗盘）框架，用来判断“到底要不要训练”；随后演示了如何把目标转化为具体的架构决策。你看到了如何搭建可靠的消融实验管线、如何单独验证每项改动，以及如何从数十亿 token 的小实验平滑扩展到数万亿 token 的大运行。我们记录了规模化时可能遇到的基础设施难题（吞吐骤降、数据加载瓶颈、隐蔽 bug），并展示了如何通过监控与系统化降风险手段尽早发现、快速定位。

实践中的后训练（Post-training）。 我们展示了，从基础模型（base model）到生产级助手需要一套系统化的方法：在训练任何内容之前先建立评估（evals），迭代监督微调（SFT）数据配比，应用偏好优化（preference optimization），并可选择进一步通过强化学习（RL）推进。你已经看到，氛围测试（vibe testing）如何捕捉到指标遗漏的 bug，聊天模板（chat templates）如何悄无声息地破坏指令遵循，以及数据配比平衡在后训练阶段的重要性为何与预训练阶段不相上下。

在整个两个阶段中，我们不断回到相同的核心洞见：通过实验验证一切，一次只改变一件事，预期规模会在新场景下引发问题，并让使用场景驱动决策，而不是盲目追逐每一篇新论文。遵循这一流程，我们训练出了 SmolLM3：一个具备竞争力的 3B 多语言推理模型，支持长上下文。在此过程中，我们深入了解了哪些方法有效、哪些会失败，以及出错时如何调试。我们已尽力记录全部经验，无论成功还是失败。

下一步？

本篇博客涵盖了现代大语言模型（LLM）训练的基础知识，但该领域发展迅速。以下是深入探索的途径：

*   亲自运行实验。 阅读消融实验（ablations）固然有用；亲自运行则能让你真正了解哪些因素至关重要。选一个小模型，建立评估，开始实验。
*   阅读源代码。 nanotron、TRL 等训练框架均为开源。深入其实现可揭示论文中常被忽略的细节。
*   关注最新研究。 近期最先进模型的论文展示了领域的发展方向。参考文献部分收录了我们精选的有影响力论文与资源清单。

我们希望本篇博客能帮助你在下一次训练项目中保持清晰与自信，无论你是在大型实验室推动前沿，还是小团队解决特定问题。

现在去训练点什么吧。当你的损失（loss）在凌晨两点神秘飙升时，请记住：每一个伟大的模型背后都有一堆调试故事。愿开源（open source）与开放科学（open science）的力量永远与你同在！

#### [致谢](https://huggingfacetb-smol-training-playbook.hf.space/#acknowledgments)

我们感谢 [Guilherme](https://huggingface.co/guipenedo)、[Hugo](https://huggingface.co/hlarcher) 和 [Mario](https://huggingface.co/mariolr) 提供的宝贵反馈，以及 [Abubakar](https://huggingface.co/abidlabs) 在 Trackio 功能方面给予的帮助。

[参考文献](https://huggingfacetb-smol-training-playbook.hf.space/#references)
-------------------------------------------------------------------------------

以下是我们精心整理的论文、书籍和博客文章列表，它们在我们的 LLM（大语言模型）训练之旅中给予了我们最大的启发。

#### [LLM 架构](https://huggingfacetb-smol-training-playbook.hf.space/#llm-architecture)

*   稠密模型（Dense models）：[Llama3](https://huggingface.co/papers/2407.21783)、[Olmo2](https://huggingface.co/papers/2501.00656)、[MobileLLM](https://huggingface.co/papers/2402.14905)
*   MoE（混合专家模型，Mixture of Experts）：[DeepSeek V2](https://huggingface.co/papers/2405.04434)、[DeepSeek V3](https://huggingface.co/papers/2412.19437)、[Scaling Laws of Efficient MoEs](https://huggingface.co/papers/2507.17702)
*   混合架构（Hybrid）：[MiniMax-01](https://huggingface.co/papers/2501.08313)、[Mamba2](https://huggingface.co/papers/2405.21060)

#### [优化器与训练参数](https://huggingfacetb-smol-training-playbook.hf.space/#optimisers--training-parameters)

*   [Muon is Scalable for LLM Training](https://huggingface.co/papers/2502.16982)、[Fantastic pretraining optimisers](https://huggingface.co/papers/2509.02046)
*   [Large Batch Training](https://arxiv.org/abs/1812.06162)、[DeepSeekLLM](https://arxiv.org/abs/2401.02954)

#### [数据整理（Data curation）](https://huggingfacetb-smol-training-playbook.hf.space/#data-curation)

*   网页： [FineWeb & FineWeb-Edu](https://huggingface.co/papers/2406.17557)、[FineWeb2](https://huggingface.co/papers/2506.20920)、[DCLM](https://huggingface.co/papers/2406.11794)
*   代码： [The Stack v2](https://huggingface.co/papers/2402.19173)、[To Code or Not to Code](https://huggingface.co/papers/2408.10914)
*   数学： [DeepSeekMath](https://huggingface.co/papers/2402.03300)、[FineMath](https://huggingface.co/papers/2502.02737)、[MegaMath](https://huggingface.co/papers/2504.02807)
*   数据混合： [SmolLM2](https://huggingface.co/papers/2502.02737)、[Does your data spark joy](https://huggingface.co/papers/2406.03476)

#### [扩展定律（Scaling laws）](https://huggingfacetb-smol-training-playbook.hf.space/#scaling-laws)

*   [Kaplan](https://huggingface.co/papers/2001.08361)、[Chinchilla](https://huggingface.co/papers/2203.15556)、[Scaling Data-Constrained Language Models](https://huggingface.co/papers/2305.16264)

#### [后训练（Post-training）](https://huggingfacetb-smol-training-playbook.hf.space/#post-training)

*   [InstructGPT:](https://huggingface.co/papers/2203.02155) OpenAI 的开山之作，将基础模型转化为有用助手。ChatGPT 的前身，也是人类攀登卡尔达舍夫（Kardashev）等级之路上的关键一步。
*   [Llama 2](https://huggingface.co/papers/2307.09288) 与 [3](https://huggingface.co/papers/2407.21783)：Meta 发布的极其详尽的技术报告，揭秘 Llama 模型背后的训练细节（愿它们安息）。两篇报告都包含大量关于人类数据收集的洞见，涵盖人类偏好与模型评估。
*   Secrets of RLHF in LLMs，[第一部分](https://huggingface.co/papers/2307.04964) 与 [第二部分](https://huggingface.co/papers/2401.06080)：这两篇论文满满都是 RLHF（Reinforcement Learning from Human Feedback，人类反馈强化学习）的实操细节，尤其是如何训练强大的奖励模型。
*   [Direct Preference Optimisation:](https://huggingface.co/papers/2305.18290) 2023 年的突破性论文，让所有人不再对 LLM（Large Language Model，大语言模型）做 RL（Reinforcement Learning，强化学习）。
*   [DeepSeek-R1:](https://huggingface.co/papers/2501.12948) 2025 年的突破性论文，又让所有人重新开始对 LLM 做 RL。
*   [Dr. GRPO:](https://huggingface.co/papers/2503.20783) 理解 GRPO（Group Relative Policy Optimization，群组相对策略优化）内在偏差及其修正方法的最重要论文之一。
*   [DAPO:](https://huggingface.co/papers/2503.14476) 字节跳动分享大量实现细节，为社区解锁稳定的 R1-Zero 式训练。
*   [ScaleRL:](https://huggingface.co/papers/2510.13786) Meta 的“肌肉秀”，推导出 RL 的扩展定律（scaling laws）。烧掉 40 万 GPU 小时，确立一套在多个数量级算力上都可可靠扩展的训练配方。
*   [LoRA without Regret:](https://thinkingmachines.ai/blog/lora/) 一篇文笔优美的博客，发现低秩 LoRA（Low-Rank Adaptation，低秩适配）也能在 RL 中媲美全参数微调（最令人惊喜的结果）。
*   [Command A:](https://huggingface.co/papers/2504.00698) Cohere 发布的一份异常详尽的技术报告，介绍多种高效后训练 LLM 的策略。

#### [基础设施](https://huggingfacetb-smol-training-playbook.hf.space/#infrastructure)

*   [UltraScale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
*   [Jax scaling book](https://jax-ml.github.io/scaling-book/)
*   [Modal GPU Glossary](https://modal.com/gpu-glossary/readme)

#### [训练框架](https://huggingfacetb-smol-training-playbook.hf.space/#training-frameworks)

*   [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
*   [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
*   [Torchtitan](https://github.com/pytorch/torchtitan)
*   [Nanotron](https://github.com/huggingface/nanotron/)
*   [NanoChat](https://github.com/karpathy/nanochat)
*   [TRL](https://github.com/huggingface/trl)

#### [评估（Evaluation）](https://huggingfacetb-smol-training-playbook.hf.space/#evaluation)

*   [LLM 评估指南（The LLM Evaluation Guidebook）](https://github.com/huggingface/evaluation-guidebook)
*   [OLMES](https://huggingface.co/papers/2406.08446)
*   [FineTasks](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fine-tasks)
*   [来自一线的经验教训（Lessons from the trenches）](https://huggingface.co/papers/2405.14782)

[脚注（Footnotes）](https://huggingfacetb-smol-training-playbook.hf.space/#footnote-label)
----------------------------------------------------------------------------------

1.   计算这些统计量的想法来自 Llama 3 技术报告（[Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783)）。

[](https://huggingfacetb-smol-training-playbook.hf.space/#user-content-fnref-f1)
2.   关于 vLLM，参见：[推理解析器（Reasoning parsers）](https://docs.vllm.ai/en/v0.10.1.1/features/reasoning_outputs.html)、[工具解析器（Tool parsers）](https://huggingfacetb-smol-training-playbook.hf.space/2421384ebcac80fbaa7cf939fc39269d)。关于 SGLang，参见：[推理解析器（Reasoning parsers）](https://docs.sglang.ai/advanced_features/separate_reasoning.html)、[工具解析器（Tool parsers）](https://docs.sglang.ai/advanced_features/tool_parser.html)

[](https://huggingfacetb-smol-training-playbook.hf.space/#user-content-fnref-f2)
3.   Transformers 团队最近新增了解析器（[parsers](https://huggingface.co/docs/transformers/main/en/chat_response_parsing)），用于提取工具调用（tool calling）和推理输出（reasoning outputs）。如果像 vLLM 这样的引擎采用这些解析器，兼容性标准在未来可能变得不那么重要。

[](https://huggingfacetb-smol-training-playbook.hf.space/#user-content-fnref-f3)


1.   Agarwal, R., Vieillard, N., Zhou, Y., Stanczyk, P., Ramos, S., Geist, M., & Bachem, O. (2024). _On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes_. [https://arxiv.org/abs/2306.13649](https://arxiv.org/abs/2306.13649)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-gkd-1)
2.   Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). _GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints_. [https://arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-gqa-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-gqa-2)
3.   Allal, L. B., Lozhkov, A., Bakouch, E., Blázquez, G. M., Penedo, G., Tunstall, L., Marafioti, A., Kydlíček, H., Lajarín, A. P., Srivastav, V., Lochner, J., Fahlgren, C., Nguyen, X.-S., Fourrier, C., Burtenshaw, B., Larcher, H., Zhao, H., Zakka, C., Morlon, M., … Wolf, T. (2025). _SmolLM2: When Smol Goes Big – Data-Centric Training of a Small Language Model_. [https://arxiv.org/abs/2502.02737](https://arxiv.org/abs/2502.02737) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-smollm2-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-smollm2-2), [3](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-smollm2-3)
4.   Almazrouei, E., Alobeidli, H., Alshamsi, A., Cappelli, A., Cojocaru, R., Debbah, M., Goffinet, É., Hesslow, D., Launay, J., Malartic, Q., Mazzotta, D., Noune, B., Pannier, B., & Penedo, G. (2023). _The Falcon Series of Open Language Models_. [https://arxiv.org/abs/2311.16867](https://arxiv.org/abs/2311.16867)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-almazrouei2023falconseriesopenlanguage-1)
5.   An, C., Huang, F., Zhang, J., Gong, S., Qiu, X., Zhou, C., & Kong, L. (2024). _Training-Free Long-Context Scaling of Large Language Models_. [https://arxiv.org/abs/2402.17463](https://arxiv.org/abs/2402.17463)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-dca-1)
6.   Aryabumi, V., Su, Y., Ma, R., Morisot, A., Zhang, I., Locatelli, A., Fadaee, M., Üstün, A., & Hooker, S. (2024). _To Code, or Not To Code? Exploring Impact of Code in Pre-training_. [https://arxiv.org/abs/2408.10914](https://arxiv.org/abs/2408.10914)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-aryabumi2024codecodeexploringimpact-1)
7.   Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., Hui, B., Ji, L., Li, M., Lin, J., Lin, R., Liu, D., Liu, G., Lu, C., Lu, K., … Zhu, T. (2023). _Qwen Technical Report_. [https://arxiv.org/abs/2309.16609](https://arxiv.org/abs/2309.16609)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-qwen1-1)
8.   Barres, V., Dong, H., Ray, S., Si, X., & Narasimhan, K. (2025). _τ 2-Bench: Evaluating Conversational Agents in a Dual-Control Environment_. [https://arxiv.org/abs/2506.07982](https://arxiv.org/abs/2506.07982)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-taubench-1)
9.   Beck, M., Pöppel, K., Lippe, P., & Hochreiter, S. (2025). _Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels_. [https://arxiv.org/abs/2503.14376](https://arxiv.org/abs/2503.14376)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-beck2025tiledflashlinearattention-1)
10.   Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., … Amodei, D. (2020). _Language Models are Few-Shot Learners_. [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-gpt3-1)
11.   Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto, H. P., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., … Zaremba, W. (2021). _Evaluating Large Language Models Trained on Code_. [https://arxiv.org/abs/2107.03374](https://arxiv.org/abs/2107.03374)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-codex-1)
12.   Chen, Y., Huang, B., Gao, Y., Wang, Z., Yang, J., & Ji, H. (2025a). _Scaling Laws for Predicting Downstream Performance in LLMs_. [https://arxiv.org/abs/2410.08527](https://arxiv.org/abs/2410.08527)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-chen2025-1)
13.   Chen, Y., Huang, B., Gao, Y., Wang, Z., Yang, J., & Ji, H. (2025b). _Scaling Laws for Predicting Downstream Performance in LLMs_. [https://arxiv.org/abs/2410.08527](https://arxiv.org/abs/2410.08527)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-chen2025scalinglawspredictingdownstream-1)
14.   Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). Generating long sequences with sparse transformers. _arXiv Preprint arXiv:1904.10509_.[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-child2019generating-1)
15.   Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., Schuh, P., Shi, K., Tsvyashchenko, S., Maynez, J., Rao, A., Barnes, P., Tay, Y., Shazeer, N., Prabhakaran, V., … Fiedel, N. (2022). _PaLM: Scaling Language Modeling with Pathways_. [https://arxiv.org/abs/2204.02311](https://arxiv.org/abs/2204.02311) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-palm-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-palm-2), [3](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-palm-3), [4](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-palm-4)
16.   Chu, T., Zhai, Y., Yang, J., Tong, S., Xie, S., Schuurmans, D., Le, Q. V., Levine, S., & Ma, Y. (2025). _SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training_. [https://arxiv.org/abs/2501.17161](https://arxiv.org/abs/2501.17161)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-chu2025-1)
17.   Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021). _Training Verifiers to Solve Math Word Problems_. [https://arxiv.org/abs/2110.14168](https://arxiv.org/abs/2110.14168)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-gsm8k-1)
18.   Cohere, T., :, Aakanksha, Ahmadian, A., Ahmed, M., Alammar, J., Alizadeh, M., Alnumay, Y., Althammer, S., Arkhangorodsky, A., Aryabumi, V., Aumiller, D., Avalos, R., Aviv, Z., Bae, S., Baji, S., Barbet, A., Bartolo, M., Bebensee, B., … Zhao, Z. (2025). _Command A: An Enterprise-Ready Large Language Model_. [https://arxiv.org/abs/2504.00698](https://arxiv.org/abs/2504.00698)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-commandacohere-1)
19.   Dagan, G., Synnaeve, G., & Rozière, B. (2024). _Getting the most out of your tokenizer for pre-training and domain adaptation_. [https://arxiv.org/abs/2402.01035](https://arxiv.org/abs/2402.01035) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-dagan2024gettingtokenizerpretrainingdomain-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-dagan2024gettingtokenizerpretrainingdomain-2)
20.   Dao, T., & Gu, A. (2024). _Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality_. [https://arxiv.org/abs/2405.21060](https://arxiv.org/abs/2405.21060) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-mamba2-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-mamba2-2)
21.   DeepSeek-AI. (2025). _DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention_. DeepSeek. [https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-dsa-1)
22.   DeepSeek-AI, :, Bi, X., Chen, D., Chen, G., Chen, S., Dai, D., Deng, C., Ding, H., Dong, K., Du, Q., Fu, Z., Gao, H., Gao, K., Gao, W., Ge, R., Guan, K., Guo, D., Guo, J., … Zou, Y. (2024). _DeepSeek LLM: Scaling Open-Source Language Models with Longtermism_. [https://arxiv.org/abs/2401.02954](https://arxiv.org/abs/2401.02954) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-deepseekai2024deepseekllmscalingopensource-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-deepseekai2024deepseekllmscalingopensource-2)
23.   DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., Zhang, X., Yu, X., Wu, Y., Wu, Z. F., Gou, Z., Shao, Z., Li, Z., Gao, Z., … Zhang, Z. (2025). _DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning_. [https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-deepseekr1-1)
24.   DeepSeek-AI, Liu, A., Feng, B., Wang, B., Wang, B., Liu, B., Zhao, C., Dengr, C., Ruan, C., Dai, D., Guo, D., Yang, D., Chen, D., Ji, D., Li, E., Lin, F., Luo, F., Hao, G., Chen, G., … Xie, Z. (2024). _DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model_. [https://arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-deepseekv2-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-deepseekv2-2)
25.   DeepSeek-AI, Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Guo, D., Yang, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., … Pan, Z. (2025). _DeepSeek-V3 Technical Report_. [https://arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-deepseekv3-1)
26.   Dehghani, M., Djolonga, J., Mustafa, B., Padlewski, P., Heek, J., Gilmer, J., Steiner, A., Caron, M., Geirhos, R., Alabdulmohsin, I., Jenatton, R., Beyer, L., Tschannen, M., Arnab, A., Wang, X., Riquelme, C., Minderer, M., Puigcerver, J., Evci, U., … Houlsby, N. (2023). _Scaling Vision Transformers to 22 Billion Parameters_. [https://arxiv.org/abs/2302.05442](https://arxiv.org/abs/2302.05442)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-dehghani2023scalingvisiontransformers22-1)
27.   Ding, H., Wang, Z., Paolini, G., Kumar, V., Deoras, A., Roth, D., & Soatto, S. (2024). _Fewer Truncations Improve Language Modeling_. [https://arxiv.org/abs/2404.10830](https://arxiv.org/abs/2404.10830)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-bfd-1)
28.   D’Oosterlinck, K., Xu, W., Develder, C., Demeester, T., Singh, A., Potts, C., Kiela, D., & Mehri, S. (2024). _Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Alignment_. [https://arxiv.org/abs/2408.06266](https://arxiv.org/abs/2408.06266)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-apo-1)
29.   Du, Z., Zeng, A., Dong, Y., & Tang, J. (2025). _Understanding Emergent Abilities of Language Models from the Loss Perspective_. [https://arxiv.org/abs/2403.15796](https://arxiv.org/abs/2403.15796) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-du2025-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-du2025-2)
30.   Dubois, Y., Galambosi, B., Liang, P., & Hashimoto, T. B. (2025). _Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators_. [https://arxiv.org/abs/2404.04475](https://arxiv.org/abs/2404.04475)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-alpacaeval-1)
31.   Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., & Kiela, D. (2024). _KTO: Model Alignment as Prospect Theoretic Optimization_. [https://arxiv.org/abs/2402.01306](https://arxiv.org/abs/2402.01306)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-kto-1)
32.   Gandhi, K., Chakravarthy, A., Singh, A., Lile, N., & Goodman, N. D. (2025). _Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs_. [https://arxiv.org/abs/2503.01307](https://arxiv.org/abs/2503.01307)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-cognitivebehaviours-1)
33.   Gao, T., Wettig, A., Yen, H., & Chen, D. (2025). _How to Train Long-Context Language Models (Effectively)_. [https://arxiv.org/abs/2410.02660](https://arxiv.org/abs/2410.02660) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-prolong-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-prolong-2), [3](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-prolong-3)
34.   Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., Yang, A., Fan, A., Goyal, A., Hartshorn, A., Yang, A., Mitra, A., Sravankumar, A., Korenev, A., Hinsvark, A., … Ma, Z. (2024). _The Llama 3 Herd of Models_. [https://arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-llama3-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-llama3-2), [3](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-llama3-3)
35.   Gu, A., & Dao, T. (2024). _Mamba: Linear-Time Sequence Modeling with Selective State Spaces_. [https://arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-mamba-1)
36.   Gu, Y., Tafjord, O., Kuehl, B., Haddad, D., Dodge, J., & Hajishirzi, H. (2025). _OLMES: A Standard for Language Model Evaluations_. [https://arxiv.org/abs/2406.08446](https://arxiv.org/abs/2406.08446) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-olmes-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-olmes-2)
37.   Guo, S., Zhang, B., Liu, T., Liu, T., Khalman, M., Llinares, F., Rame, A., Mesnard, T., Zhao, Y., Piot, B., Ferret, J., & Blondel, M. (2024). _Direct Language Model Alignment from Online AI Feedback_. [https://arxiv.org/abs/2402.04792](https://arxiv.org/abs/2402.04792)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-onlinedpo-1)
38.   Hägele, A., Bakouch, E., Kosson, A., Allal, L. B., Werra, L. V., & Jaggi, M. (2024). _Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations_. [https://arxiv.org/abs/2405.18392](https://arxiv.org/abs/2405.18392) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-wsdhagele-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-wsdhagele-2)
39.   He, Y., Jin, D., Wang, C., Bi, C., Mandyam, K., Zhang, H., Zhu, C., Li, N., Xu, T., Lv, H., Bhosale, S., Zhu, C., Sankararaman, K. A., Helenowski, E., Kambadur, M., Tayade, A., Ma, H., Fang, H., & Wang, S. (2024). _Multi-IF: Benchmarking LLMs on Multi-Turn and Multilingual Instructions Following_. [https://arxiv.org/abs/2410.15553](https://arxiv.org/abs/2410.15553)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-multiif-1)
40.   Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., … Sifre, L. (2022). _Training Compute-Optimal Large Language Models_. [https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-hoffmann2022trainingcomputeoptimallargelanguage-1)
41.   Hong, J., Lee, N., & Thorne, J. (2024). _ORPO: Monolithic Preference Optimization without Reference Model_. [https://arxiv.org/abs/2403.07691](https://arxiv.org/abs/2403.07691)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-orpo-1)
42.   Howard, J., & Ruder, S. (2018). _Universal Language Model Fine-tuning for Text Classification_. [https://arxiv.org/abs/1801.06146](https://arxiv.org/abs/1801.06146)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-ulmfit-1)
43.   Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D., Jia, F., Zhang, Y., & Ginsburg, B. (2024). _RULER: What’s the Real Context Size of Your Long-Context Language Models?_[https://arxiv.org/abs/2404.06654](https://arxiv.org/abs/2404.06654) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-ruler-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-ruler-2)
44.   Hu, S., Tu, Y., Han, X., He, C., Cui, G., Long, X., Zheng, Z., Fang, Y., Huang, Y., Zhao, W., Zhang, X., Thai, Z. L., Zhang, K., Wang, C., Yao, Y., Zhao, C., Zhou, J., Cai, J., Zhai, Z., … Sun, M. (2024). _MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies_. [https://arxiv.org/abs/2404.06395](https://arxiv.org/abs/2404.06395)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-hu2024minicpmunveilingpotentialsmall-1)
45.   Huang, S., Noukhovitch, M., Hosseini, A., Rasul, K., Wang, W., & Tunstall, L. (2024). _The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization_. [https://arxiv.org/abs/2403.17031](https://arxiv.org/abs/2403.17031)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-ndetailsrlhf-1)
46.   IBM Research. (2025). _IBM Granite 4.0: Hyper-efficient, High Performance Hybrid Models for Enterprise_. [https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-granite4-1)
47.   Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., de las Casas, D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., Lavaud, L. R., Lachaux, M.-A., Stock, P., Scao, T. L., Lavril, T., Wang, T., Lacroix, T., & Sayed, W. E. (2023). _Mistral 7B_. [https://arxiv.org/abs/2310.06825](https://arxiv.org/abs/2310.06825)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-jiang2023mistral7b-1)
48.   Kamradt, G. (2023). Needle In A Haystack - pressure testing LLMs. In _GitHub repository_. GitHub. [https://github.com/gkamradt/LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-niah-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-niah-2)
49.   Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). _Scaling Laws for Neural Language Models_. [https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-kaplan2020scalinglawsneurallanguage-1)
50.   Katsch, T. (2024). _GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling_. [https://arxiv.org/abs/2311.01927](https://arxiv.org/abs/2311.01927)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-katsch2024gateloopfullydatacontrolledlinear-1)
51.   Kazemnejad, A., Padhi, I., Ramamurthy, K. N., Das, P., & Reddy, S. (2023). _The Impact of Positional Encoding on Length Generalization in Transformers_. [https://arxiv.org/abs/2305.19466](https://arxiv.org/abs/2305.19466)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-nope-1)
52.   Khatri, D., Madaan, L., Tiwari, R., Bansal, R., Duvvuri, S. S., Zaheer, M., Dhillon, I. S., Brandfonbrener, D., & Agarwal, R. (2025). _The Art of Scaling Reinforcement Learning Compute for LLMs_. [https://arxiv.org/abs/2510.13786](https://arxiv.org/abs/2510.13786) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-scalerl-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-scalerl-2)
53.   Kingma, D. P. (2014). Adam: A method for stochastic optimization. _arXiv Preprint arXiv:1412.6980_.[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-kingma2014adam-1)
54.   Krajewski, J., Ludziejewski, J., Adamczewski, K., Pióro, M., Krutul, M., Antoniak, S., Ciebiera, K., Król, K., Odrzygóźdź, T., Sankowski, P., Cygan, M., & Jaszczur, S. (2024). _Scaling Laws for Fine-Grained Mixture of Experts_. [https://arxiv.org/abs/2402.07871](https://arxiv.org/abs/2402.07871)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-krajewski2024scalinglawsfinegrainedmixture-1)
55.   Lambert, N., Castricato, L., von Werra, L., & Havrilla, A. (2022). Illustrating Reinforcement Learning from Human Feedback (RLHF). _Hugging Face Blog_.[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-rlhf-1)
56.   Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., Brahman, F., Miranda, L. J. V., Liu, A., Dziri, N., Lyu, S., Gu, Y., Malik, S., Graf, V., Hwang, J. D., Yang, J., Bras, R. L., Tafjord, O., Wilhelm, C., Soldaini, L., … Hajishirzi, H. (2025). _Tulu 3: Pushing Frontiers in Open Language Model Post-Training_. [https://arxiv.org/abs/2411.15124](https://arxiv.org/abs/2411.15124)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-tulu3-1)
57.   Lanchantin, J., Chen, A., Lan, J., Li, X., Saha, S., Wang, T., Xu, J., Yu, P., Yuan, W., Weston, J. E., Sukhbaatar, S., & Kulikov, I. (2025). _Bridging Offline and Online Reinforcement Learning for LLMs_. [https://arxiv.org/abs/2506.21495](https://arxiv.org/abs/2506.21495)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-online-offline-1)
58.   Li, J., Fang, A., Smyrnis, G., Ivgi, M., Jordan, M., Gadre, S., Bansal, H., Guha, E., Keh, S., Arora, K., Garg, S., Xin, R., Muennighoff, N., Heckel, R., Mercat, J., Chen, M., Gururangan, S., Wortsman, M., Albalak, A., … Shankar, V. (2025). _DataComp-LM: In search of the next generation of training sets for language models_. [https://arxiv.org/abs/2406.11794](https://arxiv.org/abs/2406.11794)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-datacomp-1)
59.   Li, Q., Cui, L., Zhao, X., Kong, L., & Bi, W. (2024). _GSM-Plus: A Comprehensive Benchmark for Evaluating the Robustness of LLMs as Mathematical Problem Solvers_. [https://arxiv.org/abs/2402.19255](https://arxiv.org/abs/2402.19255)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-gsmplus-1)
60.   Li, R., Allal, L. B., Zi, Y., Muennighoff, N., Kocetkov, D., Mou, C., Marone, M., Akiki, C., Li, J., Chim, J., Liu, Q., Zheltonozhskii, E., Zhuo, T. Y., Wang, T., Dehaene, O., Davaadorj, M., Lamy-Poirier, J., Monteiro, J., Shliazhko, O., … de Vries, H. (2023). _StarCoder: may the source be with you!_[https://arxiv.org/abs/2305.06161](https://arxiv.org/abs/2305.06161)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-starcoder-1)
61.   Li, T., Chiang, W.-L., Frick, E., Dunlap, L., Wu, T., Zhu, B., Gonzalez, J. E., & Stoica, I. (2024). _From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline_. [https://arxiv.org/abs/2406.11939](https://arxiv.org/abs/2406.11939)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-arenahard-1)
62.   Liang, W., Liu, T., Wright, L., Constable, W., Gu, A., Huang, C.-C., Zhang, I., Feng, W., Huang, H., Wang, J., Purandare, S., Nadathur, G., & Idreos, S. (2025). _TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training_. [https://arxiv.org/abs/2410.06511](https://arxiv.org/abs/2410.06511)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-torchtitan-1)
63.   Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., & Cobbe, K. (2023). _Let’s Verify Step by Step_. [https://arxiv.org/abs/2305.20050](https://arxiv.org/abs/2305.20050)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-openaiprm-1)
64.   Liu, H., Xie, S. M., Li, Z., & Ma, T. (2022). _Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models_. [https://arxiv.org/abs/2210.14199](https://arxiv.org/abs/2210.14199)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-liu2022-1)
65.   Liu, Q., Zheng, X., Muennighoff, N., Zeng, G., Dou, L., Pang, T., Jiang, J., & Lin, M. (2025). _RegMix: Data Mixture as Regression for Language Model Pre-training_. [https://arxiv.org/abs/2407.01492](https://arxiv.org/abs/2407.01492)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-liu2025regmixdatamixtureregression-1)
66.   Liu, Z., Zhao, C., Iandola, F., Lai, C., Tian, Y., Fedorov, I., Xiong, Y., Chang, E., Shi, Y., Krishnamoorthi, R., Lai, L., & Chandra, V. (2024). _MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases_. [https://arxiv.org/abs/2402.14905](https://arxiv.org/abs/2402.14905)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-mobilellm-1)
67.   Loshchilov, I., & Hutter, F. (2017). _SGDR: Stochastic Gradient Descent with Warm Restarts_. [https://arxiv.org/abs/1608.03983](https://arxiv.org/abs/1608.03983)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-loshchilov2017sgdrstochasticgradientdescent-1)
68.   Lozhkov, A., Li, R., Allal, L. B., Cassano, F., Lamy-Poirier, J., Tazi, N., Tang, A., Pykhtar, D., Liu, J., Wei, Y., Liu, T., Tian, M., Kocetkov, D., Zucker, A., Belkada, Y., Wang, Z., Liu, Q., Abulkhanov, D., Paul, I., … de Vries, H. (2024). _StarCoder 2 and The Stack v2: The Next Generation_. [https://arxiv.org/abs/2402.19173](https://arxiv.org/abs/2402.19173)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-starcoder2-1)
69.   Mao, H. H. (2022). _Fine-Tuning Pre-trained Transformers into Decaying Fast Weights_. [https://arxiv.org/abs/2210.04243](https://arxiv.org/abs/2210.04243)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-mao2022finetuningpretrainedtransformersdecaying-1)
70.   Marafioti, A., Zohar, O., Farré, M., Noyan, M., Bakouch, E., Cuenca, P., Zakka, C., Allal, L. B., Lozhkov, A., Tazi, N., Srivastav, V., Lochner, J., Larcher, H., Morlon, M., Tunstall, L., von Werra, L., & Wolf, T. (2025). _SmolVLM: Redefining small and efficient multimodal models_. [https://arxiv.org/abs/2504.05299](https://arxiv.org/abs/2504.05299)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-smolvlm-1)
71.   McCandlish, S., Kaplan, J., Amodei, D., & Team, O. D. (2018). _An Empirical Model of Large-Batch Training_. [https://arxiv.org/abs/1812.06162](https://arxiv.org/abs/1812.06162)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-mccandlish2018empiricalmodellargebatchtraining-1)
72.   Merrill, W., Arora, S., Groeneveld, D., & Hajishirzi, H. (2025). _Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training_. [https://arxiv.org/abs/2505.23971](https://arxiv.org/abs/2505.23971)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-merrill2025criticalbatchsizerevisited-1)
73.   Meta AI. (2025). _The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation_. [https://ai.meta.com/blog/llama-4-multimodal-intelligence/](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-llama4-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-llama4-2)
74.   Mindermann, S., Brauner, J., Razzak, M., Sharma, M., Kirsch, A., Xu, W., Höltgen, B., Gomez, A. N., Morisot, A., Farquhar, S., & Gal, Y. (2022). _Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt_. [https://arxiv.org/abs/2206.07137](https://arxiv.org/abs/2206.07137)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-mindermann2022prioritizedtrainingpointslearnable-1)
75.   MiniMax, Li, A., Gong, B., Yang, B., Shan, B., Liu, C., Zhu, C., Zhang, C., Guo, C., Chen, D., Li, D., Jiao, E., Li, G., Zhang, G., Sun, H., Dong, H., Zhu, J., Zhuang, J., Song, J., … Wu, Z. (2025). _MiniMax-01: Scaling Foundation Models with Lightning Attention_. [https://arxiv.org/abs/2501.08313](https://arxiv.org/abs/2501.08313) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-minimax01-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-minimax01-2), [3](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-minimax01-3)
76.   Mistral AI. (2025). _Mistral Small 3.1_. [https://mistral.ai/news/mistral-small-3-1](https://mistral.ai/news/mistral-small-3-1)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-mistralsmall-1)
77.   Moshkov, I., Hanley, D., Sorokin, I., Toshniwal, S., Henkel, C., Schifferer, B., Du, W., & Gitman, I. (2025). _AIMO-2 Winning Solution: Building State-of-the-Art Mathematical Reasoning Models with OpenMathReasoning dataset_. [https://arxiv.org/abs/2504.16891](https://arxiv.org/abs/2504.16891)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-moshkov2025aimo2winningsolutionbuilding-1)
78.   Muennighoff, N., Rush, A. M., Barak, B., Scao, T. L., Piktus, A., Tazi, N., Pyysalo, S., Wolf, T., & Raffel, C. (2025). _Scaling Data-Constrained Language Models_. [https://arxiv.org/abs/2305.16264](https://arxiv.org/abs/2305.16264)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-muennighoff2025scalingdataconstrainedlanguagemodels-1)
79.   Ni, J., Xue, F., Yue, X., Deng, Y., Shah, M., Jain, K., Neubig, G., & You, Y. (2024). _MixEval: Deriving Wisdom of the Crowd from LLM Benchmark Mixtures_. [https://arxiv.org/abs/2406.06565](https://arxiv.org/abs/2406.06565)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-mixeval-1)
80.   Nrusimha, A., Brandon, W., Mishra, M., Shen, Y., Panda, R., Ragan-Kelley, J., & Kim, Y. (2025). _FlashFormer: Whole-Model Kernels for Efficient Low-Batch Inference_. [https://arxiv.org/abs/2505.22758](https://arxiv.org/abs/2505.22758)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-nrusimha2025flashformerwholemodelkernelsefficient-1)
81.   Nvidia, :, Adler, B., Agarwal, N., Aithal, A., Anh, D. H., Bhattacharya, P., Brundyn, A., Casper, J., Catanzaro, B., Clay, S., Cohen, J., Das, S., Dattagupta, A., Delalleau, O., Derczynski, L., Dong, Y., Egert, D., Evans, E., … Zhu, C. (2024). _Nemotron-4 340B Technical Report_. [https://arxiv.org/abs/2406.11704](https://arxiv.org/abs/2406.11704)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-nvidia2024nemotron4340btechnicalreport-1)
82.   NVIDIA, :, Basant, A., Khairnar, A., Paithankar, A., Khattar, A., Renduchintala, A., Malte, A., Bercovich, A., Hazare, A., Rico, A., Ficek, A., Kondratenko, A., Shaposhnikov, A., Bukharin, A., Taghibakhshi, A., Barton, A., Mahabaleshwarkar, A. S., Shen, A., … Chen, Z. (2025). _NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model_. [https://arxiv.org/abs/2508.14444](https://arxiv.org/abs/2508.14444)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-nvidia2025nvidianemotronnano2-1)
83.   NVIDIA, :, Blakeman, A., Basant, A., Khattar, A., Renduchintala, A., Bercovich, A., Ficek, A., Bjorlin, A., Taghibakhshi, A., Deshmukh, A. S., Mahabaleshwarkar, A. S., Tao, A., Shors, A., Aithal, A., Poojary, A., Dattagupta, A., Buddharaju, B., Chen, B., … Chen, Z. (2025). _Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models_. [https://arxiv.org/abs/2504.03624](https://arxiv.org/abs/2504.03624)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-nemotronh-1)
84.   OLMo, T., Walsh, P., Soldaini, L., Groeneveld, D., Lo, K., Arora, S., Bhagia, A., Gu, Y., Huang, S., Jordan, M., Lambert, N., Schwenk, D., Tafjord, O., Anderson, T., Atkinson, D., Brahman, F., Clark, C., Dasigi, P., Dziri, N., … Hajishirzi, H. (2025). _2 OLMo 2 Furious_. [https://arxiv.org/abs/2501.00656](https://arxiv.org/abs/2501.00656) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-olmo2-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-olmo2-2), [3](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-olmo2-3)
85.   OpenAI, Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., Avila, R., Babuschkin, I., Balaji, S., Balcom, V., Baltescu, P., Bao, H., Bavarian, M., Belgum, J., … Zoph, B. (2024). _GPT-4 Technical Report_. [https://arxiv.org/abs/2303.08774](https://arxiv.org/abs/2303.08774)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-gpt4-1)
86.   Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., & Lowe, R. (2022). _Training language models to follow instructions with human feedback_. [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-instructgpt-1)
87.   Penedo, G., Kydlíček, H., allal, L. B., Lozhkov, A., Mitchell, M., Raffel, C., Werra, L. V., & Wolf, T. (2024). _The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale_. [https://arxiv.org/abs/2406.17557](https://arxiv.org/abs/2406.17557)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-fineweb-1)
88.   Penedo, G., Kydlíček, H., Sabolčec, V., Messmer, B., Foroutan, N., Kargaran, A. H., Raffel, C., Jaggi, M., Werra, L. V., & Wolf, T. (2025). _FineWeb2: One Pipeline to Scale Them All – Adapting Pre-Training Data Processing to Every Language_. [https://arxiv.org/abs/2506.20920](https://arxiv.org/abs/2506.20920) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-fineweb2-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-fineweb2-2)
89.   Peng, B., Goldstein, D., Anthony, Q., Albalak, A., Alcaide, E., Biderman, S., Cheah, E., Du, X., Ferdinan, T., Hou, H., Kazienko, P., GV, K. K., Kocoń, J., Koptyra, B., Krishna, S., Jr., R. M., Lin, J., Muennighoff, N., Obeid, F., … Zhu, R.-J. (2024). _Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence_. [https://arxiv.org/abs/2404.05892](https://arxiv.org/abs/2404.05892)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-peng2024eaglefinchrwkvmatrixvalued-1)
90.   Peng, B., Quesnelle, J., Fan, H., & Shippole, E. (2023). _YaRN: Efficient Context Window Extension of Large Language Models_. [https://arxiv.org/abs/2309.00071](https://arxiv.org/abs/2309.00071) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-yarn-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-yarn-2)
91.   Peng, H., Pappas, N., Yogatama, D., Schwartz, R., Smith, N. A., & Kong, L. (2021). _Random Feature Attention_. [https://arxiv.org/abs/2103.02143](https://arxiv.org/abs/2103.02143)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-peng2021randomfeatureattention-1)
92.   Petty, J., van Steenkiste, S., Dasgupta, I., Sha, F., Garrette, D., & Linzen, T. (2024). _The Impact of Depth on Compositional Generalization in Transformer Language Models_. [https://arxiv.org/abs/2310.19956](https://arxiv.org/abs/2310.19956) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-petty2024impactdepthcompositionalgeneralization-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-petty2024impactdepthcompositionalgeneralization-2)
93.   Polo, F. M., Weber, L., Choshen, L., Sun, Y., Xu, G., & Yurochkin, M. (2024). _tinyBenchmarks: evaluating LLMs with fewer examples_. [https://arxiv.org/abs/2402.14992](https://arxiv.org/abs/2402.14992)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-tinybenchmarks-1)
94.   Press, O., Smith, N. A., & Lewis, M. (2022). _Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation_. [https://arxiv.org/abs/2108.12409](https://arxiv.org/abs/2108.12409)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-alibi-1)
95.   Pyatkin, V., Malik, S., Graf, V., Ivison, H., Huang, S., Dasigi, P., Lambert, N., & Hajishirzi, H. (2025). _Generalizing Verifiable Instruction Following_. [https://arxiv.org/abs/2507.02833](https://arxiv.org/abs/2507.02833)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-ifbench-1)
96.   Qin, Z., Han, X., Sun, W., Li, D., Kong, L., Barnes, N., & Zhong, Y. (2022). _The Devil in Linear Transformer_. [https://arxiv.org/abs/2210.10340](https://arxiv.org/abs/2210.10340)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-qin2022devillineartransformer-1)
97.   Qin, Z., Yang, S., Sun, W., Shen, X., Li, D., Sun, W., & Zhong, Y. (2024). _HGRN2: Gated Linear RNNs with State Expansion_. [https://arxiv.org/abs/2404.07904](https://arxiv.org/abs/2404.07904)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-qin2024hgrn2gatedlinearrnns-1)
98.   Qiu, Z., Huang, Z., Zheng, B., Wen, K., Wang, Z., Men, R., Titov, I., Liu, D., Zhou, J., & Lin, J. (2025). _Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models_. [https://arxiv.org/abs/2501.11873](https://arxiv.org/abs/2501.11873)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-qiu2025demonsdetailimplementingload-1)
99.   Qwen Team. (2025). _Qwen3-Next: Towards Ultimate Training & Inference Efficiency_. Alibaba Cloud. [https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-qwen3next-1)
100.   Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., & others. (2019). Language models are unsupervised multitask learners. In _OpenAI blog_ (Vol. 1, p. 9).[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-gpt2-1)
101.   Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2024). _Direct Preference Optimization: Your Language Model is Secretly a Reward Model_. [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-dpo-1)
102.   Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., & Bowman, S. R. (2024). Gpqa: A graduate-level google-proof q&a benchmark. _First Conference on Language Modeling_.[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-gpqa-1)
103.   Rozière, B., Gehring, J., Gloeckle, F., Sootla, S., Gat, I., Tan, X. E., Adi, Y., Liu, J., Sauvestre, R., Remez, T., Rapin, J., Kozhevnikov, A., Evtimov, I., Bitton, J., Bhatt, M., Ferrer, C. C., Grattafiori, A., Xiong, W., Défossez, A., … Synnaeve, G. (2024). _Code Llama: Open Foundation Models for Code_. [https://arxiv.org/abs/2308.12950](https://arxiv.org/abs/2308.12950)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-rozi%C3%A8re2024codellamaopenfoundation-1)
104.   Sennrich, R., Haddow, B., & Birch, A. (2016). _Neural Machine Translation of Rare Words with Subword Units_. [https://arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-sennrich2016neuralmachinetranslationrare-1)
105.   Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., & Guo, D. (2024). _DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models_. [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-grpo-1)
106.   Shazeer, N. (2019). _Fast Transformer Decoding: One Write-Head is All You Need_. [https://arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-mqa-1)
107.   Shi, F., Suzgun, M., Freitag, M., Wang, X., Srivats, S., Vosoughi, S., Chung, H. W., Tay, Y., Ruder, S., Zhou, D., Das, D., & Wei, J. (2022). _Language Models are Multilingual Chain-of-Thought Reasoners_. [https://arxiv.org/abs/2210.03057](https://arxiv.org/abs/2210.03057)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-mgsm-1)
108.   Shukor, M., Aubakirova, D., Capuano, F., Kooijmans, P., Palma, S., Zouitine, A., Aractingi, M., Pascal, C., Russi, M., Marafioti, A., Alibert, S., Cord, M., Wolf, T., & Cadene, R. (2025). _SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics_. [https://arxiv.org/abs/2506.01844](https://arxiv.org/abs/2506.01844)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-smolvla-1)
109.   Singh, S., Romanou, A., Fourrier, C., Adelani, D. I., Ngui, J. G., Vila-Suero, D., Limkonchotiwat, P., Marchisio, K., Leong, W. Q., Susanto, Y., Ng, R., Longpre, S., Ko, W.-Y., Ruder, S., Smith, M., Bosselut, A., Oh, A., Martins, A. F. T., Choshen, L., … Hooker, S. (2025). _Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation_. [https://arxiv.org/abs/2412.03304](https://arxiv.org/abs/2412.03304)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-globalmmlu-1)
110.   Sirdeshmukh, V., Deshpande, K., Mols, J., Jin, L., Cardona, E.-Y., Lee, D., Kritz, J., Primack, W., Yue, S., & Xing, C. (2025). _MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs_. [https://arxiv.org/abs/2501.17399](https://arxiv.org/abs/2501.17399)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-multichallenge-1)
111.   Smith, L. N., & Topin, N. (2018). _Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates_. [https://arxiv.org/abs/1708.07120](https://arxiv.org/abs/1708.07120)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-smith2018superconvergencefasttrainingneural-1)
112.   Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2023). _RoFormer: Enhanced Transformer with Rotary Position Embedding_. [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-rope-1)
113.   Sun, Y., Dong, L., Zhu, Y., Huang, S., Wang, W., Ma, S., Zhang, Q., Wang, J., & Wei, F. (2024). _You Only Cache Once: Decoder-Decoder Architectures for Language Models_. [https://arxiv.org/abs/2405.05254](https://arxiv.org/abs/2405.05254)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-sun2024cacheoncedecoderdecoderarchitectures-1)
114.   Takase, S., Kiyono, S., Kobayashi, S., & Suzuki, J. (2025). _Spike No More: Stabilizing the Pre-training of Large Language Models_. [https://arxiv.org/abs/2312.16903](https://arxiv.org/abs/2312.16903)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-takase2025spikemorestabilizingpretraining-1)
115.   Team, 5, Zeng, A., Lv, X., Zheng, Q., Hou, Z., Chen, B., Xie, C., Wang, C., Yin, D., Zeng, H., Zhang, J., Wang, K., Zhong, L., Liu, M., Lu, R., Cao, S., Zhang, X., Huang, X., Wei, Y., … Tang, J. (2025). _GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models_. [https://arxiv.org/abs/2508.06471](https://arxiv.org/abs/2508.06471)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-glm45-1)
116.   team, F. C., Copet, J., Carbonneaux, Q., Cohen, G., Gehring, J., Kahn, J., Kossen, J., Kreuk, F., McMilin, E., Meyer, M., Wei, Y., Zhang, D., Zheng, K., Armengol-Estapé, J., Bashiri, P., Beck, M., Chambon, P., Charnalia, A., Cummins, C., … Synnaeve, G. (2025). _CWM: An Open-Weights LLM for Research on Code Generation with World Models_. [https://arxiv.org/abs/2510.02387](https://arxiv.org/abs/2510.02387)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-cwm-1)
117.   Team, G., Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., Matejovicova, T., Ramé, A., Rivière, M., Rouillard, L., Mesnard, T., Cideron, G., bastien Jean-Grill, Ramos, S., Yvinec, E., Casbon, M., Pot, E., Penchev, I., … Hussenot, L. (2025). _Gemma 3 Technical Report_. [https://arxiv.org/abs/2503.19786](https://arxiv.org/abs/2503.19786)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-gemma3-1)
118.   Team, K., Bai, Y., Bao, Y., Chen, G., Chen, J., Chen, N., Chen, R., Chen, Y., Chen, Y., Chen, Y., Chen, Z., Cui, J., Ding, H., Dong, M., Du, A., Du, C., Du, D., Du, Y., Fan, Y., … Zu, X. (2025). _Kimi K2: Open Agentic Intelligence_. [https://arxiv.org/abs/2507.20534](https://arxiv.org/abs/2507.20534) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-kimik2-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-kimik2-2), [3](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-kimik2-3)
119.   Team, L., Han, B., Tang, C., Liang, C., Zhang, D., Yuan, F., Zhu, F., Gao, J., Hu, J., Li, L., Li, M., Zhang, M., Jiang, P., Jiao, P., Zhao, Q., Yang, Q., Shen, W., Yang, X., Zhang, Y., … Zhou, J. (2025). _Every Attention Matters: An Efficient Hybrid Architecture for Long-Context Reasoning_. [https://arxiv.org/abs/2510.19338](https://arxiv.org/abs/2510.19338)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-lingteam2025attentionmattersefficienthybrid-1)
120.   Team, L., Zeng, B., Huang, C., Zhang, C., Tian, C., Chen, C., Jin, D., Yu, F., Zhu, F., Yuan, F., Wang, F., Wang, G., Zhai, G., Zhang, H., Li, H., Zhou, J., Liu, J., Fang, J., Ou, J., … He, Z. (2025). _Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs_. [https://arxiv.org/abs/2503.05139](https://arxiv.org/abs/2503.05139)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-ling15-1)
121.   Team, M., Xiao, C., Li, Y., Han, X., Bai, Y., Cai, J., Chen, H., Chen, W., Cong, X., Cui, G., Ding, N., Fan, S., Fang, Y., Fu, Z., Guan, W., Guan, Y., Guo, J., Han, Y., He, B., … Sun, M. (2025). _MiniCPM4: Ultra-Efficient LLMs on End Devices_. [https://arxiv.org/abs/2506.07900](https://arxiv.org/abs/2506.07900)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-minicpm4-1)
122.   Tian, C., Chen, K., Liu, J., Liu, Z., Zhang, Z., & Zhou, J. (2025). _Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models_. [https://arxiv.org/abs/2507.17702](https://arxiv.org/abs/2507.17702) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-antgroup-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-antgroup-2), [3](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-antgroup-3)
123.   Toshniwal, S., Moshkov, I., Narenthiran, S., Gitman, D., Jia, F., & Gitman, I. (2024). _OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset_. [https://arxiv.org/abs/2402.10176](https://arxiv.org/abs/2402.10176)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-toshniwal2024openmathinstruct118millionmath-1)
124.   Tunstall, L., Beeching, E., Lambert, N., Rajani, N., Rasul, K., Belkada, Y., Huang, S., von Werra, L., Fourrier, C., Habib, N., Sarrazin, N., Sanseviero, O., Rush, A. M., & Wolf, T. (2023). _Zephyr: Direct Distillation of LM Alignment_. [https://arxiv.org/abs/2310.16944](https://arxiv.org/abs/2310.16944)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-zephyr-1)
125.   Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2023). _Attention Is All You Need_. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-transformer-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-transformer-2), [3](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-transformer-3)
126.   Waleffe, R., Byeon, W., Riach, D., Norick, B., Korthikanti, V., Dao, T., Gu, A., Hatamizadeh, A., Singh, S., Narayanan, D., Kulshreshtha, G., Singh, V., Casper, J., Kautz, J., Shoeybi, M., & Catanzaro, B. (2024). _An Empirical Study of Mamba-based Language Models_. [https://arxiv.org/abs/2406.07887](https://arxiv.org/abs/2406.07887)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-waleffe2024empiricalstudymambabasedlanguage-1)
127.   Wang, B., & Komatsuzaki, A. (2021). _GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model_. [https://github.com/kingoflolz/mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-gptj-1)
128.   Wei, J., Karina, N., Chung, H. W., Jiao, Y. J., Papay, S., Glaese, A., Schulman, J., & Fedus, W. (2024). Measuring short-form factuality in large language models. _arXiv Preprint arXiv:2411.04368_.[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-simpleqa-1)
129.   Wen, K., Hall, D., Ma, T., & Liang, P. (2025). _Fantastic Pretraining Optimizers and Where to Find Them_. [https://arxiv.org/abs/2509.02046](https://arxiv.org/abs/2509.02046) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-wen2025fantasticpretrainingoptimizers-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-wen2025fantasticpretrainingoptimizers-2)
130.   Xie, S. M., Pham, H., Dong, X., Du, N., Liu, H., Lu, Y., Liang, P., Le, Q. V., Ma, T., & Yu, A. W. (2023). _DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining_. [https://arxiv.org/abs/2305.10429](https://arxiv.org/abs/2305.10429)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-xie2023doremioptimizingdatamixtures-1)
131.   Xiong, W., Liu, J., Molybog, I., Zhang, H., Bhargava, P., Hou, R., Martin, L., Rungta, R., Sankararaman, K. A., Oguz, B., Khabsa, M., Fang, H., Mehdad, Y., Narang, S., Malik, K., Fan, A., Bhosale, S., Edunov, S., Lewis, M., … Ma, H. (2023a). _Effective Long-Context Scaling of Foundation Models_. [https://arxiv.org/abs/2309.16039](https://arxiv.org/abs/2309.16039)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-xiong2023effectivelongcontextscalingfoundation-1)
132.   Xiong, W., Liu, J., Molybog, I., Zhang, H., Bhargava, P., Hou, R., Martin, L., Rungta, R., Sankararaman, K. A., Oguz, B., Khabsa, M., Fang, H., Mehdad, Y., Narang, S., Malik, K., Fan, A., Bhosale, S., Edunov, S., Lewis, M., … Ma, H. (2023b). _Effective Long-Context Scaling of Foundation Models_. [https://arxiv.org/abs/2309.16039](https://arxiv.org/abs/2309.16039)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-ropeabf-1)
133.   Xu, H., Peng, B., Awadalla, H., Chen, D., Chen, Y.-C., Gao, M., Kim, Y. J., Li, Y., Ren, L., Shen, Y., Wang, S., Xu, W., Gao, J., & Chen, W. (2025). _Phi-4-Mini-Reasoning: Exploring the Limits of Small Reasoning Language Models in Math_. [https://arxiv.org/abs/2504.21233](https://arxiv.org/abs/2504.21233)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-phi4reasoning-1)
134.   Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J., … Qiu, Z. (2025). _Qwen3 Technical Report_. [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-qwen3-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-qwen3-2), [3](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-qwen3-3)
135.   Yang, A., Yu, B., Li, C., Liu, D., Huang, F., Huang, H., Jiang, J., Tu, J., Zhang, J., Zhou, J., Lin, J., Dang, K., Yang, K., Yu, L., Li, M., Sun, M., Zhu, Q., Men, R., He, T., … Zhang, Z. (2025). _Qwen2.5-1M Technical Report_. [https://arxiv.org/abs/2501.15383](https://arxiv.org/abs/2501.15383) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-qwen1million-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-qwen1million-2)
136.   Yang, B., Venkitesh, B., Talupuru, D., Lin, H., Cairuz, D., Blunsom, P., & Locatelli, A. (2025). _Rope to Nope and Back Again: A New Hybrid Attention Strategy_. [https://arxiv.org/abs/2501.18795](https://arxiv.org/abs/2501.18795) back: [1](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-rnope-1), [2](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-rnope-2)
137.   Yang, G., & Hu, E. J. (2022). _Feature Learning in Infinite-Width Neural Networks_. [https://arxiv.org/abs/2011.14522](https://arxiv.org/abs/2011.14522)[](https://huggingfacetb-smol-training-playbook.hf.space/#refctx-bib-mup-1)
