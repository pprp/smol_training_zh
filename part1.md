# Smol 训练手册：打造世界级 LLM 的秘诀 (中文翻译版)

URL 来源：https://huggingfacetb-smol-training-playbook.hf.space/

发布时间：2025 年 10 月 30 日

[toc]


[引言](https://huggingfacetb-smol-training-playbook.hf.space/#introduction)
-----------------------------------------------------------------------------------

如今，训练一个高性能的 LLM（Large Language Model，大语言模型）到底需要什么？

已发表的研究让这一切看起来轻而易举：巧妙的架构选择、精心策划的数据集，以及充足的算力。结果光鲜亮丽，消融实验（ablations）结构清晰、干净利落。每个决定在事后看来都显而易见。但这些论文只展示了“奏效”的部分，并带有一点玫瑰色的后见之明——它们没有记录凌晨 2 点调试数据加载器（dataloader）的抓狂、损失飙升（loss spikes）的惊魂，或是那个悄悄毁掉你整个训练的微妙张量并行（tensor parallelism） bug（后文详述！）。现实更加混乱、更加迭代，充斥着大量最终论文里永远不会出现的决策。

请与我们一起走进 [SmolLM3](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) 的训练幕后——这是一个 30 亿参数的多语言推理模型，在 11 万亿 token 上完成训练。这不是一篇普通的博客，而是一张由决策、发现与死胡同交织成的蛛网，我们将其逐条拆解，最终沉淀出打造世界级语言模型的深层洞见。

这也是我们模型训练长文系列的收官之作：我们已陆续探讨了如何大规模构建数据集（[FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)）、如何协调数千块 GPU 齐声合唱（[Ultra Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)），以及如何在每一步挑选最佳评测方案（[Evaluation Guidebook](https://github.com/huggingface/evaluation-guidebook)）。现在，我们将所有环节融会贯通，打造出一个强大的 AI 模型。我们将带你走完完整旅程——不仅分享最终奏效的配方，也揭露那些失败的尝试、基础设施崩溃的瞬间，以及左右每一次决策的调试过程。

这段故事堪比一部戏剧：你会看到，小规模消融实验（ablation）的喜人结果为何在放大后失灵；我们为何在训练了 1T tokens 后重启；如何在多语言、数学与代码的 competing objectives（相互冲突的目标）之间取得平衡，同时保持强劲的英文表现；最终，我们又如何对一款混合推理模型进行 post-training（后训练）。

我们也尽量避免罗列冷冰冰的“我们做了什么”，而是把这段冒险整理成一条有温度的叙事线。把它当成一本指南，献给所有想从“我们有了好数据集和 GPU”走向“我们炼出了真·强模型”的人。我们希望这份开放能缩小研究与生产之间的鸿沟，让你的下一次训练少几分混乱。

### [如何阅读这篇博客文章](https://huggingfacetb-smol-training-playbook.hf.space/#how-to-read-this-blog-post)

你无需从头到尾通读这篇博客文章，实际上它现在已经太长，不太可能一口气读完。整篇文章被拆成若干独立部分，可以按需跳过或单独阅读：

*   训练指南针（Training compass）： 关于是否应该自己预训练模型的高层次讨论。我们会带你梳理在烧光所有风投的钱之前必须问自己的根本问题，并教你如何系统性地思考整个决策流程。这是一个偏战略的部分；如果你想直接看技术内容，可以快速滑过。
*   预训练（Pretraining）： 紧随训练指南针之后的章节涵盖构建专属预训练配方所需的一切：如何运行消融实验（ablations）、选择评估指标、混合数据源、做架构选型、调超参数，并最终扛住训练马拉松。即使你并不打算从零开始预训练，而是想做继续预训练（continued pretraining，又称 mid-training），这部分同样适用。
*   后训练（Post-training）： 在博客的这部分，你将学到把预训练模型潜力发挥到极致的全部技巧。从 SFT、DPO 到 GRPO 的后训练字母表，再到模型合并（model merging）的暗黑艺术与炼金术，一应俱全。让这些算法真正跑通的知识大多来自血泪教训，我们会在此分享经验，希望能帮你少踩几个坑。
*   基础设施（Infrastructure）： 如果说预训练是蛋糕本体，后训练是糖霜和樱桃，那么基础设施就是工业级烤箱。没有它，什么都做不成；一旦它出问题，你愉快的周日烘焙就会变成火灾隐患。如何理解、分析并调试 GPU 集群的知识散落在各种库、文档和论坛中。本部分将带你梳理 GPU 布局、CPU/GPU/节点/存储之间的通信模式，以及如何识别并消除瓶颈。

那么，我们从哪里开始呢？挑一个你最感兴趣的章节，出发吧！

[训练指南针：为什么 → 做什么 → 怎么做](https://huggingfacetb-smol-training-playbook.hf.space/#training-compass-why--what--how)
----------------------------------------------------------------------------------------------------------------------------

机器学习（machine learning）领域对优化（optimisation）有着近乎痴迷的执着。我们紧盯损失曲线（loss curves）、模型架构（model architectures）和吞吐量（throughput）；毕竟，机器学习本质上就是在优化模型的损失函数（loss function）。然而，在深入这些技术细节之前，有一个更根本的问题却常被忽略：我们真的应该训练这个模型吗？

如下热力图所示，开源 AI 生态系统几乎每天都在发布世界级模型：Qwen、Gemma、DeepSeek、Kimi、Llama 🪦、Olmo……这份名单每月都在变长。它们并非研究原型或玩具示例，而是覆盖多语言理解、代码生成、推理等惊人广度用例的生产级模型。大多数模型附带宽松许可证，并拥有活跃社区随时为你提供支持。

这引出了一个令人不安的事实：也许你根本不需要训练自己的模型。

在一篇“LLM 训练指南”里这样开头似乎有些奇怪。但许多失败的训练项目并非因为超参数（hyperparameters）糟糕或代码有 bug，而是因为有人决定训练一个并不需要的模型。因此，在你承诺训练、并深入探讨如何执行之前，必须先回答两个问题：为什么要训练这个模型？训练什么模型？若答案不清，你将浪费数月算力和工程时间，重复世界已有的成果，甚至更糟——造出无人需要的东西。

让我们先从为什么开始，因为若不理解目的，后续任何决策都将失去连贯性。

本节与博客其余部分不同：它较少涉及实验与技术细节，更多关乎战略规划。我们将引导你判断是否需要从头训练（train from scratch）以及该构建怎样的模型。如果你已经深入思考过“为什么”和“做什么”，可直接跳转到[Every big model starts with a small ablation](https://huggingfacetb-smol-training-playbook.hf.space/#every-big-model-starts-with-a-small-ablation)章节进行技术深挖。但若你仍存疑虑，在此投入时间将为你后续节省大量精力。

### [Why：没人愿意回答的问题](https://huggingfacetb-smol-training-playbook.hf.space/#why-the-question-nobody-wants-to-answer)

咱们直说现实里发生了什么。有人（如果走运）搞到了 GPU 集群，也许是靠研究经费，也许是公司闲置算力，然后思路大概是这样的：“我们有 100 张 H100，能用三个月。训个模型吧！”模型尺寸随手定，数据集有啥拼啥。训练开始。六个月后，算力预算烧光、团队士气耗尽，训出来的模型却没人用——因为从一开始就没问过 _为什么_。

下面是你不该训模型的一些理由：

“我们训了自己的模型”这句话确实诱人，但在投入大量时间和资源之前，先问一句：你为什么要训这个模型？

下面的流程图展示了启动大规模预训练（pretraining）项目前应经历的思考过程。从技术角度看，你首先得确认有没有现成的模型可以直接提示（prompt）或微调（fine-tune）来完成任务。

通常只有三种情况值得做自定义预训练：你想做前沿研究、你的生产场景需求极其特殊，或者你想填补开源模型生态的空白。咱们快速过一遍：

#### [研究：你想理解什么？](https://huggingfacetb-smol-training-playbook.hf.space/#research-what-do-you-want-to-understand)

在 LLM（大语言模型，Large Language Model）领域，可以开展的研究非常丰富。LLM 研究项目通常有一个共同点：你首先要提出一个清晰明确的问题：

*   我们能否在这种新优化器（optimiser）上将训练扩展到 10B+ 参数？来自论文 [Muon is Scalable for LLM Training](https://huggingface.co/papers/2502.16982)
*   仅通过强化学习（reinforcement learning），而不使用监督微调（SFT，Supervised Fine-Tuning），能否产生推理能力？来自论文 [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://huggingface.co/papers/2501.12948)
*   我们能否仅使用合成教科书数据训练出优秀的小模型？来自论文 [Textbooks Are All You Need](https://huggingface.co/papers/2306.11644)
*   仅使用公开许可的数据进行训练，能否达到具有竞争力的性能？来自论文 [The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed Text](https://huggingface.co/papers/2506.05209)

尽可能具体地提出假设，并思考所需的实验规模，可以显著提高成功的概率。

#### [生产阶段：为什么不能直接复用现有模型？](https://huggingfacetb-smol-training-playbook.hf.space/#production-why-cant-you-use-an-existing-model)

企业无法直接拿现成模型（off-the-shelf model）落地，通常有三类原因：两类是技术性的，另一类关乎治理。

第一，领域特异性（domain specificity）： 当数据或任务包含高度专业化的词汇或结构，而现有模型难以胜任时。例如：

*   DNA 模型需要独特的词表并捕捉长程依赖。  
*   法律或金融模型必须深谙行业术语与逻辑。

第二，部署约束（deployment constraints）： 需要针对自有硬件、延迟或隐私要求定制模型。例如，在无人机或本地部署系统上运行 LLM，且硬件为 FPGA 等定制芯片。

一个简单的自检方法：花几天时间在 Qwen3、Gemma3 或其他当前 SOTA（state-of-the-art，最先进）模型上做提示工程、工具调用或后训练（post-training）。如果仍无法达到性能目标，就该考虑自训模型了。

即使后训练所需预算巨大，通常仍比从零开始训练划算。用 1T token 做微调，比从 0 训到 10T+ token 更经济。

第三，安全与治理（safety and governance）： 身处受监管行业或高 stakes（高风险）场景时，必须完全掌控训练数据、模型行为与更新周期。你需要确切知道模型里放了什么，并能向监管机构证明。某些情况下，自建模型是唯一选择。

以上是企业自训内部模型的主要原因。那么，那些发布开源模型的公司或组织又是出于什么考虑呢？

#### [战略性开源：你发现了可以填补的空白吗？](https://huggingfacetb-smol-training-playbook.hf.space/#strategic-open-source-do-you-see-a-gap-you-can-fill)

经验丰富的 AI 实验室发布新开源模型最常见的原因之一，是他们识别到了开源生态系统中某个具体的空白或新的 AI 使用场景。

典型流程如下：你注意到一个尚未被充分探索的领域——也许市面上缺乏具备超长上下文的强大小模型（on-device models），或者虽然有多语言模型（multilingual models），但在低资源语言（low-resource languages）上表现薄弱；又或者领域正朝着交互式世界模型（interactive world-models）如 [Genie3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/) 发展，却没有优秀的开放权重（open-weight）模型可用。

你有理由相信你能做得更好：也许你整理了更优质的训练数据，开发了更好的训练方案（training recipes），或者拥有别人无法企及的算力来做过度训练（overtrain）。你的目标非常具体：不是“史上最佳模型”，而是“最适合端侧使用的 3B 模型”或“首个支持 1M 上下文的小模型”。

这是一个真实可行的目标，成功就能创造价值：开发者会采用你的模型，它会成为他人的基础设施，或为你建立技术信誉。但成功需要经验。你必须知道什么是真正可实现的，以及如何在一个竞争激烈的空间里可靠地执行。为了具体说明，我们来看看 Hugging Face 内部如何思考这个问题。

#### [Hugging Face 的旅程](https://huggingfacetb-smol-training-playbook.hf.space/#hugging-faces-journey)

Hugging Face 为什么要训练开源模型？答案很简单：我们构建对开源生态系统有用的东西，填补其他人很少填补的空白。

这包括数据集、工具和模型训练。我们启动的每一个大语言模型（LLM, Large Language Model）训练项目，都是源于我们发现了一个空白，并相信我们可以做出有意义的贡献。

在 GPT-3（[Brown 等，2020](https://arxiv.org/abs/2005.14165)）发布后，我们启动了第一个 LLM 项目。当时，感觉没有其他人正在构建一个开放的替代方案，我们担心相关知识最终会被锁在少数几家行业实验室里。因此，我们发起了 [BigScience 研讨会](https://bigscience.huggingface.co/)，以训练一个开放版本的 GPT-3。最终得到的模型是 [Bloom](https://huggingface.co/bigscience/bloom)，它由数十位贡献者历时一年，共同构建训练栈、分词器（tokenizer）和预训练语料库，以预训练一个 175B 参数的模型。

Bloom 的继任者是 2022 年的 StarCoder（[Li et al., 2023](https://arxiv.org/abs/2305.06161)）。OpenAI 已为 GitHub Copilot 开发了 Codex（[Chen et al., 2021](https://arxiv.org/abs/2107.03374)），但它是闭源的。显然，构建一个开源替代方案将为生态系统提供价值。因此，我们与 ServiceNow 合作，在 [BigCode](https://huggingface.co/bigcode) 项目下，构建了 [The Stack](https://huggingface.co/datasets/bigcode/the-stack) 数据集，并训练了 [StarCoder 15B](https://huggingface.co/bigcode/starcoder) 来复现 Codex。[StarCoder2](https://huggingface.co/collections/bigcode/starcoder2-65de6da6e87db3383572be1a)（[Lozhkov et al., 2024](https://arxiv.org/abs/2402.19173)）源于我们认识到可以训练更长时间，并且意识到训练时间更长的小模型可能比一个大模型更有价值。我们在数万亿个 token 上训练了一个模型系列（3B/7B/15B），远超当时任何人对开放代码模型的训练规模。

[SmolLM 系列](https://huggingface.co/HuggingFaceTB) 遵循了类似的模式。我们注意到当时几乎没有强大的小模型，而我们刚刚构建了 [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)（[Penedo et al., 2024](https://arxiv.org/abs/2406.17557)），这是一个强大的预训练数据集。[SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)（135M/360M/1.7B）是我们的第一个版本。[SmolLM2](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9)（[Allal et al., 2025](https://arxiv.org/abs/2502.02737)）专注于更好的数据和更长的训练时间，在多个方面达到了 SOTA（state-of-the-art，最先进）性能。[SmolLM3](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) 扩展到 3B，同时添加了混合推理、多语言和长上下文功能，这些是 2025 年社区所重视的特性。

这种模式不仅限于预训练：我们训练了 [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha)（[Tunstall 等，2023](https://arxiv.org/abs/2310.16944)）以证明 DPO（Direct Preference Optimization，直接偏好优化）在大规模场景下有效；启动了 [Open-R1](https://github.com/huggingface/open-r1) 来复现 DeepSeek R1 的蒸馏流程；并发布了 [OlympicCoder](https://huggingface.co/open-r1/OlympicCoder-7B) 用于竞技编程，在国际信息学奥林匹克竞赛中取得了 SOTA（State-Of-The-Art，最先进）成绩。我们还将探索扩展到其他模态，包括用于视觉的 [SmolVLM](https://huggingface.co/collections/HuggingFaceTB/smolvlm-6740bd584b2dcbf51ecb1f39)（[Marafioti 等，2025](https://arxiv.org/abs/2504.05299)）和用于机器人学的 [SmolVLA](https://huggingface.co/lerobot/smolvla_base)（[Shukor 等，2025](https://arxiv.org/abs/2506.01844)）。

希望本节已经让你相信，深入思考“为什么要训练模型”是有价值的。

在本文的剩余部分，我们将假设你已经完成了这番灵魂拷问，并且确实有正当的理由去训练模型。

### [What：将目标转化为决策](https://huggingfacetb-smol-training-playbook.hf.space/#what-translating-goals-into-decisions)

既然你已经明确了为什么要训练，那么应该训练什么？这里的“什么”指的是：模型类型（dense（稠密）、MoE（混合专家）、hybrid（混合）、还是全新架构）、模型规模、架构细节以及数据混合比例。一旦确定了“为什么”，就可以推导出“什么”。例如：

*   用于设备端的快速模型 → 小型高效模型  
*   多语言模型 → 大词表分词器（tokenizer）  
*   超长上下文 → 混合（hybrid）架构  

除了受用例驱动的决策外，还有一些选择可以优化训练本身，使其更稳定、样本效率更高或速度更快。这些决策并不总是非黑即白，但你可以大致把决策过程分成两个阶段：

规划（Planning）： 在跑实验之前，把你的用例映射到需要决定的组件上。部署环境决定了模型规模的限制；时间线决定了你能承担哪些架构风险；目标能力决定了数据集需求。这一阶段就是把“为什么”里的每一条约束，具体化为“什么”里的技术规格。

验证（Validation）： 一旦有了起点和一系列潜在改动，就要系统地测试。由于测试代价高昂，应聚焦于那些可能显著提升用例性能或优化训练的改动。这时就需要做消融实验（ablations），详见[消融实验章节](https://huggingfacetb-smol-training-playbook.hf.space/#every-big-model-starts-with-a-small-ablation)。

在无关选择上做得再完美的消融，也和在关键选择上做得粗糙的消融一样浪费算力。

在接下来的章节中，你将了解定义模型时可用的全部选项，以及如何通过系统性实验缩小选择范围。在此之前，我们想分享一些经验：这些经验来自我们训练自有模型以及观察其他优秀团队构建出色 LLM（Large Language Model，大语言模型）的过程中，关于如何组建团队和规划项目的体会。

### [超能力：速度与数据](https://huggingfacetb-smol-training-playbook.hf.space/#super-power-speed-and-data)

通往罗马的路当然有很多条，但我们发现，迭代速度（iteration speed） 才是持续拉开成功 LLM 训练团队差距的关键。训练 LLM 本质上是一门“边训边学”的技艺——训得越多，团队成长越快。因此，一年只训一次模型的团队与每季度就能训一次的团队相比，后者进步会快得多。看看 Qwen 和 DeepSeek 的团队就知道了：如今家喻户晓，他们长期保持快速发布新模型的节奏。

除了迭代速度，数据整理（data curation） 无疑是 LLM 训练中最具影响力的环节。人们天然倾向于扎进架构选择以提升模型，但真正在 LLM 训练上出类拔萃的团队，无一不是对高质量数据近乎偏执。

与迭代速度紧密相关的另一个因素是团队规模：对于主要的预训练任务，只需少数几人，配备足够算力即可。今天预训练一个 Llama 3 级别的模型，大概 $2–3$ 人足矣。只有当你开始涉足更多样化的训练与下游任务（多模态、多语言、后训练等）时，才需要逐步增加人手，以在各领域做到极致。

所以，从一个装备精良的小团队起步，每 2–3 个月打造一个新模型，用不了多久你就能登顶。接下来，本文的其余部分将聚焦这支团队的日常技术实践！

[每个大模型都始于一次小型消融实验](https://huggingfacetb-smol-training-playbook.hf.space/#every-big-model-starts-with-a-small-ablation)
---------------------------------------------------------------------------------------------------------------------------------------------------

在我们开始训练一个 LLM（大语言模型，Large Language Model）之前，需要做出许多将影响模型性能与训练效率的决策：哪种架构（architecture）最能满足我们的用例？该用哪种优化器（optimiser）和学习率调度（learning rate schedule）？又该混合哪些数据源？

这些决策如何做，是常被问到的问题。人们有时以为只要深思熟虑即可。诚然，战略思考至关重要——正如我们在[上一节](https://huggingfacetb-smol-training-playbook.hf.space/#training-compass-why--what--how)中讨论过，如何识别哪些架构改动值得测试——但仅靠推理是不够的。LLM 的许多现象并不直观，关于“应该有效”的假设在实践中往往行不通。

例如，使用看似“最高质量”的数据并不总能带来更强的模型。以 [arXiv](https://arxiv.org/) 为例，它汇集了人类庞大的科学知识。直觉上，用这样丰富的 STEM 数据训练应该能产出更优秀的模型，对吧？实际上却不然，尤其对较小模型，甚至可能损害性能（[Shao et al., 2024](https://arxiv.org/abs/2402.03300)）。为何？原因在于，虽然 arXiv 论文知识密度高，但它们高度专业化，且采用狭窄的学术写作风格，与模型最擅长学习的多样化、通用文本差异很大。

那么，如果苦思冥想也无济于事，我们怎么知道什么才有效？像优秀的经验主义者一样，我们跑大量实验！机器学习并非纯数学，而是一门实验科学。

既然这些实验将指导许多关键决策，好好设计它们就格外重要。我们主要希望实验具备两项核心属性：

1.  速度（Speed）： 它们应尽可能快地运行，以便我们能频繁迭代。能跑的消融（ablation）越多，能验证的假设就越多。  
2.  可靠性（Reliability）： 它们应具备强判别力。如果我们关注的指标在早期无法有意义地区分不同配置，消融就可能收效甚微（如果指标噪声大，我们还会陷入“追噪声”的风险！）。更多细节请参阅 [FineTaks 博客文章](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fine-tasks)。

但在搭建消融实验之前，我们必须先对架构类型（architecture type）和模型规模（model size）做出基础性选择。这些由“指南针”指引的决策，会影响使用哪种训练框架、如何分配算力预算，以及从哪个基线（baseline）出发。

对于 SmolLM3，我们选择了 3B 参数的稠密 Llama 风格架构，因为目标是小型设备端模型。但在[《设计模型架构》章节](https://huggingfacetb-smol-training-playbook.hf.space/#designing-the-model-architecture)中你会看到，MoE（Mixture of Experts，混合专家）或混合架构可能更适合你的场景，而不同模型规模也各有权衡。稍后我们将深入探讨这些选择，并展示如何做出决策。现在，让我们从最实用的第一步开始：选择你的基线。

### [选择你的基线模型](https://huggingfacetb-smol-training-playbook.hf.space/#choosing-your-baseline)

每一个成功的模型都建立在经过验证的基础之上，并针对自身需求进行了修改。当 Qwen 训练其首个模型家族（[Bai et al., 2023](https://arxiv.org/abs/2309.16609)）时，他们从 Llama 的架构起步；当 Meta 训练 Llama 3 时，他们从 Llama 2 出发；Kimi K2 则始于 DeepSeek-V3 的 MoE（Mixture of Experts，混合专家）架构。这种做法不仅适用于架构，也适用于训练超参数（hyperparameters）和优化器（optimisers）。

为什么？优秀的架构和训练设置需要多年、多组织的反复迭代。标准的 Transformer 和 Adam 等优化器已经通过数千次实验被不断打磨。人们已经发现它们的失效模式、调试了不稳定性、优化了实现。从一个经过验证的基础出发，意味着继承所有这些累积的知识；而从零开始，则意味着自己要重新发现每一个问题。

以下是作为架构起点时应具备的条件：

*   符合你的约束：与你的部署目标和使用场景保持一致。
*   在大规模上被验证过：在相同或更大规模的多万亿 token 训练运行中已被证明有效。
*   文档完善：拥有在开源模型中被验证有效的已知超参数。
*   框架支持：最好在你考虑的训练框架以及计划使用的推理框架中都得到支持。

以下是截至 2025 年、适用于不同架构和模型规模的强基线选项的非 exhaustive（非穷尽）列表：

| Architecture Type | Model Family | Sizes |
| --- | --- | --- |
| Dense | [Llama 3.1](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f) | 8B, 70B |
| Dense | [Llama 3.2](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf) | 1B, 3B |
| Dense | [Qwen3](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) | 0.6B, 1.7B, 4B, 14B, 32B |
| Dense | [Gemma3](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d) | 12B, 27B |
| Dense | [SmolLM2](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9), [SmolLM3](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | 135M, 360M, 1.7B, 3B |
| MoE | [Qwen3 MoE](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) | 30B-A3B, 235B-A12B |
| MoE | [GPT-OSS](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4) | 21B-A3B, 117B-A5B |
| MoE | [Kimi Moonlight](https://huggingface.co/moonshotai/Moonlight-16B-A3B-Instruct) | 16B-A3B |
| MoE | [Kimi-k2](https://huggingface.co/collections/moonshotai/kimi-k2-6871243b990f2af5ba60617d) | 1T-A32B |
| MoE | [DeepSeek V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) | 671B-A37B |
| Hybrid | [Zamba2](https://huggingface.co/Zyphra/models?search=zamba2) | 1.2B, 2.7B, 7B |
| Hybrid | [Falcon-H1](https://huggingface.co/collections/tiiuae/falcon-h1-6819f2795bc406da60fab8df) | 0.5B, 1.5B, 3B, 7B, 34B |
| MoE + Hybrid | [Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) | 80B-A3B |
| MoE + Hybrid | [MiniMax-01](https://huggingface.co/MiniMaxAI/MiniMax-Text-01) | 456B-A46B |

因此，请根据你的 架构类型（architecture type） 选择一个与你期望的参数量接近的基线。不必过度纠结，因为你所起始的架构并非一成不变。在下一节中，我们将看到如何从一个基线出发，最终得到一个对你而言最优的架构。

#### [修改你的基线：降低风险的纪律](https://huggingfacetb-smol-training-playbook.hf.space/#modifying-your-baseline-the-discipline-of-derisking)

现在你有了一个可用的基线（baseline），并且它符合你的用例。你可以就此打住，在你的数据混合（data mixture）上训练它（假设数据质量不错），很可能得到一个不错的模型。许多成功的项目就是这么做的。但基线并非为你的具体约束而优化，它们是为构建者自己的用例和部署目标设计的。因此，很可能值得做一些修改，使其更好地契合你的目标。然而，每一次架构改动都伴随着风险：它可能提升性能，也可能拉垮性能，或者毫无效果却白白浪费了你的消融（ablation）算力。

让你保持正轨的纪律是 derisking（降低风险）：除非已验证某改动有益，否则绝不更改任何内容。

当测试表明某改动要么在你关注的能力上提升了性能，要么带来了有意义的收益（例如推理更快、内存更低、稳定性更好），同时没有让性能跌出你可接受的权衡范围，那么该改动就被视为“已降低风险”。

棘手的地方在于，你的基线和训练设置里有许多可以调整的组件：注意力机制（attention mechanisms）、位置编码（positional encodings）、激活函数（activation functions）、优化器（optimisers）、训练超参数（training hyperparameters）、归一化方案（normalisation schemes）、模型布局（model layout）等等。每一个都是潜在的实验对象，而这些组件之间往往以非线性方式相互作用。你既没有时间，也没有算力去测试所有内容或探索每一种交互。

先从有前景的改动开始，与当前基线对比测试。当某个改动奏效时，将其整合进来形成新的基线，然后再针对这个新基线测试下一个改动。如果算力预算允许，你可以单独测试每个改动，并做一次“留一法”（leave-one-out）分析。

不要陷入“对所有超参数（hyperparameters）做穷举网格搜索”或“测试每一个新出的架构变体”的陷阱。

仅仅知道如何跑实验是不够的，还得先判断哪些实验值得跑。在对任何改动动手之前，先问自己两个问题：

*   这个改动是否对我的具体用例（use case）有帮助？
*   它是否能优化我的训练过程？

如果一条修改无法明确回答上述任何一个问题，那就跳过它。

既然你已经学会通过策略规划识别有前景的方向，接下来就该进入实证验证（empirical validation）阶段了。在接下来的几节中，我们会告诉你如何在实践中真正测试这些改动。我们将涵盖如何搭建可靠的实验、解读结果，以及避开常见陷阱。随后几章，我们会手把手演示测试主流架构（architectural）、数据（data）、基础设施（infra）和训练（training）决策的具体示例。

那么，先搭一个可用于实验的简单消融（ablation）框架吧。首先，得决定选用哪个训练框架（training framework）。

### [选择训练框架](https://huggingfacetb-smol-training-playbook.hf.space/#picking-a-training-framework)

我们需要做的第一个决定是使用哪个框架（framework）来训练模型，进而运行所有消融实验（ablations）。这一选择需要平衡三个关键考量：

1. 框架必须支持我们的目标架构，或允许我们轻松扩展。
2. 它需要稳定且可用于生产，不会在训练中途神秘崩溃。
3. 它应提供高吞吐量，以便我们快速迭代并充分利用计算预算。

在实践中，这些要求可能相互冲突，带来权衡。让我们看看可用的选项。

| 框架 | 功能 | 实战检验 | 优化程度 | 代码行数（核心 / 总计） | 可扩展性与调试 |
| --- | --- | --- | --- | --- | --- |
| Megatron-LM | ✅ 丰富 | ✅ Kimi-K2、Nemotron | ✅ 3D 并行先驱 | 93k / 269k | ⚠️ 初学者难以上手 |
| DeepSpeed | ✅ 丰富 | ✅ BLOOM、GLM | ✅ ZeRO 与 3D 并行先驱 | 94k / 194k | ⚠️ 初学者难以上手 |
| TorchTitan | ⚡ 功能持续增加 | ⚠️ 较新，但已获 PyTorch 团队测试 | ⚡ 针对稠密模型优化，MoE 改进进行中 | 7k / 9k | ⚡ 中等：需并行知识 |
| Nanotron | 🎯 极简，专为 HF 预训练定制 | ✅ 已验证（StarCoder、SmolLM） | ✅ 已优化（UltraScale Playbook） | 15k / 66k | ⚡ 中等：需并行知识 |

上表总结了主流框架之间的关键权衡。前三个框架的代码行数来自 TorchTitan 技术报告（[Liang et al., 2025](https://arxiv.org/abs/2410.06511)）。接下来我们逐一详细讨论：

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 来自 Nvidia，已问世多年并历经实战检验。它驱动了 Kimi 的 K2（[Team et al., 2025](https://arxiv.org/abs/2507.20534)）等模型，提供稳定的吞吐量，并具备我们所需的大多数生产级特性。但这种成熟度也带来了复杂性：对于新手来说，代码库可能难以导航和修改。

[DeepSpeed](https://github.com/deepspeedai/DeepSpeed) 属于同一类别。它是 ZeRO 优化（ZeRO optimisation）的先驱，驱动了 BLOOM 和 GLM 等模型。与 Megatron-LM 一样，它经过大量实战测试和优化，但也面临同样的复杂性挑战。庞大的代码库（总计 19.4 万行）在入门时可能令人望而生畏，尤其是在实现自定义功能或调试意外行为时。

另一方面，PyTorch 最新的 [TorchTitan](https://github.com/pytorch/torchtitan) 库更加轻量，导航也简单得多，这得益于其紧凑且模块化的代码库。它具备预训练所需的核心功能，非常适合快速实验。然而，由于较新，它尚未经过充分实战检验，且在积极开发中仍可能略显不稳定。

我们选择了另一条路，从零开始构建了自己的框架 nanotron。这让我们拥有完全的灵活性，并对大规模预训练有了深入理解；这些洞见后来演变为 [Ultra Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)。自开源该库以来，我们也从社区获得了宝贵反馈，不过在大多数情况下，我们仍需先自行对功能进行实战测试。该框架现已支持我们训练所需的所有生产级特性，但我们仍在扩展诸如 MoE（Mixture of Experts，混合专家）支持等领域。

从零开始构建在当时是合理的，但这需要大量投入团队的专业知识，以及调试问题和补充缺失功能的时间。一个强有力的替代方案是分叉（fork）一个现有框架，并根据你的需求进行增强。例如，Thinking Machines Lab 将其内部预训练库构建为 TorchTitan 的一个分叉（[来源](https://x.com/cHHillee/status/1949470943291805832)）。

最终，你的选择取决于团队的专业知识、目标功能，以及你愿意投入多少时间进行开发，而不是直接使用最成熟的生产级方案。

如果多个框架都能满足你的需求，请在你的具体硬件上比较它们的吞吐量（throughput）。对于快速实验和速度测试，更简洁的代码库往往更具优势。

### [消融实验设置](https://huggingfacetb-smol-training-playbook.hf.space/#ablation-setup)

框架（framework）选定后，我们现在需要设计消融实验（ablation setup）。我们需要既能快速迭代、又足够大规模以产生有效信号并能迁移到最终模型的实验。下面来看如何搭建。

#### [搭建我们的消融（ablation）框架](https://huggingfacetb-smol-training-playbook.hf.space/#setting-up-our-ablation-framework)

消融（ablation）的目标是在小规模上运行实验，并获得可以可靠外推到最终生产运行的结果。

主要有两种方法。首先，我们可以采用目标模型规模，但只在更少的 token 上训练。对于 SmolLM3 的消融实验，我们在 $100\mathrm{B}$ 个 token 上训练了完整的 $3\mathrm{B}$ 模型，而不是最终的 $11\mathrm{T}$。其次，如果目标模型太大，我们可以训练一个更小的代理（proxy）模型来做消融。例如，当 Kimi 开发其 $1\mathrm{T}$ 参数的 Kimi K2 模型（活跃参数 $32\mathrm{B}$）时，用完整规模做所有消融实验的成本高得令人望而却步，因此他们在一个 $3\mathrm{B}$ 的 MoE（Mixture of Experts，混合专家）模型上运行了部分消融实验，该模型活跃参数仅为 $0.5\mathrm{B}$（[Team et al., 2025](https://arxiv.org/abs/2507.20534)）。

一个关键问题是，这些小规模的发现是否真的能够迁移。根据我们的经验，如果某件事在小规模上损害了性能，你可以自信地将其排除在大规模之外。但如果某件事在小规模上有效，你仍应确保在足够数量的 token 上进行了训练，以便以高概率断定这些发现可以外推到更大规模。训练时间越长，消融模型越接近最终模型，结果就越可靠。

在这篇博客文章中，我们将使用一个基线（baseline）的vanilla transformer进行所有消融实验（ablations）。我们的主要配置是一个1B transformer，遵循 [Llama3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B) 的架构，在 45B tokens 上训练。使用这份 nanotron [配置](https://huggingface.co/datasets/HuggingFaceTB/training-guide-nanotron-configs/blob/main/baseline_config_1B.yaml)，在 8×H100 的单节点上训练大约需要 1.5 天（每 GPU 42k tokens/s）。在 SmolLM3 训练期间，我们还在一个 3B 模型上、以 100B tokens 的规模跑了这些消融实验（配置见[此处](https://huggingface.co/datasets/HuggingFaceTB/training-guide-nanotron-configs)）。我们将在每一章末尾分享那些结果（你会发现结论是一致的）。

我们的基线 1B 配置以结构化的 YAML 格式记录了所有关键训练细节。以下是主要章节：

```

## 数据集与混合权重（Datasets and mixing weights）
data_stages:
- data:
    dataset:
      dataset_folder:
      - fineweb-edu
      - stack-edu-python
      - finemath-3plus

      dataset_weights:
      - 0.7
      - 0.2
      - 0.1

## 模型架构（Model architecture），Llama3.2 1B 配置
model:
  model_config:
    hidden_size: 2048
    num_hidden_layers: 16
    num_attention_heads: 32
    num_key_value_heads: 8  
    intermediate_size: 8192
    max_position_embeddings: 4096
    rope_theta: 50000.0
    tie_word_embeddings: true

## 训练超参数（Training hyperparameters），带余弦调度（cosine schedule）的 AdamW
optimizer:
  clip_grad: 1.0
  learning_rate_scheduler:
    learning_rate: 0.0005
    lr_decay_starting_step: 2000
    lr_decay_steps: 18000
    lr_decay_style: cosine
    lr_warmup_steps: 2000
    lr_warmup_style: linear
    min_decay_lr: 5.0e-05
  optimizer_factory:
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    name: adamW

## 并行策略（Parallelism），单节点
parallelism:
  dp: 8  # 在 8 张 GPU 上做数据并行（Data parallel）
  tp: 1  # 1B 规模无需张量或流水线并行
  pp: 1 

## 分词器（Tokenizer）
tokenizer:
  tokenizer_max_length: 4096
  tokenizer_name_or_path: HuggingFaceTB/SmolLM3-3B

## 30B tokens 的批次大小、序列长度与总训练量
tokens:
  batch_accumulation_per_replica: 16
  micro_batch_size: 3 # GBS（全局批次大小 global batch size）= dp * batch_acc * MBS * sequence = 1.5 M tokens
  sequence_length: 4096
  train_steps: 20000 # GBS * 20000 = 30B

...(truncated)
```

在我们的消融实验（ablations）中，我们会根据测试内容仅修改对应部分，其余配置保持不变：测试[架构选择](https://huggingfacetb-smol-training-playbook.hf.space/#architecture-choices)时修改 `model` 部分，测试[优化器与训练超参数](https://huggingfacetb-smol-training-playbook.hf.space/#optimiser-and-training-hyperparameters)时修改 `optimizer` 部分，测试[数据策展](https://huggingfacetb-smol-training-playbook.hf.space/#the-art-of-data-curation)时修改 `data_stages` 部分。

每次消融只改变一个变量，其余保持不变。如果同时修改多项且性能提升，将无法确定具体原因。应单独测试每项修改，随后合并成功的改动并重新评估。

运行消融实验时，某些架构改动会显著改变参数量。例如，从 tied embeddings（共享嵌入）切换到 untied embeddings（非共享嵌入）会使嵌入参数量翻倍；而从 MHA（Multi-Head Attention，多头注意力）切换到 GQA（Grouped-Query Attention，分组查询注意力）或 MQA（Multi-Query Attention，多查询注意力）则会大幅减少注意力参数量。为确保公平比较，需跟踪参数量，并偶尔调整其他超参数（如 hidden size 或层数），使模型规模大致相同。下面是我们用于估算不同配置参数量的简单函数：

```python
from transformers import LlamaConfig, LlamaForCausalLM

def count_parameters(
    tie_embeddings=True,
    num_key_value_heads=4,
    num_attention_heads=32,
    hidden_size=2048,
    num_hidden_layers=16,
    intermediate_size=8192,
    vocab_size=128256,
    sequence_length=4096,
):
    config = LlamaConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=sequence_length,
        tie_word_embeddings=tie_embeddings,
    )
    model = LlamaForCausalLM(config)  
    return f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B"
```

我们还提供了一个交互式工具，用于可视化密集（dense）transformer 的 LLM 参数分布。在进行架构决策或设置消融实验（ablations）配置时，这会非常方便。

#### [理解什么有效：评估（evaluation）](https://huggingfacetb-smol-training-playbook.hf.space/#understanding-what-works-evaluation)

一旦我们启动了消融实验（ablations），如何判断哪些方法有效、哪些无效？

任何训练模型的人的第一直觉可能是去看损失（loss），确实，这很重要。你希望看到它平稳下降，没有剧烈波动或不稳定。对于许多架构选择，损失与下游性能（downstream performance）相关性良好，通常已经足够（[Y. Chen et al., 2025](https://arxiv.org/abs/2410.08527)）。然而，仅看损失并不总是可靠。以数据消融为例，你会发现用维基百科（Wikipedia）训练的损失低于用网页训练的损失（下一个 token 更容易预测），但这并不意味着你会得到一个更强大的模型。同样，如果我们在不同运行之间更换了分词器（tokenizer），损失就无法直接比较，因为文本被切分的方式不同。某些改动可能专门影响特定能力，如推理（reasoning）和数学（math），在平均损失中被稀释。最后但同样重要的是，模型即使在预训练损失（pretraining loss）收敛后，仍可能在下游任务上继续改进（[Liu et al., 2022](https://arxiv.org/abs/2210.14199)）。

我们需要更细粒度的评估（fine-grained evaluation）来看到全貌，理解这些微妙的影响，而自然的方法就是使用下游评估（downstream evaluations），测试知识、理解、推理以及对我们重要的其他领域。

对于这些消融实验，最好聚焦于能提供良好早期信号的任务，避免嘈杂的基准测试（noisy benchmarks）。在 [FineTasks](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fine-tasks) 和 [FineWeb2](https://arxiv.org/pdf/2506.20920) 中，可靠的评估任务由四项关键原则定义：

*   单调性（Monotonicity）： 随着模型训练时间延长，基准分数应持续提升。
*   低噪声（Low noise）： 在相同配置但不同随机种子下训练模型时，基准分数不应剧烈波动。
*   高于随机性能（Above-random performance）： 许多能力只在训练后期才显现，因此那些长时间保持随机水平表现的任务对消融实验并无帮助。例如，稍后我们将解释，多选题形式的 MMLU 就属于这种情况。
*   排序一致性（Ranking consistency）： 如果某种方法在早期阶段优于另一种方法，那么随着训练继续，这种排序应保持稳定。

任务的质量还取决于任务形式（task formulation，即我们如何向模型提问）和指标选择（metric choice，即我们如何计算答案得分）。

三种常见的任务形式是多选题格式（multiple choice format，MCF）、完形填空形式（cloze formulation，CF）和自由生成（freeform generation，FG）。多选题格式要求模型从提示中明确给出的若干选项中选择，并以 A/B/C/D 为前缀（例如 MMLU 的做法）。在完形填空形式中，我们比较不同选项的似然度，看哪个更可能被生成，而无需在提示中提供这些选项。在 FG 中，我们考察给定提示下贪心解码生成的准确率。FG 需要模型具备大量潜在知识，通常对模型来说过于困难，在完整训练前的小型预训练消融实验中并不实用。因此，在进行小规模消融实验时，我们主要采用多选题格式（MCF 或 CF）。

对于经过后训练的模型，FG 成为主要形式，因为我们希望评估模型能否实际生成有用的回答。我们将在[后训练章节](https://huggingfacetb-smol-training-playbook.hf.space/#beyond-base-models--post-training-in-2025)中介绍这些模型的评估方法。

研究还表明，模型在训练早期难以掌握 MCF（多选填空，Multiple-Choice Fill-in-the-blank），只有在经过大量训练后才会习得这一技能，因此 CF（完形填空，Cloze Formulation）更适合提供早期信号（[Du et al., 2025](https://arxiv.org/abs/2403.15796)；[Gu et al., 2025](https://arxiv.org/abs/2406.08446)；J. [Li et al., 2025](https://arxiv.org/abs/2406.11794)）。因此，我们在小规模消融实验中使用 CF，并在主训练流程中集成 MCF，因为一旦模型越过某一阈值、MCF 的信噪比足够高，它就能在中期训练阶段提供更优的信号。顺便说明，为了在类似 CF 的序列似然评估中为模型答案打分，我们将准确率定义为：在全部问题中，正确答案的对数概率按字符数归一化后最高的问题占比。该归一化可避免对较短答案的偏好偏差。

我们的消融评估套件包含 [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) 消融实验中的基准，但去掉了我们发现噪声过大的 SIQA。我们额外加入了数学与代码基准（如 GSM8K 和 HumanEval）以及长上下文基准 RULER，用于长上下文消融实验。这一任务集合以多种格式测试世界知识、推理与常识，如下表所示。为加快评估速度（牺牲一定信噪比），我们仅从每个基准中抽取 1,000 题进行评估（GSM8K、HumanEval 与 RULER 除外：在 3B SmolLM3 消融实验中我们使用了全部题目，但在下文 1B 实验中予以省略）。对于所有多选基准，我们同样采用上述 CF 评估方式。注意，在多语言消融及实际训练中，我们会增加更多基准以测试多语言能力，详情后述。这些评估通过 [LightEval](https://github.com/huggingface/lighteval) 运行，下表总结了各基准的关键特性：

| 基准测试 | 领域 | 任务类型 | 题目数量 | 测试内容 |
| --- | --- | --- | --- | --- |
| MMLU（Massive Multitask Language Understanding） | 知识 | 多项选择 | 14k | 涵盖 57 个学科的广泛学术知识 |
| ARC（AI2 Reasoning Challenge） | 科学与推理 | 多项选择 | 7k | 小学水平的科学推理 |
| HellaSwag | 常识推理 | 多项选择 | 10k | 日常情境的常识推理（故事补全） |
| WinoGrande | 常识推理 | 二选一 | 1.7k | 需要世界知识的代词消解 |
| CommonSenseQA | 常识推理 | 多项选择 | 1.1k | 日常概念的常识推理 |
| OpenBookQA | 科学 | 多项选择 | 500 | 结合推理的小学科学事实 |
| PIQA（Physical Interaction QA） | 物理常识 | 二选一 | 1.8k | 日常物品的物理常识 |
| GSM8K（Grade School Math 8K） | 数学 | 自由生成 | 1.3k | 小学数学应用题 |
| HumanEval | 代码 | 自由生成 | 164 | 根据文档字符串生成完整 Python 函数 |

让我们从每个基准中挑几道示例题目，具体感受一下这些评估到底在测什么：

浏览上面的示例，可以看到各基准的题目类型。注意 MMLU 和 ARC 用多项选择考查事实知识，GSM8K 要求计算数学题并给出数值答案，而 HumanEval 则需要生成完整的 Python 代码。这种多样性确保我们在消融实验（ablation study）中能够全面检验模型能力的不同方面。

消融实验用哪种数据混合？

对于架构消融（architecture ablations），我们在一组固定的高质量数据集上训练，这些数据集能在广泛任务中提供早期信号。我们使用英语（[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)）、数学（[FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath)）和代码（[Stack-Edu-Python](https://huggingface.co/datasets/HuggingFaceTB/stack-edu)）。架构方面的发现应能很好地推广到其他数据集和领域，包括多语言数据，因此我们可以保持数据混合简单。

对于数据消融（data ablations），我们采取相反的方法：固定架构，系统地改变数据混合比例，以了解不同数据源如何影响模型性能。

一个可靠的消融设置带来的真正价值，远不止于构建一个好模型。当我们在主训练运行中不可避免地遇到问题（无论准备多充分，它们都会出现）时，我们希望对自己做出的每一个决定都有信心，并能迅速识别哪些组件没有经过充分测试，可能是问题的根源。这种准备能节省调试时间，并为我们将来的心理健康保驾护航。

#### [估算消融实验成本](https://huggingfacetb-smol-training-playbook.hf.space/#estimating-ablations-cost)

消融实验（ablations）非常棒，但它们需要 GPU 时间，因此了解这些实验的成本是值得的。下表展示了我们为 SmolLM3 预训练所做的完整算力分解：主运行（包含偶发的停机时间）、训练前与训练中的消融实验，以及因意外的扩展问题被迫重启和调试所耗费的算力（稍后详述）。

| 阶段 | GPU 数量 | 天数 | GPU 小时 |
| --- | --- | --- | --- |
| 主预训练运行 | 384 | 30 | 276,480 |
| 消融实验（预训练） | 192 | 15 | 69,120 |
| 消融实验（训练中） | 192 | 10 | 46,080 |
| 训练重启与调试 | 384/192 | 3/4 | 46,080 |
| 总成本 | - | - | 437,760 |

这些数字揭示了一个重要事实：消融实验和调试总共消耗了 161,280 GPU 小时，超过主训练运行成本的一半（276,480 GPU 小时）。我们在 SmolLM3 的整个开发过程中共运行了 100 多次消融实验：预训练消融实验耗时 20 天，训练中消融实验耗时 10 天，另有 7 天用于从意外的训练问题中恢复并重启和调试（稍后详述）。

这凸显了为何必须将消融实验成本纳入算力预算：请为训练成本、消融实验以及应对意外的缓冲预留预算。如果你追求 SOTA（state-of-the-art）性能、实施新的架构改动，或尚未拥有经过验证的方案，消融实验将成为一项重大成本中心，而非小规模实验。

在进入下一节之前，让我们先确立每位运行实验的人都应遵守的一些基本规则。

### [参与规则](https://huggingfacetb-smol-training-playbook.hf.space/#rules-of-engagement)

> 太长不看：保持偏执。

验证你的评估套件（evaluation suite）。在训练任何模型之前，确保你的评估套件能够复现你将要对比的模型已发表的结果。如果任何基准测试（benchmark）是生成式的（例如 GSM8k），请格外偏执，手动检查几个样本，确保提示（prompt）格式正确，并且任何后处理（post-processing）都能提取到正确的信息。由于评估将指导每一个决策，做好这一步对项目的成功至关重要！

测试每一次改动，无论多小。不要低估看似无害的库升级或“只改了两行”的提交带来的影响。这些微小的改动可能会引入难以察觉的错误或性能变化，从而污染你的结果。你需要一个在对你重要的用例上有强大测试套件的库，以避免回归（regression）。

一次只改一件事。在实验之间保持其他所有条件完全相同。某些改动可能会以意想不到的方式相互作用，因此我们首先要评估每个改动的单独贡献，然后再尝试将它们组合起来，观察整体影响。

在足够多的 token 上训练，并使用充分的评估。如前所述，我们需要确保评估套件有良好的覆盖，并训练足够长的时间以获得可靠的信号。在这里走捷径会导致结果嘈杂和错误决策。遵循这些规则可能显得过于谨慎，但另一种选择是花费数天时间去调试神秘的性能下降，最终发现是几天前某个无关的依赖更新导致的。黄金原则：一旦你有了一个好的设置，任何改动都必须经过测试！

[设计模型架构](https://huggingfacetb-smol-training-playbook.hf.space/#designing-the-model-architecture)
---------------------------------------------------------------------------------------------------------------------------

既然实验框架已经就绪，是时候做出将定义我们模型的重大决策了。从模型大小到注意力机制，再到分词器（tokenizer）的选择，每一个决定都会带来约束与机遇，进而影响模型的训练与使用。

请记住[训练指南针](https://huggingfacetb-smol-training-playbook.hf.space/#training-compass-why--what--how)：在做出任何技术选择之前，我们必须先明确为什么（why）和是什么（what）。我们为什么要训练这个模型？它应该长什么样？

听起来显而易见，但正如我们在训练指南针中解释的，在此处的审慎思考会塑造后续决策，并防止我们在无尽的实验空间中迷失。我们是想打造一个英文 SOTA（state-of-the-art，最先进）模型？长上下文是否是优先事项？还是我们试图验证一种新架构？在这些情况下，训练循环可能看起来相似，但我们运行的实验以及愿意接受的权衡将有所不同。尽早回答这个问题，有助于我们决定如何在数据工作与架构工作之间分配时间，以及在开始训练前各自投入多少创新。

因此，让我们以身作则，回顾指导 SmolLM3 设计的目标。我们希望得到一个适用于端侧应用的强模型，具备有竞争力的多语言性能、扎实的数学与编程能力，以及稳健的长上下文处理能力。如前所述，这促使我们选择了 30 亿（$3B$）参数的稠密（dense）模型：足够大以获得强劲能力，又足够小以舒适地运行在手机上。鉴于边缘设备的内存限制以及我们的项目周期（约 3 个月），我们选择了稠密 Transformer，而非 MoE（Mixture of Experts，混合专家）或 Hybrid（混合）架构。

我们手上有 SmolLM2 在较小规模（$1.7B$ 参数）下针对英文的成熟配方，但放大模型规模意味着要重新验证所有环节，并迎接多语言（multilinguality）和更长上下文长度（extended context length）等新挑战。一个具体例子足以说明“明确目标”如何左右我们的策略：在 SmolLM2 中，我们直到预训练尾声才尝试扩展上下文长度，结果举步维艰；因此在 SmolLM3 中，我们从一开始就做了架构层面的选择——例如采用 NoPE（NoPE）以及 intra-document masking（文档内掩码，详见后文）——以最大化成功概率，最终奏效。

目标一旦清晰，我们就可以着手做出能够实现这些目标的技术决策。本章将系统梳理我们在三大核心决策上的方法论：架构（architecture）、数据（data）和超参数（hyperparameters）。可以把这看作战略规划阶段——把这些基础打牢，就能在真正的训练马拉松中避免代价高昂的错误。

### [架构选择](https://huggingfacetb-smol-training-playbook.hf.space/#architecture-choices)

如果你看看最近的模型，比如 Qwen3、Gemma3 或 DeepSeek v3，你会发现尽管它们各有差异，但都共享同一个基础——2017 年提出的 transformer 架构（[Vaswani et al., 2023](https://arxiv.org/abs/1706.03762)）。多年来发生变化的并非其基本结构，而是对核心组件的精细打磨。无论你构建的是稠密模型（dense model）、专家混合模型（Mixture of Experts，MoE）还是混合架构，使用的都是这些相同的构建块。

这些改进源于各团队对更高性能的追求以及对特定挑战的应对：推理时的内存限制、大规模训练的不稳定性，或处理更长上下文的需求。有些改动，例如从多头注意力（Multi-Head Attention，MHA）转向更节省算力的注意力变体如分组查询注意力（Grouped Query Attention，GQA）（[Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)），已被广泛采用。另一些，比如不同的位置编码方案，仍在讨论之中。最终，今天的实验将固化为明天的基线。

那么，如今的现代大语言模型（LLM）到底在用什么呢？让我们看看领先模型已趋同的选择。遗憾的是，并非所有模型都公开训练细节，但 DeepSeek、OLMo、Kimi 和 SmolLM 等家族提供了足够的透明度，使我们得以窥见当前格局：

| 模型 | 架构 | 参数量 | 训练 token 数 | 注意力机制 | 上下文长度（最终） | 位置编码 | 精度 | 初始化（标准差） | 优化器 | 最大学习率 | 学习率调度 | 预热步数 | 批量大小 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek LLM 7B | Dense | 7B | 2T | GQA | 4K | RoPE | BF16 | 0.006 | AdamW | $4.2 \times 10^{-4}$ | Multi-Step | 2K | 9.4M |
| DeepSeek LLM 67B | Dense | 67B | 2T | GQA | 4K | RoPE | BF16 | 0.006 | AdamW | $3.2 \times 10^{-4}$ | Multi-Step | 2K | 18.9M |
| DeepSeek V2 | MoE | 236B（21B 激活） | 8.1T | MLA | 128K | Partial RoPE | — | 0.006 | AdamW | $2.4 \times 10^{-4}$ | Multi-Step | 2K | 9.4M→37.7M（预热 225B） |
| DeepSeek V3 | MoE | 671B（37B 激活） | 14.8T | MLA | 129K | Partial RoPE | FP8 | 0.006 | AdamW | $2.2 \times 10^{-4}$ | Multi-Step + Cosine | 2K | 12.6M→62.9M（预热 469B） |
| MiniMax-01 | MoE + Hybrid | 456B（45.9B 激活） | 11.4T | Linear attention + GQA | 4M | Partial RoPE | — | Xavier init + deepnorm scaling | AdamW | $2 \times 10^{-4}$ | Multi-Step | 500 | 16M→32M→64M→128M |
| Kimi K2 | MoE | 1T（32B 激活） | 15.5T | MLA | 128K | Partial RoPE | BF16 | 约 0.006 | MuonClip | $2 \times 10^{-4}$ | WSD | 500 | 67M |
| OLMo 2 7B | Dense | 7B | 5T | MHA | 4K | RoPE | BF16 | 0.02 | AdamW | $3 \times 10^{-4}$ | Cosine | 2K | 4.2M |
| SmolLM3 | Dense | 3B | 11T | GQA | 128K | NoPE | BF16 | 0.02 | AdamW | $2 \times 10^{-4}$ | WSD | 2K | 2.3M |

如果你现在还不理解其中的一些术语，比如 MLA、NoPE 或 WSD，别担心。我们会在本节逐一解释。此刻只需留意它们的多样性：不同的注意力机制（MHA、GQA、MLA）、位置编码（RoPE、NoPE、Partial RoPE）以及学习率调度策略（Cosine、Multi-Step、WSD）。

面对这一长串架构选择，我们难免会感到无从下手。和大多数类似情况一样，我们将循序渐进，逐步积累所需的全部知识。首先聚焦最简单的基线架构（dense model，稠密模型），并逐一深入探究每个架构细节。随后，我们将深入 MoE（Mixture of Experts，混合专家）和 Hybrid（混合）模型，并讨论何时选用它们才是明智之举。最后，我们将探索 tokenizer（分词器）——一个常被忽视却至关重要的组件。我们应该直接用现有的，还是训练自己的？又该如何评估分词器的好坏？

在本章的其余部分，我们将通过消融实验（ablations）验证大多数架构选择，实验设置如上一章所述：以 1B 参数的基线模型（遵循 Llama3.2 1B 架构）在 45B token 的混合数据（FineWeb-Edu、FineMath 与 Python-Edu）上训练。对于每项实验，我们同时展示训练损失曲线与下游评测分数，以衡量每次改动的影响。所有运行的配置均可在 [HuggingFaceTB/training-guide-nanotron-configs](https://huggingface.co/datasets/HuggingFaceTB/training-guide-nanotron-configs/tree/main) 中找到。

现在，让我们从每个 LLM 的核心开始：attention mechanism（注意力机制）。

#### [注意力（Attention）](https://huggingfacetb-smol-training-playbook.hf.space/#attention)

Transformer 架构中最活跃的研究领域之一就是注意力机制。虽然在预训练阶段前馈层（feedforward layers）主导了计算量，但在推理阶段，注意力成为主要瓶颈（尤其是在长上下文场景），它会推高计算成本，并且 KV 缓存（KV cache）迅速消耗 GPU 显存，从而降低吞吐量。让我们快速浏览一下主要的注意力机制以及它们在容量与速度之间的权衡。

我的注意力需要多少个头？

_多头注意力（Multi-head attention，MHA）_ 是最初的 Transformer 论文（[Vaswani et al., 2023](https://arxiv.org/abs/1706.03762)）中提出的标准注意力机制。其核心思想是：你有 N 个注意力头，每个头独立执行相同的检索任务——将隐藏状态转换为查询（queries）、键（keys）和值（values），然后利用当前查询通过与键的匹配来检索最相关的 token，最后将匹配到的 token 对应的值向前传递。在推理阶段，我们无需重新计算过去 token 的 KV 值，而是可以复用它们。用于存储过去 KV 值的内存称为 _KV-Cache_。随着上下文窗口的增长，该缓存很快就会成为推理瓶颈，并占用大量 GPU 显存。下面是一个简单计算，用于估算在 Llama 3 架构下使用 MHA、序列长度为 8192 时的 KV-Cache 内存 $s_{KV}$：

$$
\begin{aligned}
s_{KV} &= 2 \times n_{bytes} \times seq \times n_{layers} \times n_{heads} \times dim_{heads} \\
&= 2 \times 2 \times 8192 \times 32 \times 32 \times 128 = 4 \text{ GB} \textit{ (Llama 3 8B)} \\
&= 2 \times 2 \times 8192 \times 80 \times 64 \times 128 = 20 \text{ GB} \textit{ (Llama 3 70B)}
\end{aligned}
$$

注意，最前面的系数 2 来自于同时存储 key（键）和 value（值）缓存。如你所见，缓存随序列长度线性增长，但上下文窗口却呈指数级扩展，如今已可达数百万个 token。因此，提升缓存效率将使在推理时扩展上下文变得更加容易。

自然会问：我们真的需要为每个注意力头（head）都准备一组新的 KV 值吗？大概不需要。Multi-Query Attention（MQA，多查询注意力）（[Shazeer, 2019](https://arxiv.org/abs/1911.02150)）和 Grouped Query Attention（GQA，分组查询注意力）（[Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)）都解决了这一问题。最简单的情况是在所有头之间共享 KV 值，从而将 KV 缓存大小除以 $n_{heads}$，例如对 Llama 3 70B 而言可减少 64 倍！这就是 MQA 的思想，并已在诸如 StarCoder 等模型中用作 MHA（Multi-Head Attention，多头注意力）的替代方案。然而，我们可能会因此牺牲过多的注意力容量，于是可以考虑折中方案：在若干头组成的组内共享同一组 KV 值，例如 4 个头共享一组 KV。这就是 GQA 方法，在 MQA 与 MHA 之间取得了平衡。

最近，DeepSeek-v2（也在 v3 中使用）引入了 Multi-Latent Attention (MLA，多潜在注意力)（[DeepSeek-AI et al., 2024](https://arxiv.org/abs/2405.04434)），它采用了一种不同的策略来压缩缓存：不是减少 KV 值的数量，而是减小它们的大小，仅存储一个潜在变量，该变量可在运行时解压为 KV 值。借助这种方法，他们成功地将缓存降低到相当于 GQA 2.25 组的规模，同时性能还优于 MHA！为了与 RoPE 兼容，需要用一个额外的小潜在向量进行微调。在 DeepSeek-v2 中，主潜在变量维度选为 $4\times dim_{head}$，RoPE 部分为 $1/2\times dim_{head}$，因此总共是 $4.5\times dim_{head}$，同时用于 K 和 V，从而去掉了前面的系数 2。

你可以在下面的图示中直观地了解每种注意力机制：

本节讨论的注意力机制对比如下表所示。为简化起见，我们比较每个 token 使用的参数量；若要计算总内存，只需乘以每个参数的字节数（通常为 2）和序列长度即可：

| Attention Mechanism（注意力机制） | KV-Cache parameters per token（每个 token 的 KV 缓存参数量） |
| --- | --- |
| MHA | $=2×n_{heads}×n_{layers}×dim_{head}$ |
| MQA | $=2×1×n_{layers}×dim_{head}$ |
| GQA | $=2×g×n_{layers}×dim_{head}$（通常 $g=2,4,8$） |
| MLA | $=4.5×n_{layers}×dim_{head}$ |

现在让我们看看这些注意力机制在真实实验中的表现！

Ablation - GQA beats MHA（消融实验 - GQA 优于 MHA）

下面我们来比较不同的注意力机制。我们的[基线](https://huggingface.co/datasets/HuggingFaceTB/ablations-training-configs/blob/main/baseline_config_1B.yaml)模型使用 32 个 Query 头（查询头）和 8 个 KV 头（键值头），对应 GQA（Grouped Query Attention，分组查询注意力）的压缩比为 $32/8=4$。如果改用 MHA（Multi-Head Attention，多头注意力），或者进一步减少 KV 头数量、提高 GQA 压缩比，性能会如何变化？

改变 KV 头数量会显著影响参数量，尤其在 MHA 场景下。为保持一致，我们对 MHA 实验减少了层数，否则其参数量将多出 1 亿以上；其余配置均保持默认的 16 层。

| 注意力类型 | Query 头 | KV 头 | 层数 | 参数量 | 备注 |
| --- | --- | --- | --- | --- | --- |
| MQA（Multi-Query Attention，多查询注意力） | 32 | 1 | 16 | 1.21B |  |
| GQA（压缩比 16） | 32 | 2 | 16 | 1.21B |  |
| GQA（压缩比 8） | 32 | 4 | 16 | 1.22B | 基线 |
| GQA（压缩比 4） | 32 | 8 | 16 | 1.24B |  |
| GQA（压缩比 2） | 32 | 16 | 15 | 1.22B | 减少层数 |
| MHA | 32 | 32 | 14 | 1.20B | 减少层数 |
| GQA（压缩比 2） | 32 | 16 | 16 | 1.27B | 过大 - 未消融 |
| MHA | 32 | 32 | 16 | 1.34B | 过大 - 未消融 |

因此，我们比较了 MHA、MQA 以及 4 种 GQA 配置（压缩比 2、4、8、16）。对应的 nanotron 配置文件可在此[获取](https://huggingface.co/datasets/HuggingFaceTB/training-guide-nanotron-configs/tree/main/attention)。

从消融结果看，MQA 和仅保留 $1 \sim 2$ 个 KV 头的 GQA（$16$ 组）相比 MHA 显著落后；而 GQA 在 $2$、$4$、$8$ 组配置下则与 MHA 性能大致相当。

这一结论在损失曲线和下游评测中均保持一致。我们在 HellaSwag、MMLU、ARC 等基准上能清晰观察到该趋势，而 OpenBookQA 和 WinoGrande 的波动则稍大。

基于这些消融实验（ablations），GQA（Grouped Query Attention，分组查询注意力）是 MHA（Multi-Head Attention，多头注意力）的一个可靠替代方案。它在保持性能的同时，在推理阶段更加高效。一些近期模型采用了 MLA（Multi-Head Latent Attention，多头潜在注意力）以实现更大的 KV 缓存压缩，不过它尚未被广泛采用。由于在进行消融实验时 nanotron 尚未实现 MLA，我们并未对其进行消融。对于 SmolLM3，我们使用了 $4$ 个组的 GQA。

除了注意力架构本身，训练期间使用的注意力模式也很重要。让我们来看看注意力掩码（attention masking）。

文档掩码（Document masking）

我们在训练序列中如何应用注意力，既影响计算效率，也影响模型性能。这就引出了*文档掩码（document masking）*以及更广泛的、如何在数据加载器（dataloader）中构造训练样本的问题。

在预训练阶段，我们使用固定序列长度进行训练，但文档长度各异。一篇研究论文可能有 $10k$ 个词元（tokens），而一段简短的代码片段只有几百个词元。我们如何将可变长度的文档装入固定长度的训练序列？将较短文档填充（padding）到目标长度会浪费算力在无意义的填充词元上。相反，我们使用打包（packing）：打乱并将文档用序列结束（EOS）词元拼接，然后将结果切分成与序列长度匹配的固定长度块。

实际过程如下：

```
File 1: "Recipe for granola bars..." (400 tokens) <EOS>
File 2: "def hello_world()..." (300 tokens) <EOS>  
File 3: "Climate change impacts..." (1000 tokens) <EOS>  
File 4: "import numpy as np..." (3000 tokens) <EOS>  
...

After concatenation and chunking into 4k sequences:  
Sequence 1: [File 1] + [File 2] + [File 3] + [partial File 4]  
Sequence 2: [rest of File 4] + [File 5] + [File 6] + ...
```

如果某个文件足够长，能够填满我们的 $4k$ 上下文，那么一个训练序列可能只包含一个完整文件；但在大多数情况下文件较短，因此序列会包含多个随机文件的拼接。

在标准的因果掩码（causal masking）机制下，token（词元）可以访问打包序列中所有之前的 token。在上面的示例中，文件 4 里那段 Python 函数的某个 token 就能“看到”燕麦棒食谱、气候变化文章，以及任何其他被一起打包的内容。我们先快速瞥一眼典型的 4k 预训练上下文里都会有什么。一项[分析](https://www.harmdevries.com/post/context-length/)显示，CommonCrawl 和 GitHub 中大约 80–90% 的文件都短于 2k token。

下图考察了本文通篇使用的更新数据集的 token 分布：

在 FineWeb-Edu、DCLM、FineMath 和 Python-Edu 中，超过 80% 的文档不足 2k token。这意味着，在 2k 或 4k 的训练序列里使用标准因果掩码时，绝大多数 token 的计算量都花在关注无关的打包文档上。

尽管大多数基于网络的数据集由短文档组成，基于 PDF 的数据集却包含明显更长的内容。[FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) 的文档平均长度是网页文本的 2 倍，当与 FineWeb-Edu 和 DCLM 混合时还能提升性能。

除了计算效率低下，[Zhao et al. (2024)](https://doi.org/10.18653/v1/2024.acl-long.427) 还发现，这种方法会引入来自无关内容的噪声，从而损害性能。他们建议使用文档内掩码（_intra-document masking_）：修改注意力掩码，使 token 只能关注同一文档内的前文 token。下图可视化了这一差异：

[Zhu et al. (2025)](https://arxiv.org/abs/2503.15450) 在 SkyLadder 中也发现了文档内掩码的类似好处，但给出了不同解释：他们发现更短的上下文长度在训练时效果更好，而文档内掩码实际上降低了平均上下文长度。

![图 1：示意图](https://huggingfacetb-smol-training-playbook.hf.space/_astro/image_27c1384e-bcac-807c-807b-fac08be1d884.C286JbWA_AmD26.webp)

SkyLadder 的这些图表展示了多项发现：(a) 在预训练阶段，较短的上下文（shorter contexts）通常表现更好（验证集困惑度更低），(b) 文档内掩码（IntraDoc，intra-document masking）的困惑度低于随机打包（Random，random packing）和语义分组（BM25，semantic grouping），(c) 即使在没有位置编码（positional encoding）的情况下，较短上下文的优势依然存在，以及 (d) IntraDoc 使得有效上下文长度的分布向更短的方向倾斜。

Llama3（[Grattafiori 等，2024](https://arxiv.org/abs/2407.21783)）同样采用了文档内掩码进行训练，他们发现，在短上下文预训练阶段影响有限，但在长上下文扩展阶段收益显著，因为此时注意力开销（attention overhead）变得更为关键。此外，ProLong 论文（[Gao 等，2025](https://arxiv.org/abs/2410.02660)）表明，在持续预训练（continual pretraining）中利用文档掩码将 Llama3 8B 的上下文长度进行扩展，既提升了长上下文基准，也改善了短上下文基准。

我们决定在 1B 基线模型上进行消融实验（ablation），测试文档掩码是否会影响短上下文性能。配置文件见[此处](https://huggingface.co/datasets/HuggingFaceTB/training-guide-nanotron-configs/blob/main/doc_masking/doc_masking.yaml)。结果显示，与标准因果掩码（causal masking）相比，损失曲线和下游评估分数完全一致，如下图所示。

若要在 nanotron 中启用文档掩码，只需在模型配置中将该标志设为 `true`：

```
model_config:
  _attn_implementation: flash_attention_2
  _fused_rms_norm: true
  _fused_rotary_emb: true
- _use_doc_masking: false
+ _use_doc_masking: true
```

与 Llama3 类似，在短上下文任务中我们并未观察到明显影响，仅在 PIQA 上略有提升。然而，当扩展到长序列时，文档掩码（document masking） 对加速训练至关重要。这一点在我们将上下文从 4k 扩展到 64k token 的长上下文扩展中尤为关键（详见 [Training marathon](https://huggingfacetb-smol-training-playbook.hf.space/#the-training-marathon) 章节）。因此，我们在 SmolLM3 的整个训练过程中都采用了这一策略。

本节我们已介绍了注意力机制如何处理序列。接下来，让我们看看 transformer 中的另一个主要参数模块：嵌入（embeddings）。

#### [Embedding sharing（嵌入共享）](https://huggingfacetb-smol-training-playbook.hf.space/#embedding-sharing)

如果你查看我们基线消融（ablation）模型的 [config](https://huggingface.co/datasets/HuggingFaceTB/training-guide-nanotron-configs/blob/main/baseline_config_1B.yaml)，会发现与标准 transformer 的一个不同之处在于，通过 `tie_word_embeddings` 标志启用了嵌入共享。

LLM 有两个嵌入组件：输入嵌入（input embeddings）充当 token 到向量的查找表（大小为 $vocab\_size \times hidden\_dim$），以及输出嵌入（output embeddings），即最后一个线性层，将隐藏状态映射到词汇表 logits（$hidden\_dim \times vocab\_size$）。在经典情况下，这两者是独立的矩阵，因此嵌入参数总量为 $2 \times vocab\_size \times hidden\_dim$。于是，在小型语言模型中，嵌入可能占总参数量的很大比例，尤其在词汇表较大时。这使得嵌入共享（在输出层复用输入嵌入）成为小模型的自然优化手段。

更大的模型通常不使用该技术，因为嵌入在其参数预算中占比更小。例如，在不共享的情况下，总嵌入参数仅占 Llama3.2 8B 的 13%，以及 Llama3.1 70B 的 3%，如下饼图所示。

消融实验——共享嵌入的模型性能与更大但不共享的变体相当

接下来，我们将评估嵌入共享对我们消融模型的影响。我们借鉴 [MobileLLM](https://arxiv.org/abs/2402.14905) 在 125M 规模上对此技术的全面消融实验：他们证明共享可减少 11.8% 的参数，且精度下降极小。

由于取消嵌入绑定（untied embeddings）会将参数量从 1.2B 增加到 1.46B，我们将训练另一个同样取消绑定但层数更少的模型，使其总参数量仍与基线 1.2B 保持一致。我们会比较两个 1.2B 模型：采用绑定嵌入（tied embeddings）的基线（16 层）与取消绑定但层数更少（12 层）的版本，后者在相同参数预算下保持 1.2B；同时额外参考一个 1.46B 模型，它取消嵌入绑定且层数与基线相同（16 层）。你可以在[这里](https://huggingface.co/datasets/HuggingFaceTB/training-guide-nanotron-configs/blob/main/baseline_config_1B.yaml)找到 nanotron 配置。

损失与评估结果表明，尽管参数量少了 18%，我们的基线 1.2B 模型（绑定嵌入）在除 WinoGrande 外的所有基准上都与 1.46B 取消绑定版本表现相当。而参数量同为 1.2B 但取消嵌入绑定且层数减少（12 层 vs 16 层）的模型则表现最差，损失更高，下游评估分数也更低。这说明在相同参数预算下，增加模型深度比取消嵌入绑定带来的收益更大。

基于这些结果，我们在 SmolLM3 3B 模型中继续沿用绑定嵌入策略。

至此，我们已经探讨了嵌入共享策略及其权衡。然而，嵌入本身无法捕捉序列中 token 的顺序信息；提供这一信息是位置编码（positional encodings）的职责。下一节，我们将回顾位置编码策略的演进，从标准的 RoPE 到更新颖的 NoPE（No Position Embedding），后者能更高效地对长上下文进行建模。

#### [位置编码与长上下文](https://huggingfacetb-smol-training-playbook.hf.space/#positional-encodings--long-context)

当 Transformer 处理文本时，它们面临一个根本挑战：由于通过并行注意力（attention）操作一次性消费整个序列，模型天然没有词序概念。这使得训练高效，却也带来问题——如果没有显式的位置信息，在模型看来，“Adam beats Muon” 与 “Muon beats Adam” 几乎一样。

解决方案是位置嵌入（positional embeddings）：一种数学编码，为序列中的每个词元（token）赋予唯一的“地址”。然而，随着我们把上下文长度不断推升——从早期 BERT 的 512 个词元到如今百万词元级别——位置编码（positional encoding）的选择对性能与计算效率都变得愈发关键。

**位置编码的演进**

早期 Transformer 使用简单的绝对位置嵌入（Absolute Position Embeddings, APE）（[Vaswani et al., 2023](https://arxiv.org/abs/1706.03762)），本质上是可学习的查找表，将每个位置（1、2、3…）映射为一个向量，再加到词元嵌入上。这在短序列上表现良好，但有一个重大局限：模型的最大输入序列长度受限于训练时的最大长度，无法直接泛化到更长的序列。

领域随后转向相对位置编码（relative position encodings），它捕捉的是词元之间的距离，而非绝对位置。这更符合直觉：两个词相隔 3 个位置，比它们究竟位于 (5,8) 还是 (105,108) 更重要。

ALiBi（Attention with Linear Biases，线性偏置注意力）（[Press 等，2022](https://arxiv.org/abs/2108.12409)）特别之处在于，它根据 token 之间的距离修改注意力分数。两个 token 相距越远，其注意力就会通过施加在注意力权重上的简单线性偏置而受到越大的惩罚。如需查看 ALiBi 的详细实现，可访问此[资源](https://nn.labml.ai/transformers/alibi/index.html)。

但近期大型语言模型中最主流的技术是旋转位置编码（Rotary Position Embedding，RoPE）（[苏 等，2023](https://arxiv.org/abs/2104.09864)）。

**RoPE：将位置视为旋转**

RoPE 的核心洞见是将位置信息编码为高维空间中的旋转角度。它不是把位置向量加到 token 嵌入上，而是将 query 和 key 向量旋转一个与其绝对位置相关的角度。

其直观思路是：把嵌入中的每一对维度看作圆上的坐标，并按以下因素决定的角度进行旋转：

*   token 在序列中的位置
*   当前处理的是哪一对维度（不同维度对以不同频率旋转，这些频率是基准/参考频率的指数）

```python
import torch

def apply_rope_simplified(x, pos, dim=64, base=10000):
    """
    Rotary Position Embedding (RoPE)

    Idea:
    - Each token has a position index p (0, 1, 2, ...).
    - Each pair of vector dimensions has an index k (0 .. dim/2 - 1).
    - RoPE rotates every pair [x[2k], x[2k+1]] by an angle θ_{p,k}.

    
    Formula:
      θ_{p,k} = p * base^(-k / (dim/2))

    - Small k (early dimension pairs) → slow oscillations → capture long-range info.
    - Large k (later dimension pairs) → fast oscillations → capture fine detail.

    """
    rotated = []
    for i in range(0, dim, 2):
        k = i // 2  # index of this dimension pair
        # 频率项：k 越大 → 振荡越快
        inv_freq = 1.0 / (base ** (k / (dim // 2)))
        theta = pos * inv_freq  # 位置 p 与配对 k 的旋转角

        cos_t = torch.cos(torch.tensor(theta, dtype=x.dtype, device=x.device))
        sin_t = torch.sin(torch.tensor(theta, dtype=x.dtype, device=x.device))

        x1, x2 = x[i], x[i+1]

        # 应用二维旋转
        rotated.extend([x1 * cos_t - x2 * sin_t,
                        x1 * sin_t + x2 * cos_t])

    return torch.stack(rotated)

## Q, K: [batch, heads, seq, d_head]
Q = torch.randn(1, 2, 4, 8)
K = torch.randn(1, 2, 4, 8)

## 👉 在点积*之前*对 Q 和 K 应用 RoPE
Q_rope = torch.stack([apply_rope(Q[0,0,p], p) for p in range(Q.size(2))])
K_rope = torch.stack([apply_rope(K[0,0,p], p) for p in range(K.size(2))])

scores = (Q_rope @ K_rope.T) / math.sqrt(Q.size(-1))
attn_weights = torch.softmax(scores, dim=-1)
```

这段代码看起来可能比较复杂，所以我们用一个具体例子来拆解。考虑句子 _“The quick brown fox”_ 中的单词 _“fox”_。在我们的基线 1B 模型中，每个注意力头（attention head）处理的是 64 维的 query/key 向量。RoPE（Rotary Position Embedding，旋转位置编码）会把该向量拆成 32 对：(x₁, x₂)、(x₃, x₄)、(x₅, x₆)……之所以按“对”处理，是因为我们在二维空间里做旋转。为简单起见，只看第一对 (x₁, x₂)。单词 “fox” 在句中处于第 3 个位置，因此 RoPE 会把这对维度旋转：

$$
\text{rotation\_angle} = \text{position} \times \theta₀ 
                        = 3 \times \left(\frac{1}{10000^{0/32}}\right)
                        = 3 \times 1.0 
                        = 3.0 \text{ 弧度} 
                        = 172°
$$

基频（base frequency）是 10000，但对第一维对（k=0）而言指数为零，因此基频不影响计算（任何数的 0 次方都是 1）。下图可视化这一过程：

当两个 token 通过注意力交互时，“魔法”就出现了。它们旋转后表示的点积，直接通过旋转角的相位差编码了相对距离（其中 `m` 和 `n` 为 token 位置）：

$$
\text{dot\_product}(\text{RoPE}(x, m), \text{RoPE}(y, n)) = \sum_k [x_k \cdot y_k \cdot \cos((m-n) \cdot \theta_k)]
$$

注意力模式仅取决于 $(m-n)$，因此相隔 5 个位置的 token 无论处于序列的哪个绝对位置，其角度关系都相同。于是，模型学到的是基于距离的模式，可在序列的任何绝对位置生效，并能外推到更长的序列。

如何设置 RoPE 频率？

在实践中，大多数 LLM（大语言模型）预训练都从较短的上下文长度（2K–4K tokens）开始，使用几万量级的 RoPE（旋转位置编码）基础频率，如 10K 或 50K。一开始就使用非常长的序列进行训练会因注意力（attention）随序列长度呈二次方扩展而变得计算昂贵，并且长上下文数据（>4K 上下文长度的样本）也有限，这一点我们在 [Attention](https://huggingfacetb-smol-training-playbook.hf.space/#attention) 的文档掩码部分已经见过。研究还表明，这可能会损害短上下文性能（[Zhu et al., 2025](https://arxiv.org/abs/2503.15450)）。模型通常先学习单词之间的短程相关性，因此长序列帮助不大。典型的做法是先使用较短序列完成大部分预训练，然后进行持续预训练（continual pretraining），或在最后几千亿 tokens 上改用更长序列。然而，随着序列长度增加，与 token 位置成比例的旋转角度会变大，可能导致远距离 token 的注意力分数过快衰减（[Rozière et al., 2024](https://arxiv.org/abs/2308.12950)；[Xiong et al., 2023](https://arxiv.org/abs/2309.16039)）：

$$
\theta = \text{position} \times \frac{1}{\text{base}^{(k/(\text{dim}/2))}}
$$

解决方法是，在增加序列长度的同时提高基础频率，以防止这种衰减，可采用 ABF（Adjusted Base Frequency）和 YaRN（Yet another RoPE extensioN）等方法。

RoPE ABF（RoPE with Adjusted Base Frequency，调整基频的 RoPE）（[Xiong et al., 2023b](https://arxiv.org/abs/2309.16039)）：通过提高 RoPE 公式中的基频（base frequency）来解决长上下文中的注意力衰减问题。这一调整减缓了 token 位置之间的旋转角度，防止远距离 token 的注意力得分过快衰减。ABF 可一次性应用（直接提升频率），也可多阶段进行（随上下文增长逐步提升）。该方法实现简单，使嵌入向量分布更精细，便于模型区分远距离位置。尽管简单有效，ABF 在所有维度上的均匀缩放（uniform scaling）在极长上下文中可能并非最优。

YaRN（Yet another RoPE extensioN，又一种 RoPE 扩展）（[Peng et al., 2023](https://arxiv.org/abs/2309.00071)）：采用更精细的策略，通过斜坡或缩放函数对 RoPE 各维度进行非均匀频率插值。与 ABF 的均匀调整不同，YaRN 为不同频率分量应用不同缩放因子，以优化扩展后的上下文窗口。它还引入动态注意力缩放（dynamic attention scaling）和注意力 logits 的温度调整（temperature adjustment）等技术，在极大上下文尺寸下保持性能。YaRN 支持高效的“短训练、长测试”（train short, test long）策略，只需更少 token 和微调即可实现稳健外推。虽然比 ABF 复杂，YaRN 通常能在极长上下文中通过更平滑的缩放和缓解灾难性注意力丢失（catastrophic attention loss）带来更优的实证性能，也可仅在推理阶段使用，无需任何微调。

这些频率调整方法减缓了注意力分数衰减效应，并保留了远距离 token 的贡献。例如，Qwen3 的训练在序列长度从 4k 扩展到 32k 上下文时，通过 ABF（Attention with Linear Biases，线性偏置注意力）将频率从 10k 提升到 1M（随后团队应用 YaRN 达到 131k，即 4 倍外推）。请注意，目前尚无关于最优值的强共识，在上下文扩展阶段通常值得尝试不同的 RoPE（Rotary Position Embedding，旋转位置编码）值，以找到最适合你特定设置与评估基准的方案。

如今大多数主流模型都使用 RoPE：Llama、Qwen、Gemma 等。该技术已被证明在不同模型规模与架构（dense、MoE、Hybrid）下均表现稳健。让我们看看最近出现的几种 RoPE 变体。

**混合位置编码方法**

然而，随着模型向越来越大的上下文推进（[Meta AI, 2025](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)；[Yang et al., 2025](https://arxiv.org/abs/2501.15383)），即便是 RoPE 也开始面临性能挑战。在比“针里寻针”（Needle in the Haystack，NIAH）更具挑战性的长上下文基准（如 Ruler 和 HELMET）（[Hsieh et al., 2024](https://arxiv.org/abs/2404.06654)；[Yen et al., 2025](https://arxiv.org/abs/2410.02694)）上评估时，单纯增加 RoPE 频率的标准长上下文扩展方法存在局限。为此，研究者引入了更新的技术。

本节伊始我们提到，Transformer 需要位置信息才能理解 token 顺序，但最新研究对这一假设提出了挑战：如果显式位置编码终究并非必需呢？

NoPE（No Position Embedding，无位置嵌入）（[Kazemnejad et al., 2023](https://arxiv.org/abs/2305.19466)）在训练 Transformer 时完全不使用显式的位置编码，而是让模型通过因果掩码（causal masking）和注意力模式隐式地学习位置信息。作者表明，与 ALiBi 和 RoPE 相比，该方法在长度外推（length generalization）方面表现更佳。由于没有显式的位置编码可外推至训练长度之外，NoPE 天然地能够处理更长的上下文。然而在实际应用中，NoPE 模型在短上下文推理和知识任务上往往弱于 RoPE（[Yang et al](https://arxiv.org/pdf/2501.18795)）。这表明，尽管显式位置编码可能限制外推能力，但它们为训练上下文长度内的任务提供了有益的归纳偏置（inductive bias）。

RNoPE 混合方案： 鉴于上述权衡，[B. Yang et al. (2025)](https://arxiv.org/abs/2501.18795) 提出结合不同的位置编码策略或许值得探索。他们引入的 RNoPE 在模型中交替使用 RoPE 层和 NoPE 层。RoPE 层提供显式的位置信息，并对局部上下文施加近因偏置（recency bias）；NoPE 层则提升远距离信息检索能力。该技术近期已被用于 Llama4、Command A 和 SmolLM3。

为简化起见，本文后续将 RNoPE 简称为 “NoPE”。（在讨论中，人们常用 “NoPE” 指代 RNoPE）。

消融实验 —— NoPE 在短上下文上与 RoPE 持平

我们来测试混合 NoPE 方案。我们将纯 RoPE 1B 的消融基线与两种变体对比：一种在每第 4 层移除位置编码（NoPE 变体），另一种将 NoPE 与文档掩码（document masking）结合，以探究这些技术间的交互。核心问题是：能否在保持强短上下文性能的同时，获得长上下文能力？

损失与评估结果显示，三种配置的性能相近，表明 NoPE（No Position Embedding，无位置编码） 在保持强劲短上下文能力的同时，也为更好的长上下文处理奠定了基础。基于这些结果，我们在 SmolLM3 中采用了 NoPE + 文档掩码（document masking） 的组合。

**部分/分块 RoPE（Partial/Fractional RoPE）**：  
另一种互补思路是只对模型维度的一个子集应用 RoPE。与 RNoPE（Remove NoPE） 在整层之间交替使用 RoPE 和 NoPE 不同，Partial RoPE 在同一层内混合二者。近期模型如 GLM-4.5（[5 Team et al., 2025](https://arxiv.org/abs/2508.06471)）或 Minimax-01（[MiniMax et al., 2025](https://arxiv.org/abs/2501.08313)）采用了这一策略，但早在 gpt-j（[Wang & Komatsuzaki, 2021](https://github.com/kingoflolz/mesh-transformer-jax)）等旧模型中就已出现。此外，所有使用 MLA（Multi-head Latent Attention，多头隐式注意力） 的模型都必须采用 Partial RoPE，否则推理成本将高得难以接受。

MLA 通过投影吸收（projection absorption）实现高效推理：不再为每个头存储独立的键 $k_i^{(h)}$，而是缓存一个小的共享隐向量 $c_i = x_i W_c \in \mathbb{R}^{d_c}$，并将头的查询/键映射合并，使得每次打分都很廉价。令 $q_t^{(h)} = x_t W_q^{(h)}$ 与 $k_i^{(h)} = c_i E^{(h)}$，定义 $U^{(h)} = W_q^{(h)} E^{(h)}$，可得：

$$
s_{t,i}^{(h)} = \frac{1}{\sqrt{d_k}} \big(q_t^{(h)}\big)^\top k_i^{(h)} = \frac{1}{\sqrt{d_k}} \big(x_t U^{(h)}\big)^\top c_i
$$

于是只需用 $\tilde{q}_t^{(h)} = x_t U^{(h)} \in \mathbb{R}^{d_c}$ 与微小的缓存 $c_i$ 计算（无需存储每个头的 $k$）。RoPE 会破坏这一过程，因为它在两个映射之间插入了依赖位置的旋转：若使用全维度 RoPE，则……

$$
s_{t,i}^{(h)} = \frac{1}{\sqrt{d_k}}\bigl(x_t W_q^{(h)}\bigr)^\top
\underbrace{R_{t-i}}_{\text{depends on }t-i}\bigl(c_i E^{(h)}\bigr)
$$

因此，你无法将 $W_q^{(h)}$ 和 $E^{(h)}$ 预先合并成一个固定的 $U^{(h)}$。  
解决方案：部分 RoPE（Partial RoPE）。  
将头维度拆分为 $d_k = d_{\text{nope}} + d_{\text{rope}}$，在大块上不施加旋转（像以前一样吸收：$(x_t U_{\text{nope}}^{(h)})^\top c_i$），仅在小块上应用 RoPE。

**限制长上下文的注意力范围**

到目前为止，我们已经探讨了如何处理长上下文的位置信息：激活 RoPE、禁用它（NoPE）、仅在部分层上应用（RNoPE）或仅在某些隐藏维度上应用（Partial RoPE），或者调整其频率（ABF、YaRN）。这些方法通过修改模型如何编码位置，来处理比训练时更长的序列。但还有一种互补策略：与其调整位置编码，不如限制哪些 token 可以相互关注。

要理解为什么这很重要，考虑一个用 8 个 token 的序列预训练的模型。在推理时，我们想处理 16 个 token（超过训练长度）。位置 8–15 对模型的位置编码来说是分布外（out-of-distribution）的。虽然像 RoPE ABF 这样的技术通过调整位置频率来解决这个问题，但注意力范围（attention scope）方法采取了不同的路径：它们战略性地限制哪些 token 可以相互关注，在仍能处理整个序列的同时，将注意力模式保持在熟悉的范围内。这同时降低了计算成本和内存需求。下图比较了在预训练窗口为 8 的情况下，处理 16 个 token 序列的五种策略：

分块注意力（Chunked Attention） 将序列划分为固定大小的块（chunk），每个 token 只能关注自己所在块内的其他 token。在我们的示例中，16 个 token 被拆成两个 8 token 的块（0–7 和 8–15），每个 token 只能看到同一块内的其他 token。注意，token 8 到 15 完全无法回看前面的块。这样在块边界处形成隔离的注意力窗口并重置。Llama 4（[Meta AI, 2025](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)）在 RoPE 层（四分之三的解码层）使用 8192 token 的分块注意力，而 NoPE 层则保持对完整上下文的访问。通过限制每层的 KV 缓存大小，这降低了内存需求，但也意味着 token 无法关注之前的块，可能会影响某些长上下文任务。

滑动窗口注意力（Sliding Window Attention, SWA），由 Mistral 7B（[Child et al., 2019](https://huggingfacetb-smol-training-playbook.hf.space/#bib-child2019generating)；[Jiang et al., 2023](https://arxiv.org/abs/2310.06825)）推广，采用不同的思路：基于“最近的 token 最相关”的直觉。它没有硬性的块边界，而是让每个 token 只关注最近的 $N$ 个 token。在图中，每个 token 最多能看到往前 8 个位置，形成在序列上连续滑动的窗口。注意，token 15 可以关注位置 8–15，而 token 10 可以关注位置 3–10。窗口不断前移，在整个序列上保持局部上下文，而没有分块带来的人为屏障。Gemma 3 将 SWA 与全注意力交替使用，类似于混合位置编码策略将不同方法结合的做法。

双块注意力（Dual Chunk Attention, DCA）（[An et al., 2024](https://arxiv.org/abs/2402.17463)）是一种无需训练的方法，它在保留跨块信息流动的同时扩展了分块注意力（chunked attention）。在我们的示例中，使用块大小 $s=4$，将 16 个 token 分成 4 块（沿对角线可视化为 $4\times4$ 的方块）。DCA 结合了三种机制：(1) 块内注意力（Intra-chunk attention）：token 在其所属块内正常进行注意力计算（对角线模式）。(2) 块间注意力（Inter-chunk attention）：查询使用位置索引 $c-1=7$ 关注前面的块，生成被限制在 7 以内的相对位置。(3) 连续块注意力（Successive chunk attention）：使用局部窗口 $w=3$，保留相邻块之间的局部性。这样所有相对位置都保持在训练分布范围内（0 到 7），同时在块边界处实现平滑过渡。DCA 使 Qwen2.5 等模型在推理时无需对百万级 token 序列进行持续训练，即可支持最长 100 万 token 的上下文窗口。

在长上下文 Transformer 模型中会出现一个有趣现象：模型会给序列开头的 token 分配异常高的注意力分数，即使这些 token 在语义上并不重要。这种行为被称为注意力汇聚（attention sinks）（[Xiao et al.](https://arxiv.org/abs/2309.17453)）。这些初始 token 充当注意力分布的“汇聚点”，起到稳定注意力分布的作用。

其实用洞见在于：当上下文超过缓存大小时，只需保留最初几个 token 的 KV 缓存，再加上一个最近 token 的滑动窗口，就能在很大程度上恢复模型性能。这一简单修改使模型无需微调即可处理更长的序列，且不会降低性能。

现代实现以不同方式利用注意力汇聚（attention sinks）。原始研究建议在预训练期间添加一个专用的占位符标记（placeholder token），作为显式的注意力汇聚。最近，像 gpt-oss 这样的模型将注意力汇聚实现为可学习的每头偏置逻辑值（learned per-head bias logits），这些值被追加到注意力分数中，而不是作为输入序列中的实际标记。这种方法在不修改分词输入的情况下实现了相同的稳定化效果。

有趣的是，gpt-oss 还在注意力层本身使用了偏置单元（bias units），这是自 GPT-2 以来罕见的设计选择。虽然这些偏置单元通常被认为对标准注意力操作是冗余的（[Dehghani 等人](https://arxiv.org/pdf/2302.08626)的实证结果表明对测试损失影响极小），但它们可以实现注意力汇聚这一专门功能。关键洞察在于：无论实现为特殊标记、可学习偏置还是每头逻辑值，注意力汇聚都能在长上下文场景中为注意力分布提供稳定的“锚点”，使模型即使在上下文任意增长时也能存储关于整个序列的通用有用信息。

至此，我们已经涵盖了注意力的核心组件：平衡内存与计算的不同头配置（MHA、GQA、MLA），帮助模型理解标记顺序的位置编码策略（RoPE、NoPE 及其变体），以及使长上下文可处理的注意力范围技术（滑动窗口、分块和注意力汇聚）。我们还探讨了嵌入层应如何配置和初始化。这些架构选择定义了模型如何处理和表示序列。

但拥有正确的架构只是成功的一半。即使是设计良好的模型也可能在训练过程中出现不稳定，尤其是在大规模情况下。让我们来看看有助于保持训练稳定的技术。

#### [提升稳定性](https://huggingfacetb-smol-training-playbook.hf.space/#improving-stability)

现在让我们聚焦 LLM 预训练中最棘手的挑战之一：不稳定性（instability）。这些问题通常表现为损失尖峰（loss spikes）或训练损失的突然跳变，在大规模训练时尤为常见。

虽然我们将在 [Training Marathon](https://huggingfacetb-smol-training-playbook.hf.space/#the-training-marathon) 章节深入探讨不同类型的尖峰及其应对策略（涉及浮点精度、优化器和学习率），但某些架构（architectural）和训练技巧也能有效降低不稳定性，因此我们先在这里简要梳理。下文将介绍近期大规模训练（如 Olmo2（[OLMo et al., 2025](https://arxiv.org/abs/2501.00656)）和 Qwen3（[A. Yang, Li, et al., 2025](https://arxiv.org/abs/2505.09388)））中用于提升稳定性的几项简单技术：Z-loss、移除嵌入层权重衰减（removing weight decay from embeddings） 以及 QK-norm。

Z-loss

Z-loss（[Chowdhery et al., 2022](https://arxiv.org/abs/2204.02311)）是一种正则化（regularisation）技术，通过在损失函数中增加惩罚项，防止最终输出的 logits 过大。该正则化鼓励 softmax 分母保持在合理范围内，从而在训练过程中维持数值稳定性。

$$
\mathcal{L}_{\text{z-loss}} = \lambda \cdot \log^2(Z)
$$

下方在我们 1B 模型上的消融实验（ablation）表明，加入 Z-loss 既不会干扰训练损失，也不影响下游任务表现。对于 SmolLM3，我们最终没有启用它，因为当时的 Z-loss 实现引入了额外训练开销，而我们在开训前尚未完成针对性优化。

移除嵌入层权重衰减

权重衰减（weight decay）通常作为一种正则化技术应用于所有模型参数，但 [OLMo et al. (2025)](https://arxiv.org/abs/2501.00656) 发现将嵌入（embeddings）排除在权重衰减之外能够提升训练稳定性。其理由是，权重衰减会导致嵌入范数在训练过程中逐渐减小，而由于层归一化（layer normalization）的雅可比矩阵（Jacobian）与输入范数成反比，这可能会使早期层的梯度变大（[Takase et al., 2025](https://arxiv.org/abs/2312.16903)）。

我们通过训练三种配置来验证这一做法：基线配置使用标准权重衰减；第二种变体对嵌入不使用权重衰减；第三种配置则综合所有已采纳的改动（对嵌入不使用权重衰减 + NoPE + 文档掩码），以确保各项技术之间没有负面交互。三种配置的损失曲线和评估结果几乎完全一致。因此，我们在 SmolLM3 的训练中采纳了全部 3 项改动。

QKnorm

QK-norm（[Dehghani et al., 2023](https://arxiv.org/abs/2302.05442)）在计算注意力之前，对查询（query）和键（key）向量都应用层归一化。该技术有助于防止注意力 logits 过大，并被许多近期模型采用以提升稳定性。

然而，[B. Yang et al. (2025)](https://arxiv.org/abs/2501.18795) 发现 QK-norm 会损害长上下文任务的表现。他们的分析表明，QK-norm 会导致相关 token（needle）上的注意力质量降低，而不相关上下文上的注意力质量升高。他们认为，这是因为归一化操作从 query-key 点积中移除了幅度信息，使得注意力 logits 在幅度上更加接近。基于这一原因，我们未在 SmolLM3 中使用 QK-norm。此外，作为一款仅 3B 参数的小模型，它面临训练不稳定的风险也低于那些已证明 QK-norm 最有益的大模型。

#### [其他核心组件](https://huggingfacetb-smol-training-playbook.hf.space/#other-core-components)

除了我们已经介绍的组件外，还有几个架构决策值得补充说明，以确保完整性。

在参数初始化方面，现代模型通常使用截断正态初始化（mean=0，std=0.02 或 std=0.006）或类似 muP（[G. Yang & Hu, 2022](https://arxiv.org/abs/2011.14522)）的初始化方案，例如 Cohere 的 Command A（[Cohere et al., 2025](https://arxiv.org/abs/2504.00698)）。这也可以成为消融实验（ablation）的另一个主题。

在激活函数方面，SwiGLU 已成为现代大语言模型（LLM）的事实标准（除了 Gemma2 使用 GeGLU 以及 nvidia 使用 relu^2（[Nvidia et al., 2024](https://arxiv.org/abs/2406.11704)；[NVIDIA et al., 2025](https://arxiv.org/abs/2508.14444)）），取代了 ReLU 或 GELU 等旧有选择。

在更宏观的层面，架构布局的选择也会影响模型行为。尽管总参数量在很大程度上决定了语言模型的容量，但这些参数在深度（depth）和宽度（width）上的分布方式同样重要。[Petty et al. (2024)](https://arxiv.org/abs/2310.19956) 发现，在语言建模和组合任务上，更深的模型优于同等规模但更宽的模型，直到收益饱和。这种“深而窄”的策略在 MobileLLM 的消融实验（[Z. Liu et al., 2024](https://arxiv.org/abs/2402.14905)）中对十亿参数以下的 LLM 表现良好，而更宽的模型则因更高的并行度而倾向于提供更快的推理速度。现代架构在不同程度上反映了这种权衡，如这篇[博客文章](https://sebastianraschka.com/blog/2025/the-big-llm-architecture-comparison.html)所述。

我们已经涵盖了在你训练过程中最值得优化的稠密 Transformer（dense transformer）架构的核心要点。然而，近来又出现了一些针对模型整体的新型架构改进，即 MoE（Mixture of Experts，混合专家）与混合模型（hybrid models）。让我们先看看 MoE 能带来什么，从它开始。

#### [走向稀疏：MoE](https://huggingfacetb-smol-training-playbook.hf.space/#going-sparse-moe)

_Mixture-of-Experts（MoE，混合专家）_ 的直觉在于，我们并不需要为每个 token 预测都动用整个模型，就像人脑会根据当前任务（如视觉或运动）只激活对应区域一样。对 LLM 而言，这意味着当模型执行翻译任务时，那些专门学习过代码语法的部分无需参与。如果能做好这一点，就能在推理时只运行部分模型，从而节省大量算力。

从技术角度看，MoE 的目标很简单：在不增加每个 token“活跃”参数数量的前提下，扩大总参数量。简化来说，总参数影响模型的整体学习能力，而活跃参数决定了训练成本与推理速度。因此，如今许多前沿系统（如 DeepSeek V3、K2，以及闭源模型 Gemini、Grok 等）都采用了 MoE 架构。下图来自 Ling 1.5 论文（[L. Team et al., 2025](https://arxiv.org/abs/2503.05139)），对比了 MoE 与稠密模型的扩展规律：

![Image 2: Image](https://huggingfacetb-smol-training-playbook.hf.space/_astro/image_2931384e-bcac-80c4-ab02-f22c53e6fdee.dhphF60f_Z1Pe0ji.webp)

如果你是第一次接触 MoE，别担心，其机制并不复杂。我们先从标准稠密架构出发，再看转向 MoE 需要做哪些改动（下图由 [Sebastian Raschka](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/07_moe) 绘制）：

![Image 3: Image](https://huggingfacetb-smol-training-playbook.hf.space/_astro/image_2931384e-bcac-8062-bc11-d1ee3706d996.D6CeK-45_10lKF5.webp)

使用 MoE（Mixture of Experts，混合专家）时，我们将单个 MLP 替换为多个 MLP（称为“experts，专家”），并在 MLP 之前添加一个可学习的 router（路由器）。对于每个 token，router 只选择一小部分专家执行。这就是“总参数”与“活跃参数”区别的来源：模型拥有大量专家，但任意给定 token 仅使用其中少数。

设计一个 MoE 层会引出几个核心问题：

*   Expert shape & sparsity（专家规模与稀疏度）：应该使用许多小专家，还是少量大专家？每个 token 应激活多少专家？总共需要多少专家（即稀疏度或“top-k”）？是否应让某些专家始终激活，成为通用专家？
*   Utilization & specialization（利用率与专业化）：如何挑选被路由的专家，并确保它们被充分利用（避免闲置容量），同时仍鼓励其专业化？实践中这是一个负载均衡问题，对训练与推理效率影响显著。

本文聚焦一个目标：在给定固定计算预算下，如何选择 MoE 配置以最小化 loss（损失）？这与纯系统效率（throughput/latency，吞吐/延迟）不同，后者我们稍后再谈。本节大部分内容参考了蚂蚁集团 MoE scaling laws 论文（[Tian et al., 2025](https://arxiv.org/abs/2507.17702)）。

我们将采用他们提出的 Efficiency Leverage（EL，效率杠杆） 概念。简言之，EL 衡量“需要多少 dense（稠密）计算才能匹配某 MoE 设计达到的 loss”，单位为 FLOPs。EL 越高，意味着该 MoE 配置每单位计算带来的 loss 改善优于 dense 训练。

下面深入探讨如何设置 MoE 的稀疏度以提升效率杠杆。

Sparsity / activation ratio（稀疏度 / 激活比例）

> TL;DR： 稀疏度越高 → FLOPs 效率越好 → 极高稀疏度时收益递减 → 最佳点取决于你的计算预算。

在本节中，我们想找出哪种 MoE（Mixture of Experts，混合专家）配置最佳。从渐近角度看，很容易发现两个极端都不是理想设置。一方面，始终激活所有专家会让我们回到稠密（dense）模式，即所有参数始终被使用。另一方面，如果活跃参数非常少（极端情况下只激活 1 个参数），显然即使在狭窄领域也不足以解决任务。因此，我们显然需要找到某种中间地带。

在深入寻找最优配置之前，先定义两个量会很有用：激活比例（activation ratio） 及其倒数 稀疏度（sparsity）：

$$
\begin{aligned}
\text{activation ratio} &= \frac{\#\text{activated experts}}{\#\text{total experts}} \\[6pt]
\text{sparsity} &= \frac{\#\text{total experts}}{\#\text{activated experts}} = \frac{1}{\text{activation ratio}}
\end{aligned}
$$

从计算角度看，成本仅由活跃参数驱动。如果保持激活专家的数量（和大小）不变，同时增加专家总数，你的推理/训练 FLOPs 预算大致保持不变，但你增加了模型容量，因此只要训练时间足够长，模型通常会变得更好。

如果你梳理最近的 MoE 论文，会发现一些有趣的实证结论：在保持活跃专家的数量和大小不变的情况下，增加专家总数（即降低激活比例 / 提高稀疏度）会降低损失，但当稀疏度非常高时，收益会递减。

两个例子：

*   Kimi K2 图表（[K. Team et al., 2025](https://arxiv.org/abs/2507.20534)）：同时展示了这两种效应：更高的稀疏度提升性能，但随着稀疏度继续增大，收益逐渐减弱。
*   蚂蚁集团图表（[Tian et al., 2025](https://arxiv.org/abs/2507.17702)）：与 K2 结论一致，并额外指出，更高稀疏度的 MoE 从增加算力中获益更多。

![Image 4: Image](https://huggingfacetb-smol-training-playbook.hf.space/_astro/Capture_decran_2025-10-20_a_13_25_47_2921384e-bcac-8087-83e5-fa7a40c1f342.asYkEXKU_1s8wtB.webp)

![Image 5: Image](https://huggingfacetb-smol-training-playbook.hf.space/_astro/Capture_decran_2025-10-20_a_13_26_08_2921384e-bcac-80b5-ac36-fb73d6374208.D-BBIjb7_Zs7nQa.webp)

以下是若干混合专家模型稀疏度的对比表格：

| 模型名称 | 专家总数 | 每令牌激活数（含共享） | 稀疏度 |
| --- | --- | --- | --- |
| Mixtral-8×7B | 8 | 2 | 4.0 |
| Grok-1 | 8 | 2 | 4.0 |
| Grok-2 | 8 | 2 | 4.0 |
| OLMoE-1B-7B-0924 | 64 | 8 | 8.0 |
| gpt-oss 20b | 32 | 4 | 8 |
| Step-3 | 48路由+1共享=49 | 3路由+1共享=4 | 12.25 |
| GLM-4.5-Air | 128路由+1共享=129 | 8路由+1共享=9 | 14.3 |
| Qwen3-30B-A3B | 128 | 8 | 16.0 |
| Qwen3-235B-A22B | 128 | 8 | 16.0 |
| GLM-4.5 | 160路由+1共享=161 | 8路由+1共享=9 | 17.8 |
| DeepSeek-V2 | 160路由+2共享=162 | 6路由+2共享=8 | 20.25 |
| DeepSeek-V3 | 256路由+1共享=257 | 8路由+1共享=9 | 28.6 |
| gpt-oss 120b | 128 | 4 | 32 |
| Kimi K2 | 384路由+1共享=385 | 8路由+1共享=9 | 42.8 |
| Qwen3-Next-80B-A3B-Instruct | 512路由+1共享=513 | 10总激活+1共享=11 | 46.6 |

当前趋势非常明显：混合专家模型正朝着更高稀疏度的方向发展。不过最优稀疏度仍取决于硬件条件与端到端效率。例如Step-3模型以峰值效率为目标，特意未追求极限稀疏度以适应其特定硬件与带宽限制；而gpt-oss-20b因受终端设备内存限制（未激活专家仍会占用部分内存），其稀疏度设置相对较低。

粒度

除了稀疏性（sparsity）之外，我们还需要决定每个专家（expert）应该有多大。这由“粒度”（granularity）这一指标来衡量，该指标由蚂蚁集团（Ant Group）提出。我们先明确这个术语的含义。不同论文中的术语略有差异，有些使用了稍有不同的公式。这里我们采用与所引用图表一致的定义：

$$
G=\frac{\alpha \cdot d_{\text{model}}}{d_{\text{expert}}} \quad \text{with } \alpha = 2 \text{ or } 4
$$

较高的粒度值意味着在（固定参数总量的情况下）使用更多但维度更小的专家。该指标是专家维度（$d_{\text{expert}}$）与模型维度（$d_{\text{model}}$）的比值。

在稠密（dense）模型中，一个常见的经验法则是将 MLP 的维度设为 $d_{\text{intermediate}} = 4 \cdot d_{\text{model}}$。如果 $\alpha=4$（如 [Krajewski et al. (2024)](https://arxiv.org/abs/2402.07871)），你可以粗略地把粒度理解为需要多少个专家才能与稠密 MLP 的宽度相匹配（$4\, d_{\text{model}} = d_{\text{intermediate}} = G\, d_{\text{expert}}$）。

这种解释只是一个粗略的启发式方法：现代 MoE 设计通常分配的总容量远大于单个稠密 MLP，因此一对一的对应关系在实践中并不成立。蚂蚁团队的设置选择 $\alpha=2$，这仅仅是一种不同的归一化方式。为了保持一致，我们将采用这一约定并沿用下去。

下面是一张表格，列出了部分已发布 MoE 模型的不同取值：

| 模型 | $d_{\text{model}}$ | $d_{\text{expert}}$ | $G = 2 d_{\text{model}} / d_{\text{expert}}$ | 年份 |
| --- | --- | --- | --- | --- |
| Mixtral-8×7B | 4,096 | 14,336 | 0.571 | 2023 |
| gpt-oss-120b | 2880 | 2880 | 0.5 | 2025 |
| gpt-oss-20b | 2880 | 2880 | 0.5 | 2025 |
| Grok 2 | 8,192 | 16,384 | 1.0 | 2024 |
| StepFun Step-3 | 7,168 | 5,120 | 2.8 | 2025 |
| OLMoE-1B-7B | 2,048 | 1,024 | 4.0 | 2025 |
| Qwen3-30B-A3B | 2,048 | 768 | 5.3 | 2025 |
| Qwen3-235B-A22B | 4,096 | 1,536 | 5.3 | 2025 |
| GLM-4.5-Air | 4,096 | 1,408 | 5.8 | 2025 |
| DeepSeek V2 | 5,120 | 1,536 | 6.6 | 2024 |
| GLM-4.5 | 5,120 | 1,536 | 6.6 | 2025 |
| Kimi K2 | 7,168 | 2,048 | 7.0 | 2025 |
| DeepSeek V3 | 7168 | 2048 | 7.0 | 2024 |
| Qwen3-Next-80B-A3B | 2048 | 512 | 8.0 | 2025 |

让我们聊聊“粒度（granularity）”如何塑造行为（摘自[蚂蚁集团的论文](https://arxiv.org/pdf/2507.17702)）：

![Image 6: Image](https://huggingfacetb-smol-training-playbook.hf.space/_astro/75ae60ff-50be-48e1-aad2-a8fc56120d3d_2921384e-bcac-80c2-984b-d81404e4bb7c.8nxVv-OC_25NHa.webp)

粒度看起来并不是EL（expected loss，期望损失）的主要驱动因素——它有帮助，尤其是在超过2之后，但并不是决定损失的主导因子。不过存在一个甜蜜点：把粒度继续推高，收益会在某一点后趋于平缓。因此，粒度是一个有用的调参旋钮，在近期的发布中明显呈现“越大越好”的趋势，但不应孤立地单独优化。

另一种被广泛用于改进MoE（Mixture of Experts，混合专家）的方法是“共享专家（shared experts）”的概念。让我们来看一看！

Shared experts（共享专家）

共享专家（shared-expert）配置会把每个 token 都路由到一小部分始终激活的专家。这些共享专家吸收数据中基础且反复出现的模式，从而让其余专家能够更激进地专业化。实践中，你通常不需要很多共享专家；模型设计者一般选 1 个，最多 2 个。随着粒度增加（例如从类似 Qwen3 的设置过渡到更接近 Qwen3-Next 的配置），共享专家往往变得更有用。从下图可见，整体影响较为温和，并不会显著改变 EL（Expert Load）。一条简单的经验法则在大多数情况下都适用：只用 1 个共享专家即可，这与 DeepSeek V3、K2 和 Qwen3-Next 等模型的选择一致，能在不引入不必要复杂度的前提下最大化效率。下图引自 [Tian et al. (2025)](https://arxiv.org/abs/2507.17702)。

![Image 7: Image](https://huggingfacetb-smol-training-playbook.hf.space/_astro/image_2931384e-bcac-80c4-ab02-f22c53e6fdee.dhphF60f_Z1Pe0ji.webp)

因此，共享专家就是“某些 token 永远会被路由通过”的专家。那其他专家呢？我们该如何学习何时把 token 路由给哪个专家，并确保不会只用到少数几个？接下来我们将讨论负载均衡（load balancing），它正是解决这一问题的关键。

负载均衡

负载均衡是 MoE（Mixture of Experts）中的核心环节。如果设置不当，它会毁掉所有其他设计选择。通过下面这个例子，我们可以看到糟糕的负载均衡为何会带来巨大麻烦。考虑一个非常简单的分布式训练场景：我们有 4 张 GPU，把模型的 4 个专家均匀分到每张卡上。如果路由崩溃，所有 token 都被路由到专家 1，那就意味着只有 1/4 的 GPU 被利用，这对训练和推理效率都极其不利。此外，模型的有效学习容量也会下降，因为并非所有专家都被激活。

为了解决这一问题，我们可以在路由（router）上增加一个额外的损失项。下面给出了标准的基于辅助损失的负载均衡（auxiliary loss–based load balancing，LBL）公式：

$$
\mathcal{L}_{\text{Bal}} = \alpha \sum_{i=1}^{N_r} f_i\,P_i
$$

这个简单公式只用到三个因子：系数 $\alpha$ 决定损失的强度；$f_i$ 是流量比例（traffic fraction），即流经专家（expert）i 的 token 占比；最后是 $P_i$，即概率质量（probability mass），简单地对流经该专家的 token 概率求和。二者缺一不可：$f_i$ 对应实际的均衡，而 $P_i$ 平滑且可微，使梯度能够流动。若实现完美负载均衡，则有 $f_i=P_i=1/N_r$。不过，我们需要小心调节 $\alpha$：值太小会无法充分引导路由，值太大则路由均匀性会比主语言模型损失（language model loss）更重要。

也可以在不引入显式损失项的情况下实现均衡。DeepSeek v3（[DeepSeek-AI et al., 2025](https://arxiv.org/abs/2412.19437)）在送入路由 softmax 的亲和度得分（affinity scores）上增加了一个简单的偏置项。如果某个路由过载，就把得分稍微降低（一个常数因子 $\gamma$），使其被选中的概率变小；若专家利用率不足，则增加 $\gamma$。通过这一简单的自适应规则，他们同样实现了负载均衡。

一个关键细节是计算路由统计量的作用域（scope）：$f_i$ 和 $P_i$ 是在本地批次（local batch，每个 worker 的 mini-batch）内计算，还是在全局（global，跨 worker/设备聚合）计算？Qwen 团队的分析（[Qiu et al., 2025](https://arxiv.org/abs/2501.11873)）指出，当每个本地批次中的 token 多样性不足时，本地计算会损害专家特化（expert specialization，衡量路由健康度的良好代理指标）以及整体模型性能。专家特化指的是一个或多个专家在特定领域被更频繁激活的现象。换句话说，如果本地批次过于狭窄，其路由统计量就会变得嘈杂/有偏，无法实现良好的负载均衡。这意味着我们应尽可能使用全局统计量（或至少跨设备聚合）。值得注意的是，在该论文发表时，许多框架——包括 Megatron——默认都在本地计算这些统计量。

下图来自 Qwen 论文，展示了 micro-batch 与 global batch 聚合方式的差异及其对性能和特化的影响：

![Image 8: Image](https://huggingfacetb-smol-training-playbook.hf.space/_astro/Capture_decran_2025-10-21_a_15_34_27_2931384e-bcac-8066-834b-c485ae8d1fa5.Cix986wE_ZLIDik.webp)

总体而言，对 MoE 的架构选择进行消融（ablating）颇为棘手，因为诸多因素相互交织。例如，共享专家（shared expert） 的有用性可能取决于模型的粒度（granularity）。因此，值得花些时间设计一组扎实的实验，真正获得你想要的洞察！

至此，我们已经覆盖了 MoE 的基础知识，但仍有更多内容值得探索。以下是一份非穷尽的研究清单：

*   零计算专家（Zero-computation experts）、MoE 层重缩放（MoE layer rescaling）与训练监控（LongCat-Flash 论文）。

*   正交损失负载均衡（Orthogonal loss load balancing，如 ERNIE 4.5 中所用）。

*   在训练过程中调度负载均衡系数（Scheduling the load-balancing coefficient over training）。

*   与 MoE（混合专家模型，Mixture-of-Experts）相关的架构/优化交互，例如：

    *   优化器排名是否因 MoE 而改变。
    *   如何将 MuP（最大更新参数化，Maximal Update Parametrization）应用于 MoE。
    *   如何为 MoE 调整学习率（因为它们在每个批次中看到的 token 数量不同）。

*   起始密集层的数量。

*   还有很多……

我们把这些留给渴望深入的你，亲爱的读者，去继续沿着兔子洞一探究竟；现在，我们将转向最后一个重大架构选择：混合模型！

#### [拓展：混合模型（Hybrid Models）](https://huggingfacetb-smol-training-playbook.hf.space/#excursion-hybrid-models)

最近的趋势是在标准稠密（dense）或混合专家（MoE, Mixture-of-Experts）架构中引入状态空间模型（SSM, State Space Models）或线性注意力机制（linear attention mechanisms）（[MiniMax et al., 2025](https://arxiv.org/abs/2501.08313)；[Zuo et al., 2025](https://arxiv.org/abs/2507.22448)）。这些新模型试图解决 Transformer 的一些根本弱点：高效处理超长上下文。它们在循环模型（recurrent models）与 Transformer 之间取折中——前者可线性扩展、高效处理任意长度上下文，但难以充分利用上下文信息；后者在长上下文下代价高昂，却能很好地捕捉上下文模式。

已有研究（[Waleffe et al., 2024](https://arxiv.org/abs/2406.07887)) 探讨了 Mamba 模型（一种 SSM）的弱点，发现其在许多基准上表现良好，但在 MMLU 上却逊色，推测原因在于缺乏上下文内学习（in-context learning）能力。因此，人们将 SSM 块与稠密或 MoE 块结合，以兼得两者之长，故名“混合模型”。

线性注意力方法的核心思想是重排计算顺序，使注意力代价不再是 O(n²d)，从而在长上下文下依然可行。具体怎么做？先回顾推理时的注意力公式。为第 t 个 token 生成输出：

$$
\mathbf{o}_{t}=\sum_{j=1}^{t} \frac{\exp\!\bigl(\mathbf{q}_{t}^{\top}\mathbf{k}_{j}\bigr)\,\mathbf{v}_{j}}{\sum_{l=1}^{t}\exp\!\bigl(\mathbf{q}_{t}^{\top}\mathbf{k}_{l}\bigr)}
$$

现在去掉 softmax：

$$
o_t = \sum_{j=1}^{t} (q_t^\top k_j)\, v_j
$$

重排后得到：

$$
\sum_{j=1}^{t}(q_t^\top k_j)\,v_j = \Bigl(\sum_{j=1}^{t} v_j k_j^\top\Bigr) q_t.
$$

定义运行状态（running state）：

$$
S_t \triangleq \sum_{j=1}^{t} k_j v_j^\top = K_{1:t}^\top V_{1:t} \in \mathbb{R}^{d\times d}
$$

其简单更新方式为：

$$
S_t = S_{t-1} + k_t v_t^\top
$$

于是可写成：

$$
o_t = S_t q_t = S_{t-1} q_t + v_t (k_t^\top q_t)
$$

为何重排（reordering）如此重要：左侧形式 $\sum_{j\le t}(q_t^\top k_j)v_j$ 的含义是“对每个过去的 token j，计算点积 $q_t^\top k_j$（一个标量），用它缩放 $v_j$，再把这 t 个向量相加”——在第 t 步需要约 $O(t d)$ 的计算量。右侧形式将其重写为 $\bigl(\sum_{j\le t} v_j k_j^\top\bigr) q_t$：你只需维护一个运行状态矩阵 $S_t=\sum_{j\le t} v_j k_j^\top\in\mathbb{R}^{d\times d}$，它已经汇总了所有过去的 $(k_j,v_j)$。每遇到一个新 token，就用一次外积 $v_t k_t^\top$ 更新它，代价 $O(d^2)$，然后输出只需一次矩阵–向量乘法 $S_t q_t$（再花 $O(d^2)$）。因此，从头开始用左侧形式生成 T 个 token 的复杂度是 $O(T^2 d)$，而维护 $S_t$ 并使用右侧形式只需 $O(T d^2)$。直观地说：左侧 = “每步多次小规模点积–缩放–相加”；右侧 = “一次预先汇总的矩阵乘以查询”，把对序列长度的依赖换成了对维度的依赖。本文聚焦推理（inference）与递归形式（recurring form），但在训练（training）中它同样更高效，重排只需如下方程：

$$
\underset{n\times n}{(QK^\top)}\,V = Q\,\underset{d\times d}{(K^\top V)}
$$

可以看出，这看起来已经非常类似于 RNN（循环神经网络）的结构。这样我们的问题就解决了吗？差不多。但在实践中，softmax 起到了重要的稳定作用，而朴素的线性形式如果没有某种归一化可能会不稳定。这就催生了一个实用的变体——闪电注意力（Lightning Attention） 或 范数注意力（Norm Attention）！

Lightning and norm attention（闪电注意力与范数注意力）

这一家族出现在 Minimax01（[MiniMax et al., 2025](https://arxiv.org/abs/2501.08313)）以及更近的 Ring-linear（[L. Team, Han, et al., 2025](https://arxiv.org/abs/2510.19338)）中，基于 Norm Attention（范数注意力）的思想（[Qin et al., 2022](https://arxiv.org/abs/2210.10340)）。关键步骤很简单：对输出做归一化。“Lightning” 变体专注于让实现更快更高效，并让公式略有不同。两者的公式如下：

NormAttention（范数注意力）：

$$
\text{RMSNorm}(Q(K^TV))
$$

LightningAttention（闪电注意力）：

$$
Q= \text{Silu(Q)}, \; K = \text{Silu(K)}, \; V = \text{Silu(V)}
$$

$$
O = \text{SRMSNorm}(Q(KV^T))
$$

根据 Minimax01 的实验，采用 Norm Attention 的混合模型在大多数任务上都能与 softmax 注意力打成平手。

有趣的是，在诸如 Needle in a Haystack（NIAH，草垛寻针） 这样的检索任务上，它的表现可以远远优于完整的 softmax 注意力，这看似令人惊讶，但或许暗示了当 softmax 与线性层协同工作时，存在某种协同效应！

令人意外的是，刚刚发布的 MiniMax M2 并未采用混合（hybrid）或线性注意力（linear attention）。据其 [预训练负责人](https://huggingfacetb-smol-training-playbook.hf.space/[https://x.com/zpysky1125/status/1983383094607347992](https://x.com/zpysky1125/status/1983383094607347992)) 透露，虽然早期 MiniMax M1 在较小规模、当时流行的基准（MMLU、BBH、MATH）上尝试 Lightning Attention 时表现亮眼，但在更大规模上却发现它在“复杂的多跳推理任务”中存在明显缺陷。他们还指出，RL 训练期间的数值精度问题以及基础设施成熟度是主要阻碍。他们总结道：在大规模下做架构设计是一个多变量问题，既困难又算力密集，因为它对数据分布、优化器等其他参数极其敏感……

不过他们也承认，“随着 GPU 算力增长放缓而序列长度持续增加，线性和稀疏注意力（linear and sparse attention）的优势将逐步显现。” 这再次凸显了架构消融（architecture ablations）的复杂性，以及研究与生产现实之间的差距。

现在，让我们看看更多这类方法，并借助一个统一框架来理解它们。

高级线性注意力（Advanced linear attention）

循环模型的一条宝贵经验是：让状态偶尔“放下过去”。在实践中，这意味着为前一状态引入一个门控（gate） $\mathbf{G}_t$：

$$\mathbf{S}_t \;=\; \mathbf{G}_t \odot \mathbf{S}_{t-1} \;+\; \mathbf{v}_t \mathbf{k}_t^{\top}$$

几乎所有最新的线性注意力方法都包含这种门控组件，只是 $\mathbf{G}_t$ 的实现方式各异。以下是 [该论文](https://huggingfacetb-smol-training-playbook.hf.space/arxiv.org/abs/2312.06635) 列出的不同门控变体及其对应架构：

| Model | Parameterization | Learnable parameters |
| --- | --- | --- |
| Mamba ([A. Gu & Dao, 2024](https://arxiv.org/abs/2312.00752)) | $$\mathbf{G}_t = \exp(-(\mathbf{1}^\top \boldsymbol{\alpha}_t) \odot \exp(\mathbf{A})), \quad \boldsymbol{\alpha}_t = \text{softplus}(\mathbf{x}_t \mathbf{W}_{\alpha_1} \mathbf{W}_{\alpha_2})$$ | $$\mathbf{A} \in \mathbb{R}^{d_k \times d_v}, \quad \mathbf{W}_{\alpha_1} \in \mathbb{R}^{d \times \frac{d}{16}}, \quad \mathbf{W}_{\alpha_2} \in \mathbb{R}^{\frac{d}{16} \times d_v}$$ |
| Mamba-2 ([Dao & Gu, 2024](https://arxiv.org/abs/2405.21060)) | $$\mathbf{G}_t = \gamma_t \mathbf{1}^\top \mathbf{1}, \quad \gamma_t = \exp(-\text{softplus}(\mathbf{x}_t \mathbf{W}_{\gamma})\exp(a))$$ | $$\mathbf{W}_{\gamma} \in \mathbb{R}^{d \times 1}, \quad a \in \mathbb{R}$$ |
| mLSTM ([Beck et al., 2025](https://arxiv.org/abs/2503.14376); H. [Peng et al., 2021](https://arxiv.org/abs/2103.02143)) | $$\mathbf{G}_t = \gamma_t \mathbf{1}^\top \mathbf{1}, \quad \gamma_t = \sigma(\mathbf{x}_t \mathbf{W}_{\gamma})$$ | $$\mathbf{W}_{\gamma} \in \mathbb{R}^{d \times 1}$$ |
| Gated Retention ([Sun et al., 2024](https://arxiv.org/abs/2405.05254)) | $$\mathbf{G}_t = \gamma_t \mathbf{1}^\top \mathbf{1}, \quad \gamma_t = \sigma(\mathbf{x}_t \mathbf{W}_{\gamma})^{\frac{1}{\tau}}$$ | $$\mathbf{W}_{\gamma} \in \mathbb{R}^{d \times 1}$$ |
| DFW (Mao, 2022; Pramanik et al., 2023) ([Mao, 2022](https://arxiv.org/abs/2210.04243)) | $$\mathbf{G}_t = \boldsymbol{\alpha}_t^\top \boldsymbol{\beta}_t, \quad \boldsymbol{\alpha}_t = \sigma(\mathbf{x}_t \mathbf{W}_{\alpha}), \quad \boldsymbol{\beta}_t = \sigma(\mathbf{x}_t \mathbf{W}_{\beta})$$ | $$\mathbf{W}_{\alpha} \in \mathbb{R}^{d \times d_k}, \quad \mathbf{W}_{\beta} \in \mathbb{R}^{d \times d_v}$$ |
| GateLoop ([Katsch, 2024](https://arxiv.org/abs/2311.01927)) | $$\mathbf{G}_t = \boldsymbol{\alpha}_t^\top \mathbf{1}, \quad \boldsymbol{\alpha}_t = \sigma(\mathbf{x}_t \mathbf{W}_{\alpha_1})\exp(\mathbf{x}_t \mathbf{W}_{\alpha_2} \mathrm{i})$$ | $$\mathbf{W}_{\alpha_1} \in \mathbb{R}^{d \times d_k}, \quad \mathbf{W}_{\alpha_2} \in \mathbb{R}^{d \times d_k}$$ |
| HGRN-2 ([Qin et al., 2024](https://arxiv.org/abs/2404.07904)) | $$\mathbf{G}_t = \boldsymbol{\alpha}_t^\top \mathbf{1}, \quad \boldsymbol{\alpha}_t = \gamma + (1-\gamma)\sigma(\mathbf{x}_t \mathbf{W}_{\alpha})$$ | $$\mathbf{W}_{\alpha} \in \mathbb{R}^{d \times d_k}, \quad \gamma \in (0,1)^{d_k}$$ |
| RWKV-6 ([B. Peng et al., 2024](https://arxiv.org/abs/2404.05892)) | $$\mathbf{G}_t = \boldsymbol{\alpha}_t^\top \mathbf{1}, \quad \boldsymbol{\alpha}_t = \exp(-\exp(\mathbf{x}_t \mathbf{W}_{\alpha}))$$ | $$\mathbf{W}_{\alpha} \in \mathbb{R}^{d \times d_k}$$ |
| Gated Linear Attention (GLA) | $$\mathbf{G}_t = \boldsymbol{\alpha}_t^\top \mathbf{1}, \quad \boldsymbol{\alpha}_t = \sigma(\mathbf{x}_t \mathbf{W}_{\alpha_1} \mathbf{W}_{\alpha_2})^{\frac{1}{\tau}}$$ | $$\mathbf{W}_{\alpha_1} \in \mathbb{R}^{d \times 16}, \quad \mathbf{W}_{\alpha_2} \in \mathbb{R}^{16 \times d_k}$$ |

近期模型的门控线性注意力（Gated linear attention）形式，其差异主要体现在 $\mathbf{G}_t$ 的参数化上。偏置项已省略。

其中值得注意的一个变体是 Mamba-2（[Dao & Gu, 2024](https://arxiv.org/abs/2405.21060)），它被广泛应用于许多混合模型，如 Nemotron-H（[NVIDIA, :, Blakeman, et al., 2025](https://arxiv.org/abs/2504.03624)）、Falcon H1（[Zuo et al., 2025](https://arxiv.org/abs/2507.22448)）以及 Granite-4.0-h（[IBM Research, 2025](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)）。

然而，目前仍处于早期阶段，在扩展到大型混合模型（hybrid model）时，还需考虑一些重要的细微差别。尽管它们展现出潜力，MiniMax 在 [M2](https://x.com/zpysky1125/status/1983383094607347992) 上的经验表明，小规模的优势并不总能迁移到大规模生产系统，尤其是在复杂推理任务、强化学习（RL）训练稳定性以及基础设施成熟度方面。话虽如此，混合模型发展迅速，依然是前沿训练中的可靠选择。Qwen3-Next（采用门控 DeltaNet 更新）（[Qwen Team, 2025](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)）报告称，其在长上下文推理速度更快、训练更快，并在常规基准测试中表现更强。我们也期待 Kimi 的下一款模型，该模型极有可能采用他们新的 [“Kimi Delta Attention”](https://github.com/fla-org/flash-linear-attention/pull/621)。此外，还要提到稀疏注意力（Sparse Attention），它通过选择块或查询来计算注意力，从而解决与线性注意力相同的长上下文问题。一些例子包括 Native Sparse Attention（[Yuan et al., 2025](https://arxiv.org/abs/2502.11089)）、DeepSeek Sparse Attention（[DeepSeek-AI, 2025](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf)）和 InfLLM v2（[M. Team, Xiao, et al., 2025](https://arxiv.org/abs/2506.07900)）。

在进入分词器（tokenizer）之前，我们将通过构建一个小型决策树来总结架构选择，以确定是训练稠密（dense）模型、MoE 模型还是混合模型。

#### [To MoE or not MoE：选择基础架构](https://huggingfacetb-smol-training-playbook.hf.space/#to-moe-or-not-moe-choosing-a-base-architecture)

我们已经了解了稠密（dense）、混合专家（MoE, Mixture of Experts）和混合（hybrid）模型，你可能自然想知道该用哪一种？架构选择通常取决于模型部署位置、团队经验以及时间线。我们快速回顾每种架构的优缺点，并给出一个简单的决策流程，帮你找到合适的架构。

Dense transformers（稠密 Transformer）  

标准的纯解码器 Transformer，每个 token 都会激活所有参数。数学推导见 [The Annotated Transformers](https://nlp.seas.harvard.edu/2018/04/03/attention.html)，直观理解可参考 [The Illustrated Transformers](https://jalammar.github.io/illustrated-transformer/)。

优点：生态成熟、理解充分、训练稳定，单位参数性能高。  
缺点：计算量随规模线性增长，70B 模型的成本约为 3B 模型的 23 倍。

在内存受限场景或 LLM 初学者手中，这通常是默认选择。

Mixture of Experts (MoE，混合专家)  

将 Transformer 中的前馈层替换为多个“专家（experts）”。对每个 token，门控网络（gating network）只把它路由到少数几个专家。结果是用一小部分计算量获得大网络的容量。例如 [Kimi K2](https://huggingface.co/moonshotai/Kimi-K2-Instruct) 总参数量 1T，但每个 token 只激活 32B。代价是所有专家都必须加载到内存。可视化指南见[这篇博客](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)。

优点：训练与推理的单位计算性能更高。

缺点：内存占用高（必须加载所有专家）。训练比稠密 Transformer 更复杂。框架支持正在改善，但仍不如稠密模型成熟。分布式训练更是噩梦，涉及专家放置、负载均衡以及全对全通信等挑战。

适用场景：内存不受限且希望每单位计算获得极致性能时。

混合模型（Hybrid models） 将 Transformer 与状态空间模型（State Space Models, SSMs）如 Mamba 结合，在某些操作上实现线性复杂度，而注意力机制为二次方缩放。（[数学博客](https://srush.github.io/annotated-mamba/hard.html) | [可视化指南](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state)）

优点：在长上下文处理上潜力更大；对极长序列更高效。

缺点：相比稠密和 MoE 仍不成熟，缺乏经过验证的训练方案；框架支持有限。

适用场景：希望扩展到极大上下文，同时降低标准 Transformer 的推理开销。

回顾一下：先问模型部署在哪里，再结合团队经验和训练周期，评估你能承担多少探索：

对于 SmolLM3，我们想为端侧部署打造一款强大的小模型，周期约 3 个月，过去主要训练稠密模型。这排除了 MoE（内存限制）和混合架构（时间太短，无法探索新架构，且稠密模型已能达到我们目标的 128k token 长上下文），于是我们选择了 llama 风格的稠密模型。

现在我们已经研究了模型架构的内部，接下来看看分词器（tokenizer），它是数据与模型之间的桥梁。

#### [分词器（tokenizer）](https://huggingfacetb-smol-training-playbook.hf.space/#the-tokenizer)

尽管它很少像架构创新那样成为焦点，但分词（tokenization）方案很可能是任何语言模型中最被低估的组件之一。可以把它看作是人类语言与模型所处的数学世界之间的翻译器；正如任何翻译器一样，翻译质量至关重要。那么，我们如何为需求构建或选择合适的分词器呢？

分词器基础

本质上，分词器通过将连续文本切分成可单独处理的最小单位——称为 token（词元）——把原始文本转换成模型能处理的数字序列。在深入技术细节之前，我们应首先回答一些根本性问题，这些问题将指导我们的分词器设计：

*   我们要支持哪些语言？ 如果我们正在构建多语言模型，但分词器只见过英语，那么遇到非英语文本时，模型效率会很低，文本会被切分成远多于必要的 token，直接影响性能、训练成本与推理速度。
*   哪些领域对我们重要？ 除语言外，数学和代码等领域需要仔细表示数字。
*   我们是否了解目标数据混合比例？ 如果计划从头训练分词器，理想情况下应在能反映最终训练混合比例的样本上进行训练。

回答完这些问题后，我们可以审视主要的设计决策：

词汇表大小

词汇表本质上是一本字典，列出了模型能识别的所有 token（最小文本单位，如单词、子词或符号）。

更大的词表（vocabulary）可以更有效地压缩文本，因为每句话生成的 token 更少，但存在计算上的权衡。词表大小直接影响嵌入矩阵（embedding matrices）的尺寸。若词表大小为 $V$，隐藏维度为 $h$，则输入嵌入有 $V \times h$ 个参数，输出层另有 $V \times h$ 个参数。对于较小的模型，如“Embedding Sharing”一节所述，这部分会占据总参数的显著比例；但随着模型规模增大，其相对成本会下降。

最佳平衡点取决于我们的目标覆盖率（coverage）和模型规模。仅支持英语的模型通常约 5 万个 token 就够，而多语言模型往往需要 10 万以上，才能高效处理多样的书写系统和语言。现代最先进（state-of-the-art）模型如 Llama3 已采用 12.8 万+的词表，以提升跨语言的 token 效率。同一家族的小模型通过嵌入共享（embedding sharing）降低嵌入参数占比，同时仍能受益于更大的词表。[Dagan et al. (2024)](https://arxiv.org/abs/2402.01035) 分析了词表大小对压缩、推理和内存的影响。他们观察到，随着词表增大，压缩收益呈指数下降，表明存在最优大小。在推理阶段，更大的模型因压缩带来的前向计算节省超过 softmax 中额外嵌入 token 的开销，从而受益于更大的词表。在内存方面，最优大小取决于序列长度和批次大小：更长的上下文和更大的批次因 token 减少带来的 KV 缓存节省，而更适合使用更大的词表。

Tokenization algorithm

BPE（Byte-Pair Encoding，字节对编码）（[Sennrich et al., 2016](https://arxiv.org/abs/1508.07909)）仍然是最受欢迎的选择，其他算法如 WordPiece 或 SentencePiece 也存在，但采用度较低。同时，研究界对“无需分词器”（tokenizer-free）的方法兴趣日增，这类方法直接作用于字节或字符，有望彻底省去分词步骤。

在了解了定义分词器的关键参数之后，我们面临一个实际抉择：是使用现成的分词器，还是从头训练？答案取决于“覆盖度”：现有的分词器在目标词汇量下，能否很好地处理我们的语言和领域。

下图对比了 GPT-2 的纯英语分词器（[Radford et al., 2019](https://huggingfacetb-smol-training-playbook.hf.space/#bib-gpt2)）与 Gemma 3 的多语言分词器（[G. Team, Kamath, et al., 2025](https://arxiv.org/abs/2503.19786)）对同一句英语和阿拉伯语句子的切分效果。

虽然两者在英语上表现相近，但在阿拉伯语上差异显著：GPT-2 将文本拆成一百多个片段，而 Gemma 3 由于多语言训练数据及更大、更包容的词汇表，生成的 token 数量远少于此。

然而，衡量分词器质量不能仅凭肉眼观察几个切分示例就下结论，正如我们不能仅凭直觉做架构改动而不做消融实验一样。我们需要具体的指标来评估分词器质量。

衡量分词器质量

为了评估分词器的表现，我们可以采用 FineWeb2（[Penedo et al., 2025](https://arxiv.org/abs/2506.20920)）中使用的两项关键指标。

Fertility（生成密度）：

它衡量编码一个单词所需的平均 token 数。生成密度越低，压缩率越高，训练与推理速度也越快。可以这样理解：如果某个分词器对大多数单词需要多一两个 token，而另一个分词器用更少的 token 就能完成，那么后者显然更高效。

衡量 fertility（生成密度）的标准做法是计算 words-to-tokens ratio（词-词元比，即 word fertility），它衡量平均每个词需要多少个词元。该指标围绕“词”这一概念定义，因为在具备合适分词器（word tokenizer）的情况下，它能在跨语言比较中提供有意义的结果，例如在 [Spacy](https://spacy.io/) 和 [Stanza](https://stanfordnlp.github.io/stanza) 中（[Penedo et al., 2025](https://arxiv.org/abs/2506.20920)）。

在单一语言内比较不同 tokenizer（分词器）时，也可以用字符数或字节数代替词数，得到 characters-to-tokens ratio（字符-词元比）或 bytes-to-tokens ratio（字节-词元比）（[Dagan et al., 2024](https://arxiv.org/abs/2402.01035)）。然而，这些指标在跨语言比较中存在局限：字节数可能因不同文字的字节表示不同而失真（例如，汉字在 UTF-8 中占 3 字节，而拉丁字母占 1–2 字节）；同样，仅用字符数无法反映不同语言单词长度差异巨大——例如，中文词通常比德语复合词短得多。

Proportion of continued words（分词率）：

该指标告诉我们有多少比例的词被切分成多段。比例越低越好，意味着更少词被碎片化，从而使分词更高效。

下面实现这些指标：

```python
import numpy as np

def compute_tokenizer_metrics(tokenizer, word_tokenizer, text):
    """
    Computes fertility and proportion of continued words.
    
    Returns:
        tuple: (fertility, proportion_continued_words)
            - fertility: average tokens per word (lower is better)
            - proportion_continued_words: percentage of words split into 2+ tokens (lower is better)
    """
    words = word_tokenizer.word_tokenize(text)
    tokens = tokenizer.batch_encode_plus(words, add_special_tokens=False)
    tokens_per_word = np.array(list(map(len, tokens["input_ids"])))
    
    fertility = np.mean(tokens_per_word).item()
    proportion_continued_words = (tokens_per_word >= 2).sum() / len(tokens_per_word)
    
    return fertility, proportion_continued_words
```