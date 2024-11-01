<div align="center">
    <h1 align="center">Agents Evaluate Agents</h1>
    <img src="assets/devai_logo.png" alt="DevAI Logo" width="150" height="150">
    <p align="center">
        <a href="https://devai.tech"><b>Project</b></a> |  -->
<!--         <a href="https://huggingface.co/DEVAI-benchmark"><b>🤗 Dataset</b></a> | 
        <a href="https://arxiv.org/pdf/2410.10934"><b>📑 Paper</b></a> 
<!-- <a href="https://arxiv.org/pdf/2410.10934">
  <img src="https://img.shields.io/badge/Paper%20on%20Arxiv-1877F2?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Paper on Arxiv">
</a>
<a href="https://huggingface.co/DEVAI-benchmark">
  <img src="https://img.shields.io/badge/Benchmark%20score-1877F2?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="DevAI Benchmark">
</a> -->
    </p>
</div>

> [!NOTE]
> Current evaluation techniques are often inadequate for advanced **agentic systems** due to their focus on final outcomes and labor-intensive manual reviews. To overcome this limitation, we introduce the **Agent-as-a-Judge** framework. 
>

## 🤠 Features

Agent-as-a-Judge offers two key advantages:

- **Automated Evaluation**: Agent-as-a-Judge can evaluate tasks during or after execution, saving 97.72% of time and 97.64% of costs compared to human experts.
- **Provide Reward Signals**: It provides continuous, step-by-step feedback that can be used as reward signals for further agentic training and improvement.

<div align="center">
    <img src="assets/demo.gif" alt="Demo GIF" style="width: 100%; max-width: 650px;">
</div>
<div align="center">
    <img src="assets/judge_first.png" alt="AaaJ" style="width: 95%; max-width: 650px;">
</div>



## 🎮 Quick Start 

### 1. install

```python
git clone https://github.com/metauto-ai/agent-as-a-judge.git
cd agent-as-a-judge/
conda create -n aaaj python=3.11
conda activate aaaj
pip install poetry
poetry install
```

### 2. set LLM&API

Before running, rename `.env.sample` to `.env` and fill in the **required APIs and Settings** in the main repo folder to support LLM calling. The `LiteLLM` tool supports various LLMs.

### 3. run 

> [!TIP]
> See more comprehensive [usage scripts](scripts/README.md).
>


#### Usage A: **Ask Anything** for Any Workspace:

```python 

PYTHONPATH=. python scripts/run_ask.py \
  --workspace $(pwd)/benchmark/workspaces/OpenHands/39_Drug_Response_Prediction_SVM_GDSC_ML \
  --question "What does this workspace contain?"
```

You can find an [example](assets/ask_sample.md) to see how **Ask Anything** works.


#### Usage B: **Agent-as-a-Judge** for **DevAI**


```python

PYTHONPATH=. python scripts/run_aaaj.py \
  --developer_agent "OpenHands" \
  --setting "black_box" \
  --planning "efficient (no planning)" \
  --benchmark_dir $(pwd)/benchmark
```

💡 There is an [example](assets/aaaj_sample.md) that shows the process of how **Agent-as-a-Judge** collects evidence for judging.



## 🤗 DevAI Dataset 



<div align="center">
    <img src="assets/dataset.png" alt="Dataset" style="width: 100%; max-width: 600px;">
</div>

> [!IMPORTANT]
> As a **proof-of-concept**, we applied **Agent-as-a-Judge** to code generation tasks using **DevAI**, a benchmark consisting of 55 realistic AI development tasks with 365 hierarchical user requirements. The results demonstrate that **Agent-as-a-Judge** significantly outperforms traditional evaluation methods, delivering reliable reward signals for scalable self-improvement in agentic systems.
> 
> Check out the dataset on [Hugging Face 🤗](https://huggingface.co/DEVAI-benchmark).
> See how to use this dataset in the [guidelines](benchmark/devai/README.md).


<!-- <div align="center">
    <img src="assets/sample.jpeg" alt="Sample" style="width: 100%; max-width: 600px;">
</div> -->

## Reference

Feel free to cite if you find the Agent-as-a-Judge concept useful for your work:

```
@article{zhuge2024agent,
  title={Agent-as-a-Judge: Evaluate Agents with Agents},
  author={Zhuge, Mingchen and Zhao, Changsheng and Ashley, Dylan and Wang, Wenyi and Khizbullin, Dmitrii and Xiong, Yunyang and Liu, Zechun and Chang, Ernie and Krishnamoorthi, Raghuraman and Tian, Yuandong and Shi, Yangyang and Chandra, Vikas and Schmidhuber, J{\"u}rgen},
  journal={arXiv preprint arXiv:2410.10934},
  year={2024}
}
```


