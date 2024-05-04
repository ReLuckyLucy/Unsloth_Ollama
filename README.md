# 基于Unsloth框架下，使用llama3大模型为基底的模型微调



## 环境要求

llama3-8b模型需要6G显存



## 环境安装

> ​	以下为基于仓库[unslothai/unsloth：Finetune Llama 3、Mistral 和 Gemma LLM 速度提高 2-5 倍，内存减少 80% (github.com)](https://github.com/unslothai/unsloth)所编写



### Conda 安装(推荐)

选择 CUDA 11.8 或 CUDA 12.1。如果有，请使用代替以加快求解速度。有关调试 Conda 安装的帮助，请参阅此 [Github 问题](https://github.com/unslothai/unsloth/issues/73)

> 请注意<>中为cuda12.1与11.8两个版本，运行时候请选择一个，并删去另外一个以及<>

```
conda create --name unsloth_env python=3.10
conda activate unsloth_env

conda install pytorch-cuda=<12.1/11.8> pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

pip install --no-deps trl peft accelerate bitsandbytes
```



### pip安装

如果您有 Anaconda，**请不要**使用它。您必须使用 Conda 安装方法，否则东西会损坏。

1. 通过以下方式查找您的CUDA版本
```
import torch; torch.version.cuda
```

2. 对于 Pytorch 2.1.0：您可以通过 Pip （interchange / ） 更新 Pytorch。转到 https://pytorch.org/ 以了解更多信息。选择 CUDA 11.8 或 CUDA 12.1。如果您有 RTX 3060 或更高版本（A100、H100 等），请使用该路径。对于 Pytorch 2.1.1：转到步骤 3。对于 Pytorch 2.2.0：转到步骤 4
```bash
pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.0 triton \
  --index-url https://download.pytorch.org/whl/cu121
```
```bash
pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu118-ampere] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-ampere] @ git+https://github.com/unslothai/unsloth.git"
```
3. 对于 Pytorch 2.1.1：使用较新的 RTX 30xx GPU 或更高版本的路径
```bash
pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.1 triton \
  --index-url https://download.pytorch.org/whl/cu121
```
```bash
pip install "unsloth[cu118-torch211] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-torch211] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu118-ampere-torch211] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-ampere-torch211] @ git+https://github.com/unslothai/unsloth.git"
```
4. 对于 Pytorch 2.2.0：使用较新的 RTX 30xx GPU 或更高版本的路径
```bash
pip install --upgrade --force-reinstall --no-cache-dir torch==2.2.0 triton \
  --index-url https://download.pytorch.org/whl/cu121
```
```bash
pip install "unsloth[cu118-torch220] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-torch220] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu118-ampere-torch220] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-ampere-torch220] @ git+https://github.com/unslothai/unsloth.git"
```
5. 如果出现错误，请先尝试以下操作，然后返回步骤 1
```bash
pip install --upgrade pip
```
6. 对于 Pytorch 2.2.1
```bash
# RTX 3090, 4090 Ampere GPUs:
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes

# Pre Ampere RTX 2080, T4, GTX 1080 GPUs:
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes
```
7. 要排查安装问题，请尝试以下操作（所有操作都必须成功）。Xformer 应该基本上都可用。
```bash
nvcc
python -m xformers.info
python -m bitsandbytes
```



## 代码运行

> 本代码基于[[Alpaca + Llama-3 8b full example.ipynb - Colab (google.com)](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=2eSvM9zX_2d3)](https://colab.research.google.com/drive/1kpfbpSXsv1Qg4pqbCBF8D-AI4kbG-14g)

选择内核后，运行main.ipynb



## 训练参数调整

```
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
```

> 其中， max_steps为训练论次, per_device_train_batch_size为训练步长





## 微调模型保存

除了文件中给出的保存方式

```
model.save_pretrained("lora_model")
```



还有下面的几种方式
```
保存为VLLM
model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)

model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)


保存为lora
model.save_pretrained_merged("model", tokenizer, save_method = "lora",)


保存为GGUF
model.save_pretrained_gguf("model", tokenizer,)
model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")

```
> q8_0 - 快速转换。资源使用率高，但一般可以接受。
> q4_k_m - 推荐。将 Q6_K 用于一半的 attention.wv 和 feed_forward.w2 张量，否则Q4_K。
> q5_k_m - 推荐。将 Q6_K 用于一半的 attention.wv 和 feed_forward.w2 张量，否则Q5_K。


## 微调模型部署

推荐使用ollama部署

[ollama/ollama: Get up and running with Llama 3, Mistral, Gemma, and other large language models. (github.com)](https://github.com/ollama/ollama)



若需要前端可视化，推荐使用open webui部署

[open-webui/open-webui: User-friendly WebUI for LLMs (Formerly Ollama WebUI) (github.com)](https://github.com/open-webui/open-webui)