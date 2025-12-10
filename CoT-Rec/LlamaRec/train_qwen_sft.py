import os
import json
import argparse
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="SFT Qwen with CoT-Rec data (only supervise answer span)"
    )

    parser.add_argument("--dataset_name", type=str, required=True,
                        help="数据集名称，如 Grocery_and_Gourmet_Food")
    parser.add_argument("--mode", type=str, default="random",
                        help="和 2_inference.py 中的 mode 一致，一般用 random")
    parser.add_argument("--p", type=int, default=1,
                        help="0 = sft0 (history only), 1 = sft1 (带 preferences)")
    parser.add_argument("--data_dir", type=str, default="LLaMA-Factory/data",
                        help="存放 *_valid_*.json / *_test_*.json 的目录")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="基础模型名称或本地路径")
    parser.add_argument("--output_root", type=str, default="output",
                        help="输出根目录，最终是 output/{dataset_name}_{mode}_{p}/")

    # 训练相关超参（根据你现在的 GPU / 时间约束调整过）
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=1024)

    # 学习率 & scheduler：这里已经改成你说的方案
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        choices=["cosine", "linear", "constant_with_warmup"])
    parser.add_argument("--warmup_ratio", type=float, default=0.05)

    parser.add_argument("--save_steps", type=int, default=400,
                        help="每多少 step 保存一个 checkpoint，对应 2_inference.py 的 stage")

    # 可选：从 checkpoint-XXX 继续训练（只在不改 scheduler/LR 时用）
    parser.add_argument("--resume_stage", type=int, default=-1,
                        help="例如 600 表示从 checkpoint-600 接着训；-1 表示不续训")

    return parser.parse_args()


def load_chat_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class CoTRecDataset(Dataset):
    """
    每个样本只在“最后一条 assistant 消息”上算 loss。
    prompt 部分（包括 system + user + 中间 assistant 等）全部 mask 为 -100。
    """
    def __init__(self, examples, tokenizer, max_len: int):
        self.input_ids = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        for ex in examples:
            messages = ex["messages"]
            if len(messages) < 2 or messages[-1]["role"] != "assistant":
                # 数据异常直接跳过
                continue

            # 1) 只包含到“让 assistant 开始回答”的 prompt（不含最终答案）
            prompt_messages = messages[:-1]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,  # 生成到 assistant: 为止
            )

            # 2) 包含完整答案的 full 文本
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # 分别 tokenize（不再额外加 special tokens，否则长度对不上）
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

            # 超长样本：为了不搞乱 mask 逻辑，直接丢弃
            if len(full_ids) > max_len:
                continue

            labels = full_ids.copy()

            # prompt 部分全设成 -100，只保留答案那段参与 loss
            prompt_len = min(len(prompt_ids), len(labels))
            for i in range(prompt_len):
                labels[i] = -100

            self.input_ids.append(full_ids)
            self.labels.append(labels)

        print(f"[Dataset] usable samples: {len(self.input_ids)}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
        }


def make_data_collator(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collator(features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []

        for f in features:
            ids = f["input_ids"]
            labs = f["labels"]
            pad_len = max_len - len(ids)

            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(labs + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collator


def main():
    args = parse_args()

    dataset_name = args.dataset_name
    mode = args.mode
    p = args.p

    train_file = os.path.join(args.data_dir, f"{dataset_name}_{mode}_valid_{p}.json")
    eval_file = os.path.join(args.data_dir, f"{dataset_name}_{mode}_test_{p}.json")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"train_file 不存在: {train_file}")
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"eval_file 不存在: {eval_file}")

    output_dir = os.path.join(args.output_root, f"{dataset_name}_{mode}_{p}")
    os.makedirs(output_dir, exist_ok=True)

    print("=== 配置 ===")
    print("Base model :", args.base_model)
    print("Train file :", train_file)
    print("Eval  file :", eval_file)
    print("Output dir :", output_dir)
    print("LR / scheduler :", args.learning_rate, args.lr_scheduler_type, "warmup_ratio=", args.warmup_ratio)

    # 1. tokenizer & base model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    # 节省显存
    model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    # 2. LoRA 配置（容量稍微放大）
    TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 构建只在答案上算 loss 的 Dataset
    train_examples = load_chat_json(train_file)
    eval_examples = load_chat_json(eval_file)

    train_dataset = CoTRecDataset(train_examples, tokenizer, args.max_len)
    eval_dataset = CoTRecDataset(eval_examples, tokenizer, args.max_len)

    # 4. TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,

        logging_steps=20,
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        save_total_limit=2,

        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
    )

    data_collator = make_data_collator(tokenizer)

    print("Using device:", next(model.parameters()).device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # === 是否从 checkpoint 续训 ===
    resume_ckpt = None
    if args.resume_stage > 0:
        resume_ckpt = os.path.join(output_dir, f"checkpoint-{args.resume_stage}")
        if not os.path.isdir(resume_ckpt):
            raise FileNotFoundError(f"指定的 checkpoint 不存在: {resume_ckpt}")
        print(f"\n===> 从 {resume_ckpt} 继续训练（注意：此时 scheduler/LR 以 checkpoint 为准）\n")

    if resume_ckpt is not None:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()

    print("\n训练完成！checkpoint 保存在：", output_dir)
    print("之后可以用 2_inference.py 这样调用，例如：")
    print(
        f"python 2_inference.py "
        f"--dataset_name {dataset_name} "
        f"--mode {mode} "
        f"--stage 800 "
        f"--tar -1 --k 10 --device 0 --p {p}"
    )


if __name__ == "__main__":
    main()
