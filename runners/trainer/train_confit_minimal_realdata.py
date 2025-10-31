import os
import json
import torch
import pytorch_lightning as pl
import pandas as pd
from dataclasses import dataclass, field
from functools import partial
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# =============================
# 1️⃣ 基础训练参数
# =============================
@dataclass
class TrainingArguments:
    save_path: str = field(default="model_checkpoints/debug_toy")
    max_epochs: int = field(default=1)
    train_batch_size: int = field(default=1)
    val_batch_size: int = field(default=1)
    lr: float = field(default=1e-5)
    precision: str = field(default="16-mixed")  # ✅ 适配 L4
    strategy: str = field(default="auto")
    seed: int = field(default=42)
    no_save: bool = field(default=False)


# =============================
# 2️⃣ 数据加载函数
# =============================
def load_recruiting_data(data_dir, tokenizer, max_length=256):
    """加载真实 resume/job + label 数据"""
    resume_df = pd.read_csv(os.path.join(data_dir, "all_resume.csv"))
    job_df = pd.read_csv(os.path.join(data_dir, "all_job.csv"))
    train_labels = [json.loads(line) for line in open(os.path.join(data_dir, "train_labeled_data.jsonl"))]
    val_labels = [json.loads(line) for line in open(os.path.join(data_dir, "valid_classification_data.jsonl"))]

    # resume/job 字段使用 text_resume / text_job
    resume_dict = {r["user_id"]: r["text_resume"] for r in resume_df.to_dict("records")}
    job_dict = {j["jd_no"]: j["text_job"] for j in job_df.to_dict("records")}

    def build_pairs(label_list):
        pairs = []
        for item in label_list:
            rid, jid = item["user_id"], item["jd_no"]
            if rid in resume_dict and jid in job_dict:
                pairs.append((resume_dict[rid], job_dict[jid]))
        return pairs

    train_pairs = build_pairs(train_labels)
    val_pairs = build_pairs(val_labels)

    def encode_pairs(pairs):
        data = []
        for rtext, jtext in pairs:
            resume_inputs = tokenizer(rtext, padding="max_length", truncation=True,
                                      max_length=max_length, return_tensors="pt")
            job_inputs = tokenizer(jtext, padding="max_length", truncation=True,
                                   max_length=max_length, return_tensors="pt")
            data.append({"resume": resume_inputs, "job": job_inputs})
        return data

    train_data = encode_pairs(train_pairs)
    val_data = encode_pairs(val_pairs)

    def collate_fn(batch):
        resumes = {k: torch.cat([x["resume"][k] for x in batch]) for k in batch[0]["resume"]}
        jobs = {k: torch.cat([x["job"][k] for x in batch]) for k in batch[0]["job"]}
        return {"batched_resume": resumes, "batched_job": jobs}

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, val_loader


# =============================
# 3️⃣ 模型定义 (精简版 ConFit)
# =============================
class MiniConFitModel(pl.LightningModule):
    def __init__(self, pretrained="sentence-transformers/paraphrase-MiniLM-L6-v2", lr=1e-5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained)
        self.lr = lr
        self.temperature = 0.07

    def mean_pool(self, last_hidden, mask):
        mask_expanded = mask.unsqueeze(-1).expand(last_hidden.size())
        return (last_hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)

    def forward(self, inputs):
        outputs = self.encoder(**inputs)
        embeddings = self.mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def training_step(self, batch, batch_idx):
        resume_vecs = self.forward(batch["batched_resume"])
        job_vecs = self.forward(batch["batched_job"])
        sim = torch.einsum("id,jd->ij", resume_vecs, job_vecs) / self.temperature
        labels = torch.arange(sim.size(0), device=self.device)
        loss = torch.nn.functional.cross_entropy(sim, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, 20)
        return [optimizer], [scheduler]


# =============================
# 4️⃣ 主函数
# =============================
def main():
    args = TrainingArguments()
    pl.seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")

    print("🔹 Loading recruiting_data_v2_toy dataset ...")
    train_loader, val_loader = load_recruiting_data("dataset/recruiting_data_v2_toy", tokenizer)

    print("🔹 Building model ...")
    model = MiniConFitModel(lr=args.lr)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        strategy=args.strategy,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=1,
        default_root_dir=args.save_path,
    )

    print("🚀 Start training ...")
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
