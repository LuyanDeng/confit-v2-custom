# =============================
# 0️⃣ 环境导入
# =============================
import os, json, torch, pandas as pd, torch.nn.functional as F
import pytorch_lightning as pl
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch.nn as nn
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint

# =============================
# 1️⃣ 训练参数
# =============================
@dataclass
class TrainingArguments:
    save_path: str = field(default="model_checkpoints/linkedin_v2")
    max_epochs: int = field(default=3)
    train_batch_size: int = field(default=8)
    val_batch_size: int = field(default=8)
    lr: float = field(default=2e-5)
    precision: str = field(default="16-mixed")
    strategy: str = field(default="auto")
    seed: int = field(default=42)
    project: str = field(default="mini-confit-linkedin")
    run_name: str = field(default="baseline-mha")

# =============================
# 2️⃣ 数据加载
# =============================
def load_recruiting_data(data_dir, tokenizer, max_length=512):
    resume_df = pd.read_csv(os.path.join(data_dir, "all_resume.csv"))
    job_df = pd.read_csv(os.path.join(data_dir, "all_job.csv"))
    train_labels = [json.loads(line) for line in open(os.path.join(data_dir, "train_labeled_data.jsonl"))]
    val_labels = [json.loads(line) for line in open(os.path.join(data_dir, "valid_classification_data.jsonl"))]

    resume_dict = {r["user_id"]: r["text_resume"] for r in resume_df.to_dict("records")}
    job_dict = {j["jd_no"]: j["text_job"] for j in job_df.to_dict("records")}

    def build_pairs(label_list):
        pairs = []
        for item in label_list:
            rid, jid = item["user_id"], item["jd_no"]
            if rid in resume_dict and jid in job_dict:
                pairs.append((resume_dict[rid], job_dict[jid]))
        return pairs

    def encode_pairs(pairs):
        data = []
        for rtext, jtext in pairs:
            r_inputs = tokenizer(rtext, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            j_inputs = tokenizer(jtext, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            data.append({"resume": r_inputs, "job": j_inputs})
        return data

    train_pairs, val_pairs = build_pairs(train_labels), build_pairs(val_labels)
    train_data, val_data = encode_pairs(train_pairs), encode_pairs(val_pairs)

    def collate_fn(batch):
        resumes = {k: torch.cat([x["resume"][k] for x in batch]) for k in batch[0]["resume"]}
        jobs = {k: torch.cat([x["job"][k] for x in batch]) for k in batch[0]["job"]}
        return {"batched_resume": resumes, "batched_job": jobs}

    return (
        DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn),
        DataLoader(val_data, batch_size=4, shuffle=False, collate_fn=collate_fn)
    )

# =============================
# 3️⃣ 模型定义（含 Multi-Head Attention）
# =============================
class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim=384, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv):
        attn_out, _ = self.cross_attn(q, kv, kv)
        return self.norm(q + self.dropout(attn_out))

class MiniConFitModel(pl.LightningModule):
    def __init__(self, pretrained="sentence-transformers/paraphrase-MiniLM-L6-v2", lr=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = AutoModel.from_pretrained(pretrained)
        self.cross_attn = CrossAttentionLayer(hidden_dim=self.encoder.config.hidden_size)
        self.temperature = 0.07
        self.lr = lr

    def mean_pool(self, last_hidden, mask):
        mask_exp = mask.unsqueeze(-1).expand(last_hidden.size())
        return (last_hidden * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)

    def encode(self, inputs):
        outputs = self.encoder(**inputs)
        return outputs.last_hidden_state, inputs["attention_mask"]

    def forward(self, resume_inputs, job_inputs):
        r_hidden, r_mask = self.encode(resume_inputs)
        j_hidden, j_mask = self.encode(job_inputs)
        r_cross = self.cross_attn(r_hidden, j_hidden)
        j_cross = self.cross_attn(j_hidden, r_hidden)
        r_vec = F.normalize(self.mean_pool(r_cross, r_mask), dim=1)
        j_vec = F.normalize(self.mean_pool(j_cross, j_mask), dim=1)
        return r_vec, j_vec

    def training_step(self, batch, batch_idx):
        r_vec, j_vec = self.forward(batch["batched_resume"], batch["batched_job"])
        sim = torch.einsum("id,jd->ij", r_vec, j_vec) / self.temperature
        labels = torch.arange(sim.size(0), device=self.device)
        loss = F.cross_entropy(sim, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        r_vec, j_vec = self.forward(batch["batched_resume"], batch["batched_job"])
        sim = torch.einsum("id,jd->ij", r_vec, j_vec)
        acc = (sim.argmax(dim=1) == torch.arange(sim.size(0), device=self.device)).float().mean()
        self.log("val_acc", acc, prog_bar=True)
        return acc

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        sched = get_linear_schedule_with_warmup(opt, 0, 100)
        return [opt], [sched]

# =============================
# 4️⃣ 主函数
# =============================
def main():
    args = TrainingArguments()
    pl.seed_everything(args.seed)
    wandb.init(project=args.project, name=args.run_name)

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
    # train 2000 data
    train_loader, val_loader = load_recruiting_data("dataset/linkedin_data_v2", tokenizer)
    print("Train size:", len(train_loader.dataset))
    print("Valid size:", len(val_loader.dataset))
    print(pd.read_json("dataset/linkedin_data_v2/train_labeled_data.jsonl", lines=True).head())


    # model = MiniConFitModel(lr=args.lr)
    #load previous checkpoints
    ckpt_path = "model_checkpoints/linkedin_v1/epoch=2-val_acc=0.967.ckpt"
    print(f"🔄 Loading checkpoint from {ckpt_path}")
    model = MiniConFitModel.load_from_checkpoint(ckpt_path)
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=args.save_path,                
    #     filename="epoch{epoch}-val_acc{val_acc:.3f}",  
    #     save_top_k=1,                          
    #     monitor="val_acc",                     
    #     mode="max"
    # )
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.save_path}/continued",
        filename="continued-epoch{epoch}-val_acc{val_acc:.3f}",
        save_top_k=1,
        monitor="val_acc",
        mode="max"
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        strategy=args.strategy,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=1,
        default_root_dir=args.save_path,       # ✅ 强制保存到本地
        callbacks=[checkpoint_callback],
        logger=pl.loggers.WandbLogger(project=args.project, name=args.run_name),
    )

    trainer.fit(model, train_loader, val_loader)
    wandb.finish()

if __name__ == "__main__":
    main()
