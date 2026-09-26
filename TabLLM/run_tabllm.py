"""
TabLLM-style pipeline for CBC diagnosis classification, adapted from the
original T0_3B (seq2seq) + IA3 script to EPFLiGHT/Apertus-8B-MeditronFO, a
decoder-only (causal LM) medical-specialist model.

Why this needed more than a model-name swap:
  - T0_3B is encoder-decoder; Apertus-8B-MeditronFO is decoder-only
    (architecture "ApertusForCausalLM", Llama-style proj names). That changes
    the model class, the training-example format, the trainer, and the
    rank-classification scorer.
  - 8B params doesn't fit a 20-30GB budget in bf16 alongside gradients/optimizer
    state, so the base model is loaded in 4-bit NF4 (bitsandbytes) and only a
    tiny IA3 adapter is trained on top -- same method as the original TabLLM
    script, just retargeted to causal-LM module names.

Checkpoint selection: --epochs is a hard ceiling. An adapter checkpoint is
saved every epoch, training loss is logged every epoch, and at the end the
checkpoint from the epoch with the LOWEST training loss is reloaded and
saved as the final adapter. Note: training loss usually falls monotonically,
so in practice this will often just be the last epoch -- if you want the
selection to mean something beyond "did it finish going down", you'd need a
held-out validation split, but this keeps things simple as requested.

Modes (--mode):
  - zeroshot: no training. Rank-classify the test set directly against the
    (optionally 4-bit) base model. Supports --shots N to prepend N labeled
    exemplars to the prompt (in-context / few-shot).
  - finetune: IA3-finetune the base model on train_labeled.csv, then
    rank-classify the test set with the lowest-train-loss checkpoint.
  - both (default): loads the quantized base model ONCE, evaluates it
    zero-shot first, then attaches IA3 to that same in-memory model and
    fine-tunes it -- avoids loading the 8B weights twice.
  - eval: load a PREVIOUSLY SAVED IA3 adapter from --output-dir (e.g.
    "/home/2023eeb1196/CBC/tabllm_apertus_checkpoint/Prospective") and just
    rank-classify --test-path against it. No training happens, and no
    --train-path is needed at all: feature columns are derived from the
    test CSV itself (same DROP_COLS exclusion used during training), so the
    exact same serialize_row()/prompt-building/rank-classification path
    that produced the checkpoint is reused unchanged for scoring.

Multi-token class name handling (unchanged in spirit from the original):
  - TRAINING: standard causal LM teacher forcing already scores every target
    token; the only extra step is masking the prompt tokens' labels to -100
    so loss is computed only on the answer (+ EOS).
  - INFERENCE: rank classification compares log-likelihoods across candidates
    of very different lengths. LENGTH NORMALIZATION (dividing each
    candidate's summed log-likelihood by its token count) corrects for
    longer diagnosis names otherwise being unfairly penalized.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import IA3Config, PeftModel, get_peft_model, TaskType, prepare_model_for_kbit_training
from peft.utils import load_peft_weights, set_peft_model_state_dict
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# Workaround for a query/key/value dtype mismatch seen with this
# transformers/peft/bitsandbytes combo when scoring with a loaded IA3
# adapter on a 4-bit quantized model (query stays bf16, key/value come out
# fp32 from some upstream module -- root cause not fully isolated across a
# few rounds of debugging, so we align dtypes right at the SDPA boundary
# instead. This only affects numerical precision, not correctness, since
# it's downcasting fp32 activations to bf16 immediately before the same
# matmul that would otherwise error out.)
_orig_sdpa = F.scaled_dot_product_attention


def _dtype_safe_sdpa(query, key, value, *args, **kwargs):
    if key.dtype != query.dtype:
        key = key.to(query.dtype)
    if value.dtype != query.dtype:
        value = value.to(query.dtype)
    return _orig_sdpa(query, key, value, *args, **kwargs)


F.scaled_dot_product_attention = _dtype_safe_sdpa
torch.nn.functional.scaled_dot_product_attention = _dtype_safe_sdpa

# Same category of issue is showing up at more than one point (attention,
# then lm_head) despite explicit parameter casts -- activations are drifting
# to fp32 somewhere mid-forward (most likely inside a norm layer's internal
# upcast-then-downcast not fully restoring dtype in this transformers
# version) rather than it being one specific stuck parameter. Patch F.linear
# itself so ANY matmul in the model aligns input/bias to the weight's dtype,
# instead of chasing each individual failure site.
_orig_linear = F.linear


def _dtype_safe_linear(input, weight, bias=None):
    if input.dtype != weight.dtype:
        input = input.to(weight.dtype)
    if bias is not None and bias.dtype != weight.dtype:
        bias = bias.to(weight.dtype)
    return _orig_linear(input, weight, bias)


F.linear = _dtype_safe_linear
torch.nn.functional.linear = _dtype_safe_linear

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_PATH = ""
TEST_PATH = ""
OUTPUT_DIR = ""

TARGET_COL = "Revised_final_diagnosis"

DROP_COLS = [
    "pseudo_patient_id", "Analyzer_model", "Analyzer_id",
    "RDW-SD(fL)", "RDW-CV(%)", "PDW(fL)", "MPV(fL)", "P-LCR(%)", "PCT(%)",
    "Revised_final_diagnosis",
]

# Same 10-class restriction used in the TabPFN/CatBoost/XGBoost comparison runs.
VALID_CLASSES = [
    'Acute lymphoblastic leukemia', 'Chronic myeloid leukemia', 'Acute myeloid leukemia non M3',
    'Multiple myeloma', 'Aplastic anemia', 'Chronic lymphocytic leukemia',
    'Acute promyelocytic leukemia', 'Acute leukemia', 'Eosinophilia', 'Primary myelofibrosis',
]

MODEL_NAME = "EPFLiGHT/Apertus-8B-MeditronFO"

# Llama-style projection names, confirmed via swiss-ai/Apertus-8B's config.json
# (architecture "ApertusForCausalLM": q_proj/k_proj/v_proj/o_proj/gate_proj/
# up_proj/down_proj).
IA3_TARGET_MODULES = ["k_proj", "v_proj", "down_proj"]
IA3_FEEDFORWARD_MODULES = ["down_proj"]

MAX_INPUT_LEN = 2048
MAX_TARGET_LEN = 40


def clean_labels(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zeroshot", "finetune", "both", "eval"], default="both",
                         help="zeroshot: no training. finetune: IA3-tune then evaluate. "
                              "both: zero-shot eval, then IA3-tune the SAME loaded model and evaluate again. "
                              "eval: load a previously saved IA3 adapter from --output-dir and just evaluate "
                              "it on --test-path, no training and no --train-path needed.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--train-path", type=Path, default=Path(TRAIN_PATH),
                         help="Ignored in --mode eval.")
    parser.add_argument("--test-path", type=Path, default=Path(TEST_PATH))
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT_DIR),
                         help="In --mode eval, this is the path to load the saved IA3 adapter FROM "
                              "(e.g. /home/2023eeb1196/CBC/tabllm_apertus_checkpoint/Prospective). "
                              "In other modes, this is where checkpoints/results are saved TO.")
    parser.add_argument("--no-4bit", action="store_true",
                         help="Disable 4-bit quantization (needs ~16GB+ just for bf16 weights -- "
                              "only use this if you have more than the 20-30GB budget to spare).")
    parser.add_argument("--shots", type=int, default=0,
                         help="Number of labeled exemplars to prepend to the prompt in zero-shot mode "
                              "(in-context / few-shot). 0 = pure zero-shot.")
    parser.add_argument("--epochs", type=int, default=20,
                         help="Hard ceiling on training epochs. The checkpoint kept at the end is "
                              "whichever epoch had the lowest training loss.")
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-len", type=int, default=MAX_INPUT_LEN)
    parser.add_argument("--max-target-len", type=int, default=MAX_TARGET_LEN)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading + Text Template serialization (paper's best-performing method)
# ---------------------------------------------------------------------------
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df[TARGET_COL] = clean_labels(train_df[TARGET_COL])
    test_df[TARGET_COL] = clean_labels(test_df[TARGET_COL])

    n_train_before, n_test_before = len(train_df), len(test_df)
    train_df = train_df[train_df[TARGET_COL].isin(VALID_CLASSES)].reset_index(drop=True)
    test_df = test_df[test_df[TARGET_COL].isin(VALID_CLASSES)].reset_index(drop=True)
    print(f"Train: kept {len(train_df)}/{n_train_before} rows after filtering to valid_classes")
    print(f"Test:  kept {len(test_df)}/{n_test_before} rows after filtering to valid_classes")

    feature_cols = [c for c in train_df.columns if c not in DROP_COLS]
    missing_in_test = [c for c in feature_cols if c not in test_df.columns]
    if missing_in_test:
        raise ValueError(f"Test set is missing columns present in train: {missing_in_test}")

    known_classes = sorted(VALID_CLASSES)  # fixed class set, not whatever survives filtering
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}, "
          f"Features: {len(feature_cols)}, Classes: {len(known_classes)}")
    return train_df, test_df, feature_cols, known_classes


def load_eval_only_data(test_path):
    """Same label-cleaning + VALID_CLASSES filtering as load_data(), but for
    --mode eval where there's no train split at all. feature_cols is derived
    straight from the test CSV's own columns (minus DROP_COLS) -- the exact
    same exclusion list used to build feature_cols from the train CSV during
    training, so serialize_row()/build_prompt() produce identical-format
    prompts to whatever the checkpoint was trained on, as long as the test
    CSV has the same column schema as the original train CSV."""
    test_df = pd.read_csv(test_path)
    test_df[TARGET_COL] = clean_labels(test_df[TARGET_COL])

    n_test_before = len(test_df)
    test_df = test_df[test_df[TARGET_COL].isin(VALID_CLASSES)].reset_index(drop=True)
    print(f"Test: kept {len(test_df)}/{n_test_before} rows after filtering to valid_classes")

    feature_cols = [c for c in test_df.columns if c not in DROP_COLS]
    known_classes = sorted(VALID_CLASSES)
    print(f"Test rows: {len(test_df)}, Features: {len(feature_cols)}, Classes: {len(known_classes)}")
    return test_df, feature_cols, known_classes


def serialize_row(row: pd.Series, feature_cols) -> str:
    parts = [f"{col} is {row[col]}" for col in feature_cols]
    return ", ".join(parts) + "."


TASK_DESCRIPTION = (
    "Here are the complete blood count results for a patient. "
    "{serialized} "
    "Question: What is the most likely diagnosis for this patient?\n"
    "Answer:"
)


def build_prompt(row, feature_cols, few_shot_prefix=""):
    return few_shot_prefix + TASK_DESCRIPTION.format(serialized=serialize_row(row, feature_cols))


def build_few_shot_prefix(train_df, feature_cols, shots, seed):
    """Prepend `shots` labeled exemplars (same serialization + answer) before
    the actual query prompt -- in-context learning for the zero-shot path."""
    if shots <= 0:
        return ""
    sample = train_df.sample(n=min(shots, len(train_df)), random_state=seed)
    blocks = []
    for _, row in sample.iterrows():
        blocks.append(TASK_DESCRIPTION.format(serialized=serialize_row(row, feature_cols))
                      + " " + row[TARGET_COL] + "\n\n")
    return "".join(blocks)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_base_model_and_tokenizer(model_name, use_4bit):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        dtype=torch.bfloat16,  # `torch_dtype` is deprecated; with quantization, an ignored/deprecated
                                # dtype kwarg can leave non-quantized submodules in fp32 while quantized
                                # layers compute in bf16 -- exactly the mismatch seen at inference.
        device_map="auto",
        trust_remote_code=True,
    )
    cast_float32_params_to_bf16(model)
    return model, tokenizer


def cast_float32_params_to_bf16(model):
    """Some non-quantized submodules (RMSNorm weights, lm_head, etc.) can end
    up stuck in float32 despite dtype=torch.bfloat16 at load time -- seen
    here as query/key/value dtype mismatches in attention, then a
    hidden_states/lm_head.weight mismatch one layer later. bitsandbytes'
    quantized Params4bit weights are NOT float32 (they're packed as uint8),
    so this cast is safe: it only touches genuinely non-quantized fp32
    leftovers, downcasting them to match the rest of the model."""
    n_cast = 0
    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)
            n_cast += 1
    if n_cast:
        print(f"Cast {n_cast} base-model parameter tensor(s) from float32 to bfloat16.")


def attach_ia3(model):
    ia3_config = IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=IA3_TARGET_MODULES,
        feedforward_modules=IA3_FEEDFORWARD_MODULES,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, ia3_config)
    model.print_trainable_parameters()
    model.config.use_cache = False  # required alongside gradient checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # needed for grad checkpointing through a quantized+PEFT base
    return model


# ---------------------------------------------------------------------------
# Training (causal LM, prompt tokens masked out of the loss)
# ---------------------------------------------------------------------------
def build_causal_lm_dataset(df, feature_cols, tokenizer, max_input_len, max_target_len):
    records = []
    for _, row in df.iterrows():
        prompt = build_prompt(row, feature_cols)
        records.append({"input_text": prompt, "target_text": row[TARGET_COL]})
    hf_dataset = Dataset.from_list(records)

    def tokenize_fn(example):
        prompt_ids = tokenizer(example["input_text"], truncation=True,
                                max_length=max_input_len, add_special_tokens=True)["input_ids"]
        # Leading space before the target keeps BPE tokenization consistent
        # with how the same text would tokenize mid-sentence (matches the
        # convention used for scoring at inference time below).
        target_ids = tokenizer(" " + example["target_text"], truncation=True,
                                max_length=max_target_len, add_special_tokens=False)["input_ids"]
        eos_id = tokenizer.eos_token_id
        input_ids = prompt_ids + target_ids + [eos_id]
        labels = [-100] * len(prompt_ids) + target_ids + [eos_id]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    return hf_dataset.map(tokenize_fn, remove_columns=hf_dataset.column_names)


def plot_loss_curve(epoch_losses, best_epoch, output_path):
    epochs = [e for e, _ in epoch_losses]
    losses = [l for _, l in epoch_losses]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, losses, marker="o", color="tab:blue", label="Train loss")
    best_loss = dict(epoch_losses)[best_epoch]
    ax.scatter([best_epoch], [best_loss], color="red", zorder=5, label=f"Best (epoch {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_title("Finetuning loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_finetune(model, tokenizer, train_df, feature_cols, args):
    model = attach_ia3(model)
    tokenized_train = build_causal_lm_dataset(train_df, feature_cols, tokenizer,
                                               args.max_input_len, args.max_target_len)
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

    checkpoint_dir = args.output_dir / "trainer_state"
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,  # hard ceiling; best checkpoint is picked by train loss below
        learning_rate=args.learning_rate,
        bf16=True,
        logging_strategy="epoch",  # one loss value per epoch, aligned with saved checkpoints
        save_strategy="epoch",
        save_total_limit=args.epochs,  # keep every epoch's checkpoint so we can pick the best afterward
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
    )
    trainer.train()

    # One {"loss": ..., "epoch": ..., "step": ...} entry per epoch, since
    # logging_strategy="epoch" == save_strategy="epoch" here.
    epoch_losses = [(round(h["epoch"]), h["loss"]) for h in trainer.state.log_history if "loss" in h]
    step_by_epoch = {round(h["epoch"]): h["step"] for h in trainer.state.log_history if "loss" in h}
    best_epoch, best_loss = min(epoch_losses, key=lambda pair: pair[1])
    best_step = step_by_epoch[best_epoch]
    print(f"Best training loss: {best_loss:.4f} at epoch {best_epoch} (step {best_step})")

    plot_loss_curve(epoch_losses, best_epoch, args.output_dir / "loss_curve.png")
    pd.DataFrame(epoch_losses, columns=["epoch", "train_loss"]).to_csv(
        args.output_dir / "loss_history.csv", index=False
    )
    print(f"Loss curve + history saved under {args.output_dir}")

    # Reload the lowest-loss epoch's adapter weights into the (currently
    # last-epoch) model before saving/evaluating.
    best_checkpoint_path = checkpoint_dir / f"checkpoint-{best_step}"
    adapter_weights = load_peft_weights(str(best_checkpoint_path))
    set_peft_model_state_dict(model, adapter_weights)

    model.save_pretrained(str(args.output_dir))  # PEFT model -> saves only the small IA3 adapter
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"IA3 adapter (epoch {best_epoch}) + tokenizer saved to {args.output_dir}")
    return model


# ---------------------------------------------------------------------------
# Rank-classification inference (causal LM version)
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_candidates_causal(prompt, candidates, model, tokenizer, max_input_len):
    """Length-normalized log-likelihood of each candidate's continuation
    tokens, conditioned on the shared prompt, in a single batched forward
    pass. Length normalization is the fix for multi-token class names:
    without it, a longer diagnosis name is penalized relative to a shorter
    one purely for length, not fit."""
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_input_len,
                            add_special_tokens=True)["input_ids"]
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    prompt_len = len(prompt_ids)

    sequences, cand_lens = [], []
    for candidate in candidates:
        cand_ids = tokenizer(" " + candidate, add_special_tokens=False)["input_ids"]
        sequences.append(prompt_ids + cand_ids + [eos_id])
        cand_lens.append(len(cand_ids) + 1)  # + EOS

    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for i, seq in enumerate(sequences):
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :len(seq)] = 1
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits.float(), dim=-1)  # upcast for numerically stable log-softmax

    scores = []
    for i, seq in enumerate(sequences):
        cand_len = cand_lens[i]
        target_positions = list(range(prompt_len, prompt_len + cand_len))
        pred_positions = [p - 1 for p in target_positions]  # next-token prediction: shift by one
        target_ids = torch.tensor(seq[prompt_len:prompt_len + cand_len], device=device)
        token_log_probs = log_probs[i, pred_positions, target_ids]
        scores.append((token_log_probs.sum() / cand_len).item())
    return scores


def evaluate_and_save(model, tokenizer, test_df, feature_cols, known_classes, output_dir,
                       max_input_len, tag, few_shot_prefix=""):
    model.eval()
    y_true = test_df[TARGET_COL].tolist()
    y_pred = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Rank-classifying test set ({tag})"):
        prompt = build_prompt(row, feature_cols, few_shot_prefix=few_shot_prefix)
        scores = score_candidates_causal(prompt, known_classes, model, tokenizer, max_input_len)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        y_pred.append(known_classes[best_idx])

    output_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame({"true_label": y_true, "predicted_label": y_pred})
    results_path = output_dir / f"test_predictions_{tag}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"[{tag}] Test predictions saved to {results_path}")

    report = classification_report(y_true, y_pred, labels=known_classes, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_path = output_dir / f"classification_report_{tag}.csv"
    report_df.to_csv(report_path)
    print(f"[{tag}] Classification report saved to {report_path}")
    print(f"[{tag}] Accuracy: {report['accuracy']:.4f}, "
          f"Macro F1: {report['macro avg']['f1-score']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    use_4bit = not args.no_4bit
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- eval mode: load a saved IA3 adapter from args.output_dir and just
    # score args.test_path against it. No train CSV is read at all, and
    # feature_cols comes straight off the test CSV (see load_eval_only_data
    # docstring) so the prompt format matches training exactly.
    if args.mode == "eval":
        test_df, feature_cols, known_classes = load_eval_only_data(args.test_path)

        print(f"Loading {args.model_name} (4-bit={use_4bit}) ...")
        model, tokenizer = load_base_model_and_tokenizer(args.model_name, use_4bit)

        print(f"Loading previously saved IA3 adapter from {args.output_dir} ...")
        model = PeftModel.from_pretrained(model, str(args.output_dir))

        # IA3 adapter weights (the ia3_l scaling vectors) load as float32 by
        # default, while the 4-bit base model computes in bf16. Since IA3
        # rescales k_proj/v_proj outputs elementwise, a bf16 hidden state
        # times an fp32 scaling vector upcasts to fp32 -- producing bf16
        # queries alongside fp32 keys/values, which SDPA rejects. Cast the
        # adapter's float32 params to bf16 so everything matches.
        n_cast = 0
        for param in model.parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)
                n_cast += 1
        print(f"Cast {n_cast} adapter parameter tensor(s) from float32 to bfloat16.")

        evaluate_and_save(model, tokenizer, test_df, feature_cols, known_classes, args.output_dir,
                           args.max_input_len, tag="finetuned_reloaded")
        return

    train_df, test_df, feature_cols, known_classes = load_data(args.train_path, args.test_path)
    with open(args.output_dir / "known_classes.txt", "w") as f:
        f.write("\n".join(known_classes))

    print(f"Loading {args.model_name} (4-bit={use_4bit}) ...")
    model, tokenizer = load_base_model_and_tokenizer(args.model_name, use_4bit)

    if args.mode in ("zeroshot", "both"):
        few_shot_prefix = build_few_shot_prefix(train_df, feature_cols, args.shots, args.seed)
        tag = f"zeroshot_{args.shots}shot" if args.shots > 0 else "zeroshot"
        evaluate_and_save(model, tokenizer, test_df, feature_cols, known_classes, args.output_dir,
                           args.max_input_len, tag, few_shot_prefix=few_shot_prefix)

    if args.mode in ("finetune", "both"):
        model = run_finetune(model, tokenizer, train_df, feature_cols, args)
        evaluate_and_save(model, tokenizer, test_df, feature_cols, known_classes, args.output_dir,
                           args.max_input_len, tag="finetuned")


if __name__ == "__main__":
    main()