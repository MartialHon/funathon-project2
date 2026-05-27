

# %%
## 1. Initialisation
import mlflow
import random
import s3fs

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchTextClassifiers.value_encoder import ValueEncoder
from torchTextClassifiers.tokenizers import WordPieceTokenizer
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers

load_dotenv(override=True)

## 2. Load data
df = pl.read_parquet('https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet')
# %%
print(df[0])
print(len(df))
code, count = df['code'].value_counts()

plt.plot(code, count, 'o')
plt.show()

# %%
n_classes = len(df['code'].unique())
print(n_classes)

# %%

## 3. Split data
train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df = train_test_split(tmp_df, test_size=0.50, random_state=42)

X_train, y_train = train_df["label"].to_numpy(), train_df["code"].to_numpy()
X_val, y_val = val_df["label"].to_numpy(), val_df["code"].to_numpy()
X_test, y_test = test_df["label"].to_numpy(), test_df["code"].to_numpy()

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

assert len(np.unique(y_train)) == n_classes
# %%

print(X_train.shape, X_test.shape, X_val.shape)
print(len(X_train)/len(df), len(X_test)/len(df), len(X_val)/len(df))
# %%
print(y_train[0])

code, count = df['code'].value_counts()

plt.plot(code, count, 'o')
plt.show()

# %%
label_encoder = LabelEncoder().fit(y_train)
value_encoder = ValueEncoder(label_encoder=label_encoder,)
# %%

tokenizer = WordPieceTokenizer(
    vocab_size=10000,
    output_dim=10
)
tokenizer.train(X_train)

print("Output tensor size:", tokenizer.tokenize(X_train[0]).input_ids.shape)
print("Vocabulary size:", tokenizer.vocab_size)

# %%
print("Output tensor size:", tokenizer.tokenize(X_train[0]).input_ids.shape)

# Look at an example of tokenization
print("Raw text", X_train[0])
print(
    "Tokens id:",
    tokenizer.tokenize(X_train[0]).input_ids.squeeze(0)
)
print(
    "Tokens:",
    tokenizer.tokenizer.convert_ids_to_tokens(
        tokenizer.tokenize(X_train[0]).input_ids.squeeze(0)
    )
)
# %%
##5. Model architecture

## 6. Training


model_config = ModelConfig(
    embedding_dim=96,
    num_classes=n_classes
)

ttc = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
    value_encoder=value_encoder,
)

train_config = TrainingConfig(
    lr=5 * 1e-4,
    batch_size=128,
    num_epochs=1)

# %%
#ttc.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, training_config=train_config)
# %%


fs = s3fs.S3FileSystem(
    anon=True,  # public bucket
    endpoint_url="https://minio.lab.sspcloud.fr",
)

local_dir = "./mlflow-artifacts/"
fs.get(
    "projet-funathon/diffusion/mlflow-artifacts/",
    local_dir,
    recursive=True,
)
# Rebuild the torchTextClassifiers object from the downloaded files
ttc = torchTextClassifiers.load(local_dir)

ttc.pytorch_model.eval()

# %%

## With mlflow
"""
mlflow.set_experiment("nace-codification")
mlflow.pytorch.autolog()

with mlflow.start_run():
    ttc.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, training_config=train_config)
"""

# %%

## 7 Prediction and explainability
"""
test
"""
random_indices = random.sample(range(len(X_test)), 3)
example_texts = X_test[random_indices]
example_true_codes = y_test[random_indices]
print(example_texts)
top_k = 5
results = ttc.predict(example_texts, top_k=top_k, explain_with_captum=True)
for i, text in enumerate(example_texts):
    predicted_codes = [results["prediction"][i][k] for k in range(top_k)]
    confidence = [results["confidence"][i][k].item() for k in range(top_k)]
    print(f"\nText: {text}")
    print(f"  True code: {example_true_codes[i]}")
    for code, conf in zip(predicted_codes, confidence):
        print(f"  {code}  (confidence: {conf:.3f})")
# %%
print(X_train)

# %%
from torchTextClassifiers.utilities.plot_explainability import (
    map_attributions_to_char, map_attributions_to_word,
    plot_attributions_at_char, plot_attributions_at_word, figshow,
)

text_idx = 0
top_k_idx = 0
text_sample         = example_texts[text_idx]
offsets             = results["offset_mapping"][text_idx]
word_ids            = results["word_ids"][text_idx]
predicted_code = results["prediction"][text_idx][top_k_idx]

attributions  = results["captum_attributions"][text_idx][top_k_idx] # (seq_len,)

words, word_attributions = map_attributions_to_word(
    attributions.unsqueeze(0), text_sample, word_ids, offsets
)
char_attributions = map_attributions_to_char(attributions.unsqueeze(0), offsets, text_sample)

titles = [f"Attributions for NACE code {predicted_code}"]

figshow(plot_attributions_at_char(
    text=text_sample, attributions_per_char=char_attributions, titles=titles,
)[0])

figshow(plot_attributions_at_word(
    text=text_sample, words=words.values(), attributions_per_word=word_attributions, titles=titles,
)[0])

# %%
results_test = ttc.predict(X_test, top_k=1)
preds    = results_test["prediction"].squeeze(1)
accuracy = (preds == y_test).mean()
print(f"Test accuracy: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)} correct)")
# %%
