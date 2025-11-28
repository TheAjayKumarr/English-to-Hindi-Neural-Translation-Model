# English-to-Hindi Neural Translation Model  
*A Transformer-based NMT system using Hugging Face, TensorFlow & IITB EN–HI Dataset*

---

## 📌 Overview  
This project builds an **English → Hindi Neural Machine Translation (NMT)** model using the **Helsinki-NLP/opus-mt-en-hi** checkpoint.  
It fine-tunes the model on the **IITB English–Hindi parallel corpus (1.6M sentence pairs)** to improve translation accuracy.

The pipeline includes:

- Dataset loading & preprocessing  
- Tokenization using MarianMT tokenizer  
- Fine-tuning the seq2seq model with TensorFlow  
- Saving & testing the final model  

---

## 📂 Dataset  
**IITB English–Hindi Dataset**  
🔗 https://huggingface.co/datasets/cfilt/iitb-english-hindi  

| Split | Samples |
|-------|---------|
| Train | 1,659,083 |
| Validation | 520 |
| Test | 2,507 |

Example format:

```python
{
  "translation": {
    "en": "English sentence",
    "hi": "हिंदी वाक्य"
  }
}
```
## 🧠 Model

Base Model: Helsinki-NLP/opus-mt-en-hi
🔗 https://huggingface.co/Helsinki-NLP/opus-mt-en-hi

Architecture:

- Transformer encoder–decoder
- MarianMT (Seq2Seq)
- Pretrained on OPUS

🔧 Installation
```python
pip install datasets transformers[sentencepiece] sacrebleu
```

## 🛠️ Preprocessing

Tokenization & truncation (max length: 128):
```python
def preprocess_function(examples):
    inputs = [ex["translation"]["en"] for ex in examples["translation"]]
    targets = [ex["translation"]["hi"] for ex in examples["translation"]]

    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

```
Apply preprocessing:
```python
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
```
## 🏋️ Training

Training config:
```python
batch_size = 16
learning_rate = 2e-5
weight_decay = 0.01
epochs = 15
```

Prepare datasets:
```python
train_dataset = model.prepare_tf_dataset(...)
validation_dataset = model.prepare_tf_dataset(...)
```

Train:
```python
model.fit(train_subset, validation_data=validation_dataset, epochs=15)
```
## 💾 Saving the Model
```python
model.save_pretrained("tf_model/")
```
## 🔍 Inference Example
```python
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
model = TFAutoModelForSeq2SeqLM.from_pretrained("tf_model/")
text = "Hey, this is Ajay Kumar and team, with our project"
tokens = tokenizer([text], return_tensors="np")
output = model.generate(**tokens, max_length=128)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Sample Output:
```python
अरे, यह हमारी परियोजना के साथ अजय कुमार और टीम है।
```
