import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import MarianTokenizer, MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, MarianConfig
from datasets import Dataset
from evaluate import load
import re
import matplotlib.pyplot as plt
import torch

def load_split_dataset(dataset):

    # load the dataset
    data = pd.read_csv(dataset, sep='\t', names=['english', 'arabic'])

    # Drop any rows that contain NaN values
    data.dropna(inplace=True)

    # clean the data
    data['arabic'] = data['arabic'].apply(clean_arabic)

    # Split data
    train_df, temp_df = train_test_split(data, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print("Training samples:", len(train_df))
    print("Validation samples:", len(val_df))
    print("Test samples:", len(test_df))

    return train_df, val_df, test_df

def clean_arabic(text):

    # Remove diacritics
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)

    # Remove tatweel (kashida)
    text = re.sub(r'\u0640', '', text)

    # Remove non-arabic characters
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)

    # Normalize arabic letters
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'ء', text)
    text = re.sub(r'ئ', 'ء', text)
    text = re.sub(r'ة', 'ه', text)

    return text


def tokenize_data(examples):


    model_name = "Helsinki-NLP/opus-mt-ar-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    inputs = tokenizer(examples['arabic'], max_length=128, padding='max_length', truncation=True)
    targets = tokenizer(examples['english'], max_length=128, padding='max_length', truncation=True)

    inputs['labels'] = targets['input_ids']

    return inputs
    # text_input = [str(i) for i in examples['arabic'].values]
    # return tokenizer(text_input, text_target=examples['english'], truncation=True, padding="max_length", max_length=128)


def data_preparation(train_df, val_df, test_df):

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    train_dataset = train_dataset.map(tokenize_data, batched=True)
    val_dataset = val_dataset.map(tokenize_data, batched=True)
    test_dataset = test_dataset.map(tokenize_data, batched=True)

    # train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    # val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    # test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return train_dataset, val_dataset, test_dataset


def marianMT_training(train_dataset, val_dataset, test_dataset):

    #model_name = "Helsinki-NLP/opus-mt-ar-en"
    model_name = "./marianMT/checkpoint-11500"
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Load config and modify dropout
    config = MarianConfig.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results07",
        #overwrite_output_dir=True,
        eval_strategy="epoch",
        # save_strategy="epoch",
        metric_for_best_model="eval_loss",
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=30,
        load_best_model_at_end=False,
        weight_decay=0.1,
        logging_dir='./logs',
        predict_with_generate=True,
        save_total_limit=None
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()


    print(model)

    history = trainer.state.log_history

    train_loss = [x['loss'] for x in history if 'loss' in x]
    eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]

    plt.plot(train_loss, label='Train Loss')
    plt.plot(eval_loss, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.title("Training vs Validation Loss")
    plt.savefig(f'./figures/MarianMTModel_accuracy_loss_1.png')


def model_evaluation(dataset):

    model = MarianMTModel.from_pretrained("./marianMT/checkpoint-11500")
    tokenizer = MarianTokenizer.from_pretrained("./marianMT/checkpoint-11500")

    model.eval()

    arabic_text = dataset["arabic"]
    english_text = dataset["english"]

    batch_size=16
    translations = []

    for i in range(0, len(arabic_text), batch_size):
        batch=arabic_text[i:i+batch_size]
        inputs=tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            translated = model.generate(**inputs)

        decoded=tokenizer.batch_decode(translated, skip_special_tokens=True)
        translations.extend(decoded)

    metric = load("sacrebleu")
    results = metric.compute(predictions=translations, references=[[ref] for ref in english_text])

    print(f"BLEU score on the test set for the Arabic-English translation is: {results['score']}")

def translate():

    model = MarianMTModel.from_pretrained("./marianMT/checkpoint-11500")
    tokenizer = MarianTokenizer.from_pretrained("./marianMT/checkpoint-11500")

    model.eval()

    arabic_sentences = ["الطلاب يدرسون بجد استعدادًا للامتحانات النهائية.","الذكاء الاصطناعي يغير العالم بسرعة.",  "أطلقت الشركة منتجًا جديدًا يستخدم الذكاء الاصطناعي لتحسين تجربة المستخدم."]

    for arabic_text in arabic_sentences:

        inputs=tokenizer(arabic_text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            translated = model.generate(**inputs, num_beams=5,
        length_penalty=1.0,
        early_stopping=True)

        english_translator=tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

        print(f"Arabic: {arabic_text}")
        print(f"English: {english_translator}")