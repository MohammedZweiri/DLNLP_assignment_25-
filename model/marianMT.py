"""Accomplishing Arabic to English Machine Translation via marianMT pretrained model.

    """

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import MarianTokenizer, MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, MarianConfig, MBartForConditionalGeneration, MBart50TokenizerFast
from datasets import Dataset
from evaluate import load
import re
import matplotlib.pyplot as plt
import torch



def tokenize_data(examples):
    """
    This function tokenizes both arabic and english texts data.

    Args:
        Datasets
    Outputs:
        tokenized datasets
    """

    try:

        # Load tokenizer
        model_name = "Helsinki-NLP/opus-mt-ar-en"
        tokenizer = MarianTokenizer.from_pretrained(model_name)

        # Perform tokenization
        inputs = tokenizer(examples['arabic'], max_length=128, padding='max_length', truncation=True)
        targets = tokenizer(examples['english'], max_length=128, padding='max_length', truncation=True)

        inputs['labels'] = targets['input_ids']

        return inputs

    except Exception as e:
        print(f"❌ Tokenization process failed. Error: {e}")



def data_preparation(train_df, val_df, test_df):
    """
    This function prepares texts data for tokenization

    Args:
        Datasets
    Outputs:
        tokenized datasets
    """

    try:

        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)

        train_dataset = train_dataset.map(tokenize_data, batched=True)
        val_dataset = val_dataset.map(tokenize_data, batched=True)
        test_dataset = test_dataset.map(tokenize_data, batched=True)

        return train_dataset, val_dataset, test_dataset

    except Exception as e:
        print(f"❌ Data preparation failed. Error: {e}")



def training_convergence(train_dataset, val_dataset):
    """
    This function  performs convergence training for marianMT

    Args:
        Training and Validation datasets
    Outputs:
        
    """

    try:

        model_name = "Helsinki-NLP/opus-mt-ar-en"

        # Load marianMT tokenizer
        tokenizer = MarianTokenizer.from_pretrained(model_name)

        # Load marianMT model
        model = MarianMTModel.from_pretrained(model_name)

        # Training options
        training_args = Seq2SeqTrainingArguments(
            output_dir="./results01",
            #overwrite_output_dir=True,
            eval_strategy="epoch",
            # save_strategy="epoch",
            metric_for_best_model="eval_loss",
            learning_rate=1e-2,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=20,
            load_best_model_at_end=False,
            weight_decay=0.1,
            logging_dir='./logs',
            predict_with_generate=True,
            save_total_limit=1
        )

        # Training run options
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # Train the model
        trainer.train()

        # Output the model struture
        print(model)

        # Plot the training and validation loss graph
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
        plt.savefig(f'./figures/mbartMTModel_accuracy_loss_1.png')

    except Exception as e:
        print(f"❌ Model convergence training failed. Error: {e}")



def model_evaluation(dataset):

    """
    This function evaluation for marianMT using BLEU, TER and METEOR

    Args:
        test datasets
    Outputs:
        model evaluation

       """
    

    try:

        model = "./pretrained_model/checkpoint-11500"
        # Load the model and tokenizer
        model = MarianMTModel.from_pretrained()
        tokenizer = MarianTokenizer.from_pretrained()

        # Perform model evalutation
        model.eval()

        # Separate the test set to arabic and english text
        arabic_text = dataset["arabic"]
        english_text = dataset["english"]

        # Set the batch size
        batch_size=16
        translations = []

        # 
        for i in range(0, len(arabic_text), batch_size):
            batch=arabic_text[i:i+batch_size]
            inputs=tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

            # Perform translation test
            with torch.no_grad():
                translated = model.generate(**inputs)

            decoded=tokenizer.batch_decode(translated, skip_special_tokens=True)
            translations.extend(decoded)

        # Compute BLEU score
        print("⏳ Computing BLEU scores...")
        bleu_metric = load("sacrebleu")
        bleu_score = bleu_metric.compute(predictions=translations, references=[[ref] for ref in english_text])
        print("✅ Computing BLEU completed ✅")


        # Compute TER score
        print("⏳ Computing TER scores...")
        ter_metric = load("ter")
        ter_score = ter_metric.compute(predictions=translations, references=[[ref] for ref in english_text])
        print("✅ Computing TER completed ✅")

        # Compute METEOR score
        print("⏳ Computing METEOR scores...")
        meteor_metric = load("meteor")
        meteor_score = meteor_metric.compute(predictions=translations, references=[[ref] for ref in english_text])
        print("✅ Computing METEOR completed ✅")

        print(f"BLEU score on the test set for the Arabic-English translation is: {bleu_score['score']}")
        print(f"TER score on the test set for the Arabic-English translation is: {ter_score['score']}")
        print(f"METEOR score on the test set for the Arabic-English translation is: {meteor_score['meteor']}")

    except Exception as e:
        print(f"❌ Model testing and evaluation failed. Error: {e}")




def translate():

    """
    This function  performs convergence training for marianMT

    Args:
        Training and Validation datasets
    Outputs:

       """

    try:

        # Load the trainined model
        model = MarianMTModel.from_pretrained("./pretrained_model/checkpoint-11500")
        tokenizer = MarianTokenizer.from_pretrained("./pretrained_model/checkpoint-11500")

        # Perform model evaluation
        model.eval()

        # Make the model attempt to translate three arabic sentences. Their original english translation is included for sanity check.
        arabic_sentences = ["الطلاب يدرسون بجد استعدادًا للامتحانات النهائية.","الذكاء الاصطناعي يغير العالم بسرعة.",  "أطلقت الشركة منتجًا جديدًا يستخدم الذكاء الاصطناعي لتحسين تجربة المستخدم."]
        english_sentences = ["The students are studying hard getting ready for the final exams", "Aritificial Intelligence is rapidly changing the world.", "A company has launched a new product aimed at improving user's experience"]

        for arabic_text in arabic_sentences:

            i = 0
            inputs=tokenizer(arabic_text, return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                translated = model.generate(**inputs, num_beams=5,
            length_penalty=1.0,
            early_stopping=True)

            english_translator=tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

            print(f"Arabic Sentence: {arabic_text}")
            print(f"Original English Translation: {english_sentences[i]}")
            print(f"MarianMT fine-tuned model English translation: {english_translator}")

            i = i+1


    except Exception as e:
        print(f"❌ Arabic to English translation failed. Error: {e}")