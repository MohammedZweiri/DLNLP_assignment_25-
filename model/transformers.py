"""Accomplishing Task B via Convolutional Neural Networks.

    This module acquires BlooddMNIST data from medmnist library, then it uses the CNN model to accuractly predict the 8 different
    classes of the blood diseases.

    """

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, MultiHeadAttention, Dropout
from keras.optimizers import Adam


# Parameters
embedding_dim = 256
num_heads = 8
ff_dim = 2048
dropout_rate = 0.1


def positional_encoding(length, depth):

    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]/depth
    angle_rates = 1 / (1000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):

    inputs = Input(shape=(None, d_model), name=f"{name}_inputs")

    # Self-attention
    attention = MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, name=f"{name}_attention"
    )(inputs, inputs)
    attention = Dropout(dropout, name=f"{name}_attention_dropout")(attention)
    attention = LayerNormalization(epsilon=1e-6, name=f"{name}_attention_layernorm")(inputs + attention)

    # Feed-forward
    outputs = Dense(units, activation="relu", name=f"{name}_dense_1")(attention)
    outputs = Dense(d_model, name=f"{name}_dense_2")(outputs)
    outputs = Dropout(dropout, name=f"{name}_outputs_dropout")(outputs)
    outputs = LayerNormalization(epsilon=1e-6, name=f"{name}_outputs_layernorm")(attention + outputs)

    return Model(inputs=inputs, outputs=outputs, name=name)


def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):

    inputs = Input(shape=(None, d_model), name=f"{name}_inputs")
    enc_outputs = Input(shape=(None, d_model), name=f"{name}_encoder_outputs")

    # Self-attention
    attention1 = MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, name=f"{name}_attention_1"
    )(inputs, inputs, use_causal_mask=True)
    attention1 = LayerNormalization(epsilon=1e-6, name=f"{name}_attention_1_layernorm")(inputs + attention1)

    # Cross-attention
    attention2 = MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, name=f"{name}_attention_2"
    )(attention1, enc_outputs)
    attention2 = Dropout(dropout, name=f"{name}_attention_dropout")(attention2)
    attention2 = LayerNormalization(epsilon=1e-6, name=f"{name}_attention_layernorm")(attention1 + attention2)


    # Feed-forward
    outputs = Dense(units, activation="relu", name=f"{name}_dense_1")(attention2)
    outputs = Dense(d_model, name=f"{name}_dense_2")(outputs)
    outputs = Dropout(dropout, name=f"{name}_outputs_dropout")(outputs)
    outputs = LayerNormalization(epsilon=1e-6, name=f"{name}_outputs_layernorm")(attention2 + outputs)

    return Model(inputs=[inputs, enc_outputs], outputs=outputs, name=name)


def transformer(
    arabic_vocab_size,
    english_vocab_size,
    max_len,
    embedding_dim = 256,
    num_heads = 8,
    ff_dim = 2048,
    num_encoder_layers = 6,
    num_decoder_layers = 6,
    dropout_rate=0.1
):
    
    # Encoder
    encoder_inputs = Input(shape=(max_len,), name="encoder_inputs")
    encoder_embedding = Embedding(arabic_vocab_size, embedding_dim, name="encoder_embedding")(encoder_inputs)

    # Add positional encoding
    pos_encoding = positional_encoding(max_len, embedding_dim)
    encoder_embedding = encoder_embedding + pos_encoding[:max_len, :]

    encoder_outputs = Dropout(dropout_rate)(encoder_embedding)

    # Stack encoder layers
    for i in range(num_encoder_layers):
        encoder_layer_instance = encoder_layer(
            ff_dim, embedding_dim, num_heads, dropout_rate, name=f"encoder_layer_{i}"
        )
        encoder_outputs = encoder_layer_instance(encoder_outputs)

    # Decoder
    decoder_inputs = Input(shape=(max_len - 1,), name="decoder_inputs")
    decoder_embedding = Embedding(english_vocab_size, embedding_dim, name="decoder_embedding")

    # Add positional encoding
    decoder_embedding = decoder_embedding + pos_encoding[:max_len-1, :]
    decoder_outputs = Dropout(dropout_rate)(decoder_embedding)

    # Stack decoder layers
    for i in range(num_decoder_layers):
        decoder_layer_instance = decoder_layer(
            ff_dim, embedding_dim, num_heads, dropout_rate, name=f"decoder_layer_{i}"
        )
        decoder_outputs = decoder_layer_instance([decoder_outputs, encoder_outputs])

    # Final output layer
    outputs = Dense(english_vocab_size, activation="softmax")(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], outputs)

    return model



def evaluate_model(true_labels, predicted_labels, predict_probs, label_names):
    """Evaluate the CNN model.

    This function evaluates the CNN model and produces classification report and confusion matrix

    Args:
            true_labels
            predicted_labels
            predict_probs
            label_names

    """

    try:

        if(true_labels.ndim==2):
            true_labels = true_labels[:,0]
        if(predicted_labels.ndim==2):
            predicted_labels=predicted_labels[:,0]
        if(predict_probs.ndim==2):
            predict_probs=predict_probs[:,0]

        # Calculates accuracry, precision, recall and f1 scores.
        print(f"Accuracy: {accuracy_score(true_labels, predicted_labels)}")
        print(f"Precision: {precision_score(true_labels, predicted_labels, average='micro')}")
        print(f"Recall: {recall_score(true_labels, predicted_labels, average='micro')}")
        print(f"F1 Score: {f1_score(true_labels, predicted_labels, average='micro')}")

        # Performs classification report
        print("Classification report : ")
        print(classification_report(true_labels, predicted_labels, target_names=label_names))

        # Generates confusion matrix
        matrix = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(10, 7), dpi=200)
        ConfusionMatrixDisplay(matrix, display_labels=label_names).plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
        plt.title("Confusion Matrix for CNN")
        plt.savefig('B/figures/Confusion_Matrix_CNN.png', bbox_inches = 'tight')

    except Exception as e:
        print(f"Evaluating the model has failed. Error: {e}")



def transformer_model_training(train_dataset, validation_dataset, test_dataset):
    """CNN model training

    This function trains the CNN model and tests it on the dataset. Then, it will evaluate it and produce the accuracy and plot loss.

    Args:
            training, validation and test datasets.
    

    """

    try:
            

        # CNN model
        model = transformer(
            arabic_vocab_size=arabic_vocab_size,
            english_vocab_size=english_vocab_size,
            max_len=max_len,
            embedding_dim=embedding_dim,
            num_heads=8,
            ff_dim=ff_dim,
            num_encoder_layers = 6,
            num_decoder_layers = 6,
            dropout_rate=0.1
        )

        # Output the model summary
        print(model.summary())

        # Plot the CNN model
        plot_model(model, 
                to_file='B/figures/CNN_Model_testB_add.png', 
                show_shapes=True,
                show_layer_activations=True)

        # Compile the CNN model
        model.compile(loss='sparse_categorical_crossentropy',
                optimizer=Adam(learning_rate=0.001),
                metrics=['accuracy'])
        
        # Handle the class imbalance.
        weights = class_imbalance_handling(train_dataset)

        # Fit the CNN model
        history = model.fit(train_dataset.imgs, train_labels, 
                epochs=40,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)],
                validation_data=(validation_dataset.imgs, val_labels),
                batch_size=32,
                shuffle=True,
                class_weight=weights)
        
        # save the CNN model
        utils.save_model("B",model, "CNN_model_taskB_final_add")

        # Evaluate the model
        test_dataset_prob = model.predict(test_dataset.imgs, verbose=0)
        test_predict_labels = np.argmax(test_dataset_prob, axis=-1)
        evaluate_model(test_dataset.labels, test_predict_labels, test_dataset_prob, class_labels)
        utils.plot_accuray_loss("B",history)

    except Exception as e:
        print(f"Training and saving the CNN model failed. Error: {e}")


def CNN_model_testing(test_dataset):
    """CNN model testing

    This function loads the final CNN model and tests it on the test dataset. Then, it will evaluate it and produce the accuracy and plot loss.

    Args:
            training, validation and test datasets.
    

    """

    try:
            
        # Class labels
        class_labels = ["Cassava Bacterial Blight (CBB)",
                "Cassava Brown Streak Disease (CBSD)",
                "Cassava Green Mottle (CGM)",
                "Cassava Mosaic Disease (CMD)",
                "Healthy"]

        # Load the CNN model
        model = utils.load_model("B","CNN_model_taskB_final")

        # Output the model summary
        print(model.summary())

        # Evaluate the model
        test_dataset_prob = model.predict(test_dataset.imgs, verbose=0)
        test_predict_labels = np.argmax(test_dataset_prob, axis=-1)
        evaluate_model(test_dataset.labels, test_predict_labels, test_dataset_prob, class_labels)

    except Exception as e:
        print(f"Loading and testing the CNN model failed. Error: {e}")