"""Accomplishing Task B via Convolutional Neural Networks.

    This module acquires BlooddMNIST data from medmnist library, then it uses the CNN model to accuractly predict the 8 different
    classes of the blood diseases.

    """
from tensorflow import keras
import tensorflow as tf
from keras.utils import  plot_model
from keras.models import Model
from keras.layers import Input, Embedding, Dense, LayerNormalization, MultiHeadAttention, Dropout
import numpy as np
from keras_nlp.layers import TokenAndPositionEmbedding, TransformerEncoder
from keras_nlp.layers import TransformerDecoder
from keras.optimizers import Adam
from src import utils





# def positional_encoding(length, depth):

#     depth = depth/2
#     positions = np.arange(length)[:, np.newaxis]
#     depths = np.arange(depth)[np.newaxis, :]/depth
#     angle_rates = 1 / (1000**depths)
#     angle_rads = positions * angle_rates
#     pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

#     return tf.cast(pos_encoding, dtype=tf.float32)


# def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):

#     inputs = Input(shape=(None, d_model), name=f"{name}_inputs")

#     # Self-attention
#     attention = MultiHeadAttention(
#         num_heads=num_heads, key_dim=d_model // num_heads, name=f"{name}_attention"
#     )(inputs, inputs)
#     attention = Dropout(dropout, name=f"{name}_attention_dropout")(attention)
#     attention = LayerNormalization(epsilon=1e-6, name=f"{name}_attention_layernorm")(inputs + attention)

#     # Feed-forward
#     outputs = Dense(units, activation="relu", name=f"{name}_dense_1")(attention)
#     outputs = Dense(d_model, name=f"{name}_dense_2")(outputs)
#     outputs = Dropout(dropout, name=f"{name}_outputs_dropout")(outputs)
#     outputs = LayerNormalization(epsilon=1e-6, name=f"{name}_outputs_layernorm")(attention + outputs)

#     return Model(inputs=inputs, outputs=outputs, name=name)


# def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):

#     inputs = Input(shape=(None, d_model), name=f"{name}_inputs")
#     enc_outputs = Input(shape=(None, d_model), name=f"{name}_encoder_outputs")

#     # Self-attention
#     attention1 = MultiHeadAttention(
#         num_heads=num_heads, key_dim=d_model // num_heads, name=f"{name}_attention_1"
#     )(inputs, inputs, use_causal_mask=True)
#     attention1 = LayerNormalization(epsilon=1e-6, name=f"{name}_attention_1_layernorm")(inputs + attention1)

#     # Cross-attention
#     attention2 = MultiHeadAttention(
#         num_heads=num_heads, key_dim=d_model // num_heads, name=f"{name}_attention_2"
#     )(attention1, enc_outputs)
#     attention2 = Dropout(dropout, name=f"{name}_attention_dropout")(attention2)
#     attention2 = LayerNormalization(epsilon=1e-6, name=f"{name}_attention_layernorm")(attention1 + attention2)


#     # Feed-forward
#     outputs = Dense(units, activation="relu", name=f"{name}_dense_1")(attention2)
#     outputs = Dense(d_model, name=f"{name}_dense_2")(outputs)
#     outputs = Dropout(dropout, name=f"{name}_outputs_dropout")(outputs)
#     outputs = LayerNormalization(epsilon=1e-6, name=f"{name}_outputs_layernorm")(attention2 + outputs)

#     return Model(inputs=[inputs, enc_outputs], outputs=outputs, name=name)


# def transformer(
#     arabic_vocab_size,
#     english_vocab_size,
#     max_len,
#     embedding_dim = 256,
#     num_heads = 8,
#     ff_dim = 2048,
#     num_encoder_layers = 6,
#     num_decoder_layers = 6,
#     dropout_rate=0.1
# ):
    
#     # Encoder
#     print("Encoding\n")
#     encoder_inputs = Input(shape=(max_len,), name="encoder_inputs")
#     encoder_embedding = Embedding(arabic_vocab_size, embedding_dim, name="encoder_embedding")(encoder_inputs)

#     # Add positional encoding
#     print("Add positional encoding\n")
#     pos_encoding = positional_encoding(max_len, embedding_dim)
#     encoder_embedding = encoder_embedding + pos_encoding[:max_len, :]

#     encoder_outputs = Dropout(dropout_rate)(encoder_embedding)

#     # Stack encoder layers
#     print("Stack encoder layers\n")
#     for i in range(num_encoder_layers):
#         encoder_layer_instance = encoder_layer(
#             ff_dim, embedding_dim, num_heads, dropout_rate, name=f"encoder_layer_{i}"
#         )
#         encoder_outputs = encoder_layer_instance(encoder_outputs)

#     # Decoder
#     print("Decoders\n")
#     decoder_inputs = Input(shape=(max_len - 1,), name="decoder_inputs")
#     decoder_embedding = Embedding(english_vocab_size, embedding_dim, name="decoder_embedding")(decoder_inputs)

#     # Add positional decoding
#     print("Add positional decoding\n")
#     decoder_embedding = decoder_embedding + pos_encoding[:max_len-1, :]
#     decoder_outputs = Dropout(dropout_rate)(decoder_embedding)

#     # Stack decoder layers
#     print("Stack decoder layers\n")
#     for i in range(num_decoder_layers):
#         decoder_layer_instance = decoder_layer(
#             ff_dim, embedding_dim, num_heads, dropout_rate, name=f"decoder_layer_{i}"
#         )
#         decoder_outputs = decoder_layer_instance([decoder_outputs, encoder_outputs])

#     # Final output layer
#     print("Final output layer\n")
#     outputs = Dense(english_vocab_size, activation="softmax")(decoder_outputs)

#     model = Model([encoder_inputs, decoder_inputs], outputs)

#     return model



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



def transformer_model_training(inputs, outputs, arabic_vocab_size, english_vocab_size, sequence_len):
    """CNN model training

    This function trains the CNN model and tests it on the dataset. Then, it will evaluate it and produce the accuracy and plot loss.

    Args:
            training, validation and test datasets.
    

    """

    try:
            

        # CNN model
        np.random.seed(42)
        tf.random.set_seed(42)

        # Parameters
        embedding_dim = 256
        num_heads = 8
        # ff_dim = 2048
        # dropout_rate = 0.1
        print("Encoding Start!!")
        encoder_input = Input(shape=(None,), dtype='int64', name='encoder_input')
        x = TokenAndPositionEmbedding(arabic_vocab_size, sequence_len, embedding_dim)(encoder_input)
        encoder_output = TransformerEncoder(embedding_dim, num_heads)(x)
        encoder_seq_input = Input(shape=(None, embedding_dim))

        
        decoder_input = Input(shape=(None,), dtype='int64', name='decoder_input')
        x = TokenAndPositionEmbedding(english_vocab_size, sequence_len, embedding_dim, mask_zero=True)(decoder_input)
        x = TransformerDecoder(embedding_dim, num_heads)(x, encoder_seq_input)
        #x = Dropout()(x)

        decoder_output = Dense(english_vocab_size, activation='softmax')(x)
        decoder = Model([decoder_input, encoder_seq_input], decoder_output)
        decoder_output = decoder([decoder_input, encoder_output])

        model = Model([encoder_input, decoder_input], decoder_output)

        # Output the model summary
        print(model.summary())

        # Plot the CNN model
        plot_model(model, 
                to_file='./figures/Transformers_Model_test_1.png', 
                show_shapes=True,
                show_layer_activations=True)


        print("Compile the model")
        # Compile the CNN model
        model.compile(loss='sparse_categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])
        

        print("Call backs")
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
        # Fit the CNN model
        print("Fit the model")
        history = model.fit(inputs, 
                            outputs,
                            validation_split=0.2,
                            epochs=20,
                            callbacks=callbacks,
                            batch_size=32)
        
        # save the CNN model
        utils.save_model(model, "Transformers_model_task_final_1")

        # Evaluate the model
        utils.plot_accuray_loss(history)

    except Exception as e:
        print(f"Training and saving the Transformer model failed. Error: {e}")


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