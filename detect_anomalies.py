import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
import time
import joblib
import matplotlib.pyplot as plt
import logging
import coloredlogs

# Configuring logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,
                    fmt='%(asctime)s %(levelname)s %(message)s')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Global constants
SEQ_LENGTH = 60
TRAIN_END_DATE = '2023-12-31'
HEAD_SIZE = 256
NUM_HEADS = 4
FF_DIM = 4
NUM_TRANSFORMER_BLOCKS = 4
MLP_UNITS = [128]
DROPOUT = 0.1
MLP_DROPOUT = 0.1
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

DATA_DIR = 'data'
MODELS_DIR = 'models'
PROCESSED_DATA_DIR = 'processed_data'

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Configuration for multiprocessing
strategy = tf.distribute.MultiWorkerMirroredStrategy()

def load_data(file_path):
    """
    Loads data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with the loaded data.
    """
    logger.info("Loading price data")
    return pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')

def select_features(df):
    """
    Selects the features to be used for training the model.

    Args:
        df (pd.DataFrame): DataFrame with the original data.

    Returns:
        pd.DataFrame: DataFrame with the selected features.
    """
    logger.info("Selecting features")
    features = ['log_return', 'volume', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl', 
                'volatility_kcc', 'volatility_kch', 'volatility_kcl', 'volatility_atr', 
                'trend_macd', 'trend_macd_signal', 'trend_macd_diff']
    ret_df = df[features]
    ret_df.dropna(inplace=True)

    return ret_df

def load_aggregated_data(timeframe):
    """
    Loads aggregated data for a given timeframe from a CSV file.

    Args:
        timeframe (str): The timeframe for the aggregated data.

    Returns:
        pd.DataFrame: DataFrame with the aggregated data.
    """
    file_path = os.path.join(DATA_DIR, f'bitcoin_{timeframe}_data.csv')
    if os.path.exists(file_path):
        logger.info(f"Loading aggregated data for {timeframe} timeframe")
        return pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
    else:
        logger.error(f"Aggregated data file for {timeframe} timeframe not found")
        return None

def merge_datasets(df, aggregated_dfs):
    """
    Merges the original dataset with aggregated datasets.

    Args:
        df (pd.DataFrame): The original dataframe.
        aggregated_dfs (dict): Dictionary with aggregated dataframes.

    Returns:
        pd.DataFrame: The merged dataframe.
    """
    logger.info("Merging datasets")
    for timeframe, agg_df in aggregated_dfs.items():
        agg_df = agg_df.add_suffix(f'_{timeframe}')
        df = df.join(agg_df, how='left')
    df.ffill(inplace=True)  # Forward fill any NaN values
    df.bfill(inplace=True)  # Backward fill any remaining NaN values
    return df

def save_datasets(train_data, valid_data, test_data):
    """
    Saves the datasets to CSV files.

    Args:
        train_data (pd.DataFrame): Training data.
        valid_data (pd.DataFrame): Validation data.
        test_data (pd.DataFrame): Test data.
    """
    logger.info("Saving datasets to CSV files")
    train_data.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train_data.csv'))
    valid_data.to_csv(os.path.join(PROCESSED_DATA_DIR, 'valid_data.csv'))
    test_data.to_csv(os.path.join(PROCESSED_DATA_DIR, 'test_data.csv'))
    logger.info("Datasets saved to CSV files")

def load_datasets():
    """
    Loads the datasets from CSV files if they exist.

    Returns:
        tuple: DataFrames for training, validation, and test sets.
    """
    train_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_data.csv'), index_col='timestamp', parse_dates=['timestamp'])
    valid_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'valid_data.csv'), index_col='timestamp', parse_dates=['timestamp'])
    test_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_data.csv'), index_col='timestamp', parse_dates=['timestamp'])
    logger.info("Datasets loaded from CSV files")
    return train_data, valid_data, test_data

def apply_pca(df, n_components, dataset_name):
    """
    Applies PCA to reduce the dimensionality of the data.

    Args:
        df (pd.DataFrame): DataFrame with the preprocessed data.
        n_components (int): Number of principal components.
        dataset_name (str): Name of the dataset being processed.

    Returns:
        np.ndarray: Transformed data after applying PCA.
    """
    logger.info(f"Applying PCA on {dataset_name} dataset")
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(df)
    joblib.dump(pca, os.path.join(MODELS_DIR, f'pca_{dataset_name}.pkl'))
    logger.info(f"Saved PCA model to {os.path.join(MODELS_DIR, f'pca_{dataset_name}.pkl')}")
    return transformed_data

def determine_pca_components(df):
    """
    Determines the number of PCA components that explain at least 95% of the variance.

    Args:
        df (pd.DataFrame): DataFrame with the preprocessed data.

    Returns:
        int: Number of PCA components.
    """
    pca = PCA().fit(df)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    logger.info("Explained Variance Ratio:")
    logger.info(cumulative_variance)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    logger.info(f"Number of PCA components explaining 95% variance: {n_components}")
    return n_components

def split_data(df, train_end_date):
    """
    Splits the data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with the preprocessed data.
        train_end_date (str): End date for the training set.

    Returns:
        tuple: DataFrames for training and test sets.
    """
    logger.info("Splitting data into training and test sets")
    train_data = df[:train_end_date]
    test_data = df[train_end_date:]
    return train_data, test_data

def normalize_data(train_data, valid_data, test_data):
    """
    Normalizes the data.

    Args:
        train_data (pd.DataFrame): Training data.
        valid_data (pd.DataFrame): Validation data.
        test_data (pd.DataFrame): Test data.

    Returns:
        tuple: Normalized numpy arrays for training, validation, and test sets.
    """
    logger.info("Normalizing data")
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    valid_data = scaler.transform(valid_data)
    test_data = scaler.transform(test_data)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    logger.info(f"Saver scaler to {os.path.join(MODELS_DIR, 'scaler.pkl')}")
    return train_data, valid_data, test_data

def create_tf_dataset(data, seq_length, batch_size, dataset_name):
    """
    Creates a tf.data.Dataset from numpy arrays.

    Args:
        data (np.ndarray): Input data.
        seq_length (int): Sequence length for each sample.
        batch_size (int): Batch size for the dataset.
        dataset_name (str): Name of the dataset being created.

    Returns:
        tf.data.Dataset: TensorFlow dataset.
    """
    logger.info(f"Creating {dataset_name} dataset")
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    X, y = np.array(X), np.array(y)
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.cache().shuffle(buffer_size=len(X)).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

class TransformerBlock(layers.Layer):
    """
    Transformer block for the model.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class Time2Vector(layers.Layer):
    """
    Time2Vector layer to capture temporal information.
    """
    def __init__(self, seq_len):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        self.weights_linear = self.add_weight(name='weight_linear',
                                              shape=(int(self.seq_len),),
                                              initializer='uniform',
                                              trainable=True)
        self.bias_linear = self.add_weight(name='bias_linear',
                                           shape=(int(self.seq_len),),
                                           initializer='uniform',
                                              trainable=True)
        self.weights_periodic = self.add_weight(name='weight_periodic',
                                                shape=(int(self.seq_len),),
                                                initializer='uniform',
                                                trainable=True)
        self.bias_periodic = self.add_weight(name='bias_periodic',
                                             shape=(int(self.seq_len),),
                                             initializer='uniform',
                                             trainable=True)

    def call(self, x):
        x = tf.math.reduce_mean(x[:,:,:4], axis=-1)
        time_linear = self.weights_linear * x + self.bias_linear
        time_linear = tf.expand_dims(time_linear, axis=-1)
        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1)
        return tf.concat([time_linear, time_periodic], axis=-1)

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout, mlp_dropout):
    """
    Builds and compiles the Transformer model.

    Args:
        input_shape (tuple): Shape of the input data.
        head_size (int): Dimension of the attention layer.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimension of the feed-forward network.
        num_transformer_blocks (int): Number of Transformer blocks.
        mlp_units (list): Units of the dense layers.
        dropout (float): Dropout rate.
        mlp_dropout (float): Dropout rate in MLP.

    Returns:
        tf.keras.Model: Compiled model.
    """
    inputs = layers.Input(shape=input_shape)
    time_embedding = Time2Vector(SEQ_LENGTH)(inputs)
    x = layers.Concatenate(axis=-1)([inputs, time_embedding])
    x = layers.Dense(head_size, dtype='float32')(x)  # Embedding to the same dimension as head_size
    for _ in range(num_transformer_blocks):
        x = TransformerBlock(head_size, num_heads, ff_dim, dropout)(x, training=True)

    x = layers.GlobalAveragePooling1D()(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(input_shape[-1])(x)
    return Model(inputs, outputs)

def plot_loss(history):
    """
    Plots the training and validation loss.

    Args:
        history (tf.keras.callbacks.History): Training history.
    """
    logger.info("Plotting training and validation loss")
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

def main():
    """
    Main function to execute the process of loading data, preprocessing,
    generating and training the model, and visualization.
    """
    logger.info("Starting main function")
    
    # Check if processed datasets exist
    train_data_path = os.path.join(PROCESSED_DATA_DIR, 'train_data.csv')
    valid_data_path = os.path.join(PROCESSED_DATA_DIR, 'valid_data.csv')
    test_data_path = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')

    if os.path.exists(train_data_path) and os.path.exists(valid_data_path) and os.path.exists(test_data_path):
        train_data_split, valid_data_split, test_data_split = load_datasets()
    else:
        # Load and preprocess data
        df = load_data(os.path.join(DATA_DIR, 'bitcoin_1min_data.csv'))
        df = select_features(df)

        # Load aggregated data
        timeframes = ['5min', '15min', '1h', '4h', '1d']
        #Refactor this
        aggregated_dfs = {timeframe: select_features(load_aggregated_data(timeframe)) for timeframe in timeframes}
        
        # Merge datasets
        df = merge_datasets(df, aggregated_dfs)
        
        train_data, test_data = split_data(df, TRAIN_END_DATE)

        # Split training data into training, validation, and test sets (70%, 15%, 15%)
        logger.info("Splitting training data into training, validation, and test sets")
        train_size = int(len(train_data) * 0.7)
        valid_size = int(len(train_data) * 0.15)

        train_data_split = train_data[:train_size]
        valid_data_split = train_data[train_size:train_size + valid_size]
        test_data_split = train_data[train_size + valid_size:]

        # Save the datasets
        save_datasets(train_data_split, valid_data_split, test_data_split)
    
    # Normalize data
    train_data_split, valid_data_split, test_data_split = normalize_data(train_data_split, valid_data_split, test_data_split)

    # Determine number of PCA components
    n_components = determine_pca_components(train_data_split)

    # Apply PCA
    train_data_split = apply_pca(train_data_split, n_components, 'train')
    valid_data_split = apply_pca(valid_data_split, n_components, 'validation')
    test_data_split = apply_pca(test_data_split, n_components, 'test')

    # Create datasets
    train_dataset = create_tf_dataset(train_data_split, SEQ_LENGTH, BATCH_SIZE, 'train')
    valid_dataset = create_tf_dataset(valid_data_split, SEQ_LENGTH, BATCH_SIZE, 'validation')
    test_dataset = create_tf_dataset(test_data_split, SEQ_LENGTH, BATCH_SIZE, 'test')

    with strategy.scope():
        input_shape = (SEQ_LENGTH, n_components)
        model = build_model(input_shape, HEAD_SIZE, NUM_HEADS, FF_DIM, NUM_TRANSFORMER_BLOCKS, MLP_UNITS, DROPOUT, MLP_DROPOUT)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")

        # Print model summary
        model.summary()

        # Define Early Stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        )

        # Measure training time
        start_time = time.time()

        logger.info("Training model")
        # Train the model with Early Stopping and multiple workers
        history = model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=EPOCHS,
            callbacks=[early_stopping],
            verbose=1
        )

        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"Training Time: {training_time:.2f} seconds")

        # Save the model
        model.save(os.path.join(MODELS_DIR, 'transformer_model'))
        logger.info(f"Saved model to {os.path.join(MODELS_DIR, 'transformer_model')}")

    # Evaluate the model on the test set
    logger.info("Evaluating the model on the test set")
    test_loss = model.evaluate(test_dataset)
    logger.info(f'Test Loss: {test_loss}')

    # Plot training and validation loss
    plot_loss(history)

if __name__ == "__main__":
    main()