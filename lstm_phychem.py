import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, balanced_accuracy_score

def train_lstm_model(x_train, x_test, y_train, y_test, model_filename='lstm.keras', plot_filename='training_lost.png', epochs=50, batch_size=2):

    #convert data to numpy
    x_train = np.array(x_train)
    x_test  = np.array(x_test)
    y_train = np.array(y_train)
    y_test  = np.array(y_test)

    # Reshape data for LSTM: (samples, timesteps, features)
    num_features = x_train.shape[1]  # Number of features (length of ECFP vector)
    x_train = x_train.reshape((x_train.shape[0], 1, num_features))
    x_test = x_test.reshape((x_test.shape[0], 1, num_features))

    # Split data into training and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check the shapes of the datasets
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(1, num_features), return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # For binary classification
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.3)
    model.save(model_filename)

    # Get prediction probabilities for the test set
    y_test_pred_prob = model.predict(x_test).ravel()  # Flatten the array
    y_train_pred_prob = model.predict(x_train).ravel()  # Flatten the array

    # Convert probabilities to class predictions
    y_test_pred = (y_test_pred_prob > 0.5).astype(int)
    y_train_pred = (y_train_pred_prob > 0.5).astype(int)
    
    # Compute the metrics
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    mcc_train = matthews_corrcoef(y_train, y_train_pred)
    mcc_test = matthews_corrcoef(y_test, y_test_pred)
    precision_train = precision_score(y_train, y_train_pred)
    precision_test  = precision_score(y_test,  y_test_pred)
    recall_train    = recall_score(y_train, y_train_pred)
    recall_test     = recall_score(y_test,  y_test_pred)
    # Compute the confusion matrix for the training and test sets
    tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_train_pred).ravel()
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_test_pred).ravel()

    # Calculate specificity for train and test sets
    specificity_train = tn_train / (tn_train + fp_train)
    specificity_test = tn_test / (tn_test + fp_test)

    # Compute ROC curve data
    fpr_test, tpr_test, thresholds = roc_curve(y_test, y_test_pred_prob)
    fpr_train, tpr_train, thresholds = roc_curve(y_train, y_train_pred_prob)
    # Compute AUC
    roc_auc_test = auc(fpr_test, tpr_test)
    roc_auc_train = auc(fpr_train, tpr_train)
    return accuracy_train, accuracy_test,mcc_train,mcc_test,precision_train,precision_test,recall_train,recall_test, specificity_train,specificity_test,roc_auc_test,roc_auc_train, model_filename, plot_filename
    
def main():
    # Load data from CSV
    x_train = pd.read_csv("corrosion_x_train_descriptors_shuffled.csv",index_col=0)
    x_test = pd.read_csv("corrosion_x_test_descriptors_shuffled.csv",index_col=0)
    y_train = pd.read_csv("corrosion_y_train_numeric.csv",index_col=0)
    y_test = pd.read_csv("corrosion_y_test_numeric.csv",index_col=0)
    accuracy_train, accuracy_test,mcc_train,mcc_test,precision_train,precision_test,recall_train,recall_test, specificity_train,specificity_test, roc_auc_test,roc_auc_train, model_file, plot_file = train_lstm_model(x_train, x_test, y_train, y_test)
    # print evaluation metrics
    print(f'Accuracy Train: {accuracy_train:.3f}')
    print(f'Accuracy Test: {accuracy_test:.3f}')
    print(f'Train Matthews Correlation Coefficient (MCC): {mcc_train:.3f}')
    print(f'Test Matthews Correlation Coefficient (MCC): {mcc_test:.3f}')
    print(f'Precison Train: {precision_train:.3f}')
    print(f'Precison Test: {precision_test:.3f}')
    print(f'Sensitivity (Recall) Train: {recall_train:.3f}')
    print(f'Sensitivity (Recall) Test: {recall_test:.3f}')
    print(f'Train ROC AUC: {roc_auc_train:.3f}')
    print(f'Test ROC AUC: {roc_auc_test:.3f}')
    print(f"Train Specificity: {specificity_train:.3f}")
    print(f"Test Specificity: {specificity_test:.3f}")
    
if __name__ =="__main__":
    main()
