from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix, roc_auc_score, mean_squared_error,r2_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


global filename
global df, X_train, X_test, y_train, y_test
global lgb_model

def upload():
    global filename, df
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Fill missing values with mode for each column
    df.fillna(df.mode().iloc[0], inplace=True)
    
    text.delete('1.0', END)
    text.insert(END, 'Dataset loaded\n')
    text.insert(END, "Dataset Size: " + str(len(df)) + "\n")

def splitdataset(): 
    global df, X_train, X_test, y_train, y_test

    # Encode string columns to numerical values
    label_encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])

    X = np.array(df.drop(["COVID-19"], axis=1))
    y = np.array(df["COVID-19"])
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)    
    # Display dataset split information
    text.delete('1.0', END)
    text.insert(END, "Dataset split\n")
    text.insert(END, "Splitted Training Size for Machine Learning : " + str(len(X_train)) + "\n")
    text.insert(END, "Splitted Test Size for Machine Learning    : " + str(len(X_test)) + "\n")
    
    # Display shapes of X_train, X_test, y_train, y_test
    text.insert(END, "\nShape of X_train: " + str(X_train.shape) + "\n")
    text.insert(END, "Shape of X_test: " + str(X_test.shape) + "\n")
    text.insert(END, "Shape of y_train: " + str(y_train.shape) + "\n")
    text.insert(END, "Shape of y_test: " + str(y_test.shape) + "\n\n")


def covid_graph():
    global df

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot countplot for COVID-19 cases on the first subplot
    sns.countplot(x='COVID-19', data=df, palette="PuRd", ax=axs[0])
    for p in axs[0].patches:
        axs[0].annotate(f'\n{p.get_height()}', (p.get_x() + 0.4, p.get_height() + 100), ha='center', va='top', color='white', size=10)
    axs[0].set_title('Count of COVID-19 Cases')

    # Plot pie chart for COVID-19 cases on the second subplot
    df["COVID-19"].value_counts().plot.pie(explode=[0.1, 0.1], autopct='%1.1f%%', shadow=True, ax=axs[1])
    axs[1].set_title('COVID-19 Cases Distribution')
    axs[1].set_ylabel('')

    # Show the combined plots
    plt.tight_layout()
    plt.show()

def breath_problem_graph():
    global df

    # Print data for debugging
    print(df.head())
    print(df['Breathing Problem'].unique())
    print(df['COVID-19'].unique())

    # Check if the columns 'Breathing Problem' and 'COVID-19' exist in the dataframe
    if 'Breathing Problem' not in df.columns or 'COVID-19' not in df.columns:
        messagebox.showerror("Error", "'Breathing Problem' or 'COVID-19' column not found in the dataset.")
        return

    # Create the count plot for Breathing Problem with COVID-19 as hue
    ax = sns.countplot(x='Breathing Problem', hue='COVID-19', data=df, palette="Set2")
    for p in ax.patches:
        ax.annotate(f'\n{p.get_height()}', (p.get_x() + 0.2, p.get_height() + 100), ha='center', va='top', color='white', size=10)
    
    plt.title('Breathing Problem vs COVID-19')
    plt.show()


def symptoms_status():
    global df
    
    # Encode string columns to numerical values
    label_encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])

    # Plot the histograms of all the features in the dataframe
    df.hist(figsize=(20, 15))
    plt.tight_layout()
    plt.show()



def logestic_regression():
    global df, X_train, X_test, y_train, y_test,lr_accuracy

    # Train Logistic Regression model
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    # Predict on test data
    y_pred = logreg.predict(X_test)

    # Calculate evaluation metrics
    lr_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Display results in the text box
    text.delete('1.0', END)
    text.insert(END, "Logistic Regression Results:\n")
    text.insert(END, f"Accuracy: {lr_accuracy:.4f}\n\n")
    text.insert(END, "Classification Report:\n")
    text.insert(END, report + "\n")

def KNN():
    global df, X_train, X_test, y_train, y_test,knn_accuracy

    # Create KNN classifier
    knn = KNeighborsClassifier()

    # Train the model using the training sets
    knn.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = knn.predict(X_test)

    # Compute evaluation metrics
    knn_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Display results in the text box
    text.delete('1.0', END)
    text.insert(END, "KNN Results:\n")
    text.insert(END, f"Accuracy: {knn_accuracy:.4f}\n\n")
    text.insert(END, "Classification Report:\n")
    text.insert(END, report + "\n")

def Random_forest():
    global df, X_train, X_test, y_train, y_test,rf_accuracy,rf

    # Create Random Forest classifier
    rf = RandomForestClassifier()

    # Train the model using the training sets
    rf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = rf.predict(X_test)

    # Compute evaluation metrics
    rf_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Display results in the text box
    text.delete('1.0', END)
    text.insert(END, "Random Forest Results:\n")
    text.insert(END, f"Accuracy: {rf_accuracy:.4f}\n\n")
    text.insert(END, "Classification Report:\n")
    text.insert(END, report + "\n")

def plot_bar_graph():
    global lr_accuracy, knn_accuracy, rf_accuracy
    
    # Check if accuracies are computed
    if lr_accuracy is None or knn_accuracy is None or rf_accuracy is None:
        messagebox.showerror("Error", "Run algorithms first to compute accuracies.")
        return

    # Define data for plotting
    algorithms = ['Logistic Regression', 'KNN', 'Random Forest']
    accuracies = [lr_accuracy, knn_accuracy, rf_accuracy]
    colors = ['skyblue', 'lightgreen', 'lightcoral']

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting bars
    bars = ax.bar(algorithms, accuracies, color=colors)

    # Adding labels and title
    ax.set_xlabel('Algorithms', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Accuracy of Machine Learning Algorithms', fontsize=16)

    # Adding values on top of the bars
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{accuracy:.2%}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Adjust layout and show plot
    plt.ylim(0, 1)  # Limit y-axis to 0-100%
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()



def predict():
    # Open file manager to select CSV file(s)
    filenames = filedialog.askopenfilenames(title="Select CSV file(s)", filetypes=[("CSV files", "*.csv")])

    if filenames:
        for filename in filenames:
            # Read the selected CSV file
            input_data = pd.read_csv(filename)

            # Fill missing values with mode for each column
            input_data.fillna(input_data.mode().iloc[0], inplace=True)

            # Preprocess input data (if needed)
            label_encoder = LabelEncoder()
            for column in input_data.columns:
                if input_data[column].dtype == 'object':
                    input_data[column] = label_encoder.fit_transform(input_data[column])

            # Perform prediction using Random Forest model
            y_pred = rf.predict(input_data)

            # Display the prediction result for each row
            text.insert(END, f"Predictions for {filename}:\n")
            for idx, prediction in enumerate(y_pred):
                if prediction == 1:
                    text.insert(END, f"Row {idx + 1}: COVID-19 Detected\n")
                    
                else:
                    text.insert(END, f"Row {idx + 1}: COVID-19 Not Detected\n")
                    
            text.insert(END, "\n")



main = tk.Tk()
main.title("Machine Learning Based Approches For Detecting Covid-19 Using Clinical Test Data") 
main.geometry("1600x1500")

font = ('times', 16, 'bold')
title = tk.Label(main, text='Machine Learning Based Approches For Detecting Covid-19 Using Clinical Test Data.',font=("times"))
title.config(bg='Dark Blue', fg='white')
title.config(font=font)           
title.config(height=3, width=125)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = tk.Text(main, height=20, width=150)
scroll = tk.Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=100)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = tk.Button(main, text="Upload Dataset", command=upload, bg="sky blue", width=15)
uploadButton.place(x=50, y=500)
uploadButton.config(font=font1)

pathlabel = tk.Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=250, y=500)

splitButton = tk.Button(main, text="Split Dataset", command=splitdataset, bg="light green", width=15)
splitButton.place(x=250, y=550)
splitButton.config(font=font1)

covid_graph = tk.Button(main, text="covid_graph", command=covid_graph, bg="lightgrey", width=15)
covid_graph.place(x=50, y=550)
covid_graph.config(font=font1)

breath_problem_graph = tk.Button(main, text="breath_problem_graph", command=breath_problem_graph, bg="pink", width=15)
breath_problem_graph.place(x=650, y=500)
breath_problem_graph.config(font=font1)

symptoms_status = tk.Button(main, text="symptoms_status", command=symptoms_status, bg="yellow", width=15)
symptoms_status.place(x=450, y=550)
symptoms_status.config(font=font1)

logestic_regression = tk.Button(main, text="logestic_regression", command=logestic_regression, bg="lightgreen", width=15)
logestic_regression.place(x=650, y=550)
logestic_regression.config(font=font1)

Run_KNN = tk.Button(main, text="Run_KNN", command=KNN, bg="lightblue", width=15)
Run_KNN.place(x=850, y=550)
Run_KNN.config(font=font1)

Random_forest = tk.Button(main, text="Random_forest", command=Random_forest, bg="orange", width=15)
Random_forest.place(x=1050, y=550)
Random_forest.config(font=font1)

plotButton = tk.Button(main, text="Plot Results", command=plot_bar_graph, bg="lightgrey", width=15)
plotButton.place(x=50, y=600)
plotButton.config(font=font1)

predict_button = tk.Button(main, text="Prediction", command=predict, bg="orange", width=15)
predict_button.place(x=250, y=600)
predict_button.config(font=font1)

main.config(bg='#32d1a7')
main.mainloop()
