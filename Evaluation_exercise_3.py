# Databricks notebook source
# MAGIC %md
# MAGIC ### Install and Import All Required Libraries/Packages

# COMMAND ----------

# MAGIC %pip install hyperopt mlflow 

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow
import mlflow.spark
import time
from hyperopt import fmin, tpe, hp, Trials
from hyperopt import STATUS_OK

# COMMAND ----------

# MAGIC %md
# MAGIC ## BLOOCK 1: DATA PREPARATION

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.1 Load the Dataset

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/CreditScore_MultiClass.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

#display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Descriptive Analysis and Exploratory Data Analysis (EDA): 

# COMMAND ----------

# Select and change data types using .select()
descriptives_df = df.select(
    col("Age").cast("int"),                       # Cast Age to integer
    col("Annual_Income").cast("float"),           # Cast Annual_Income to float
    col("Monthly_Inhand_Salary").cast("float"),   # Cast Monthly_Inhand_Salary to float
    col("Num_Bank_Accounts").cast("int"),       # Cast Num_Bank_Accounts to float
    col("Num_Credit_Card").cast("int"),         # Cast Num_Credit_Card to float
    col("Interest_Rate").cast("float"),           # Cast Interest_Rate to float
    col("Num_of_Loan").cast("int"),             # Cast Num_of_Loan to float
    col("Num_of_Delayed_Payment").cast("int"),  # Cast Num_of_Delayed_Payment to float
    col("Num_Credit_Inquiries").cast("int"),    # Cast Num_Credit_Inquiries to float
    col("Outstanding_Debt").cast("float"),        # Cast Outstanding_Debt to float
    col("Payment_Behaviour").cast("string"),       # Cast Payment_Behaviour to float
    col("Last_Loan_1").cast("string"),             # Cast Last_Loan_1 to float
    col("Credit_Score").cast("string")            # Cast Credit_Score to string
)

# Display the schema to verify changes
descriptives_df.printSchema()
# Display the dataset for Visulaisations

display(descriptives_df)

# COMMAND ----------

# DBTITLE 1,re selection


# COMMAND ----------

# MAGIC %md
# MAGIC ### Note:
# MAGIC The descriptives table above show there are no Missing Values in each of the variable. So no missing values will be handled in the data preparation/features selection stage.

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Feature Preparation  & 
# MAGIC # 4. Target Variable Transformation: 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Target Variable Transformation 
# MAGIC Convert string column to integers(for categorical variables) and float (for continuos variables)
# MAGIC
# MAGIC

# COMMAND ----------

selected_df = df.withColumn(
    "Credit_Score",
    when(col("Credit_Score") == "Poor", 0)
    .when(col("Credit_Score") == "Standard", 1)
    .when(col("Credit_Score") == "Good", 2)
    .otherwise(None)  # Handle unexpected values
)
display(selected_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature selection
# MAGIC All the columns are in string format. So they have to be converted to suitable formats for the modelling

# COMMAND ----------

# Select and change data types using .select()
selected_df = df.select(
    col("Age").cast("int"),                       # Cast Age to integer
    col("Annual_Income").cast("float"),           # Cast Annual_Income to float
    col("Monthly_Inhand_Salary").cast("float"),   # Cast Monthly_Inhand_Salary to float
    col("Num_Bank_Accounts").cast("int"),       # Cast Num_Bank_Accounts to float
    col("Num_Credit_Card").cast("int"),         # Cast Num_Credit_Card to float
    col("Interest_Rate").cast("float"),           # Cast Interest_Rate to float
    col("Num_of_Loan").cast("int"),             # Cast Num_of_Loan to float
    col("Num_of_Delayed_Payment").cast("int"),  # Cast Num_of_Delayed_Payment to float
    col("Num_Credit_Inquiries").cast("int"),    # Cast Num_Credit_Inquiries to float
    col("Outstanding_Debt").cast("float"),        # Cast Outstanding_Debt to float
    col("Payment_Behaviour").cast("string"),       # Cast Payment_Behaviour to float
    col("Last_Loan_1").cast("string"),             # Cast Last_Loan_1 to float
    col("Credit_Score").cast("string")            # Cast Credit_Score to string
)

# Display the schema to verify changes
#selected_df.printSchema()
# Display the dataset for Visulaisations

#display(selected_df)



# COMMAND ----------

# MAGIC %md
# MAGIC ### Confirm there are no missing values

# COMMAND ----------

from pyspark.sql.functions import col, sum

# Check for the count of null or NaN values in each column
null_counts = selected_df.select(
    [(sum(col(column).isNull().cast("int")).alias(column)) for column in selected_df.columns]
)

# Show the result
null_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The entire dataset visualisations and distributions above show that the dataset is clean without missing values. so we do not need to input NAs.

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Final Analytical Dataset: 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split the Data into Training and test dataset

# COMMAND ----------

#Just for testing
splits = selected_df.randomSplit([.7, .3])
train = splits[0]
test = splits[1]
print("Training Rows:", train.count(), " Rows Testing:", test.count())

# COMMAND ----------

# MAGIC %md
# MAGIC Display column names for further processing

# COMMAND ----------

# Get column names of the DataFrame
df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC # Block II: Model Building and Evaluation 
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Basic Multi-Class Classification: 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data preprocessing Using Pipeline

# COMMAND ----------

# Define categorical and numerical columns
categorical_features = ["Payment_Behaviour", "Last_Loan_1"]
numerical_features = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", 
    "Num_of_Delayed_Payment", "Num_Credit_Inquiries", "Outstanding_Debt"
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the steps/stages: combination of feature engineering and model training

# COMMAND ----------

# Step 1: Index categorical features
catIndexers = [
    StringIndexer(inputCol=col, outputCol=f"{col}Idx", handleInvalid="keep") for col in categorical_features
]


# Step 2: One-hot encode indexed categorical features
catEncoders = [
    OneHotEncoder(inputCol=f"{col}Idx", outputCol=f"{col}Vec") for col in categorical_features
]

# Step 3: Index the label column (Credit_Score)
labelIndexer = StringIndexer(inputCol="Credit_Score", outputCol="label")

# Step 4: Assemble numerical features
numVector = VectorAssembler(inputCols=numerical_features, outputCol="numericFeatures")

# Step 5: Scale numerical features
numScaler = StandardScaler(inputCol="numericFeatures", outputCol="scaledNumericFeatures", withStd=True, withMean=True)

# Step 6: Assemble all features (scaled numerical + encoded categorical)
featureVector = VectorAssembler(
    inputCols=["scaledNumericFeatures"] + [f"{col}Vec" for col in categorical_features],
    outputCol="features"
)

# Step 7: Define the Random Forest algorithm
algo = RandomForestClassifier(labelCol="label", featuresCol="features", 
                               numTrees=100, maxDepth=10, seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC Build then the Pipeline for data preprocessing and Model training

# COMMAND ----------

# Step 8: Create the pipeline
pipeline = Pipeline(stages=catIndexers + catEncoders + [labelIndexer, numVector, numScaler, featureVector, algo])

# Train the pipeline model
pipeline_model = pipeline.fit(train)
print("Model trained")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use the pipeline to run prediction

# COMMAND ----------

# Use the pipeline to prepare the data and fit the model algo

prediction = pipeline_model.transform(test)
predicted = prediction.select("Features", "probability", col("prediction").astype("Int"), col("Credit_Score").alias("trueLabel"))
display(predicted)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Now evaluate model based on accuracy, Precision, Recall and F1 Score for the  Model based on the overall Credit score class labels and also based on each of the 3 class labels

# COMMAND ----------

# Use the indexed 'label' column, not the 'Credit_Score' column
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

# Simple accuracy
accuracy = evaluator.evaluate(prediction, {evaluator.metricName: "accuracy"})
print("Accuracy:", accuracy)

# Class metrics
labels = [0, 1, 2]  # Adjusted for the numeric labels from StringIndexer
print("\nIndividual class metrics:")
for label in sorted(labels):
    print(f"Class {label}")

    # Precision
    precision = evaluator.evaluate(prediction, {
        evaluator.metricLabel: label,
        evaluator.metricName: "precisionByLabel"
    })
    print(f"\tPrecision: {precision}")

    # Recall
    recall = evaluator.evaluate(prediction, {
        evaluator.metricLabel: label,
        evaluator.metricName: "recallByLabel"
    })
    print(f"\tRecall: {recall}")

    # F1 score
    f1 = evaluator.evaluate(prediction, {
        evaluator.metricLabel: label,
        evaluator.metricName: "fMeasureByLabel"
    })
    print(f"\tF1 Score: {f1}")

# Weighted (overall) metrics
overallPrecision = evaluator.evaluate(prediction, {evaluator.metricName: "weightedPrecision"})
print("Overall Precision:", overallPrecision)
overallRecall = evaluator.evaluate(prediction, {evaluator.metricName: "weightedRecall"})
print("Overall Recall:", overallRecall)
overallF1 = evaluator.evaluate(prediction, {evaluator.metricName: "weightedFMeasure"})
print("Overall F1 Score:", overallF1)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Interpretation of the results, including 
# MAGIC ###model performance metrics 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Credit_Score classes Mapping (Poor = 0, Standard = 1, Good = 2):
# MAGIC
# MAGIC The model achieved an overall accuracy of 70.13%, meaning it correctly classified approximately 70% of the samples across all credit score categories. For the Poor category (Class 0), the model performed best, with a precision of 71.12% and recall of 78.57%, resulting in a strong F1 score of 74.66%, indicating effective identification of Poor credit scores. For the Standard category (Class 1), performance was balanced with precision at 71.92%, recall at 70.20%, and an F1 score of 71.05%, reflecting moderate accuracy. However, for the Good category (Class 2), performance dropped significantly, with a precision of 61.55%, recall of 44.37%, and an F1 score of 51.56%, showing difficulty in correctly predicting Good credit scores. The overall precision (69.66%), recall (70.13%), and F1 score (69.55%) suggest reasonable model performance, but improvements are needed for better classification of Good credit scores.

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Model Tracking with MLflow: 

# COMMAND ----------

import time

# Define the timing decorator
def timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start time
        result = func(*args, **kwargs)  # Run the function
        end_time = time.time()  # End time
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to run.")
        return result
    return wrapper

# COMMAND ----------

@timing  # Place the decorator here
def train_credit_score_model(training_data, test_data, numTrees, maxDepth):
    # Experiments
    import mlflow
    import mlflow.spark
    import time

    # Spark Machine Learning related packages
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    # Custom features and label columns
    categorical_features = ["Payment_Behaviour", "Last_Loan_1"]
    numerical_features = [
        "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card", 
        "Interest_Rate", "Num_of_Loan", "Num_of_Delayed_Payment", "Num_Credit_Inquiries", "Outstanding_Debt"
    ]

    # parameters
    numTrees = 100
    maxDepth = 10

    # Start an MLflow run
    with mlflow.start_run():
        # Step 1: Index categorical features
        catIndexers = [
            StringIndexer(inputCol=col, outputCol=f"{col}Idx", handleInvalid="keep") for col in categorical_features
        ]

        # Step 2: One-hot encode indexed categorical features
        catEncoders = [
            OneHotEncoder(inputCol=f"{col}Idx", outputCol=f"{col}Vec") for col in categorical_features
        ]

        # Step 3: Index the label column (Credit_Score)
        labelIndexer = StringIndexer(inputCol="Credit_Score", outputCol="label")

        # Step 4: Assemble numerical features
        numVector = VectorAssembler(inputCols=numerical_features, outputCol="numericFeatures")

        # Step 5: Scale numerical features
        numScaler = StandardScaler(inputCol="numericFeatures", outputCol="scaledNumericFeatures", withStd=True, withMean=True)

        # Step 6: Assemble all features (scaled numerical + encoded categorical)
        featureVector = VectorAssembler(
            inputCols=["scaledNumericFeatures"] + [f"{col}Vec" for col in categorical_features],
            outputCol="features"
        )

        # Random Forest algorithm
        algo = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=numTrees, maxDepth=maxDepth)

        # Chain the steps as stages in a pipeline
        pipeline = Pipeline(stages=catIndexers + catEncoders + [labelIndexer, numVector, numScaler, featureVector, algo])

        # Log training parameter values
        print("Training Random Forest model ...")
        mlflow.log_param('numTrees', numTrees)
        mlflow.log_param('maxDepth', maxDepth)
        
        # Start the model
        model = pipeline.fit(training_data)

        # Evaluate the model and log metrics
        prediction = model.transform(test_data)
        metrics = ['accuracy', 'weightedRecall', 'weightedPrecision']
        for metric in metrics:
            evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName=metric)
            metricValue = evaluator.evaluate(prediction)
            print(f"{metric}: {metricValue}")
            # log it
            mlflow.log_metric(metric, metricValue)
        
        # Log the model itself
        unique_model_name = "random_forest_classifier-" + str(time.time())
        mlflow.spark.log_model(model, unique_model_name, mlflow.spark.get_default_conda_env())
        modelpath = f"/model/{unique_model_name}"
        mlflow.spark.save_model(model, modelpath)

        print("Experiment run completed")


# COMMAND ----------

# MAGIC %md
# MAGIC Call credit score Function to Integrate MLflow to log models, track performance metrics, and visualize model 
# MAGIC comparisons. 

# COMMAND ----------

# MAGIC %md
# MAGIC Click "experiment" link in the output to show the model metrics

# COMMAND ----------

train_credit_score_model(train, test, 100, 10) # numTrees=100, maxDepth=10

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Hyperparameter Tuning:

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Create and Objective function
# MAGIC The objective function trains a Random Forest model with the given hyperparameters (numTrees and maxDepth), processes features (categorical and numerical) through indexing, encoding, and scaling, and evaluates the model’s accuracy on the test data. It returns the negative accuracy (since Hyperopt minimizes the loss) to facilitate hyperparameter tuning and helps identify the best combination of parameters for the model.

# COMMAND ----------

def objective(params):
    # Features
    categorical_features = ['Payment_Behaviour', 'Last_Loan_1']
    numerical_features = [
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Num_of_Delayed_Payment',
        'Num_Credit_Inquiries', 'Outstanding_Debt'
    ]

    # Step 1: Index categorical features
    catIndexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}Idx", handleInvalid="keep") for col in categorical_features
    ]
    
    # Step 2: One-hot encode indexed categorical features
    catEncoders = [
        OneHotEncoder(inputCol=f"{col}Idx", outputCol=f"{col}Vec") for col in categorical_features
    ]
    
    # Step 3: Index the label column (Credit_Score)
    labelIndexer = StringIndexer(inputCol="Credit_Score", outputCol="label")
    
    # Step 4: Assemble numerical features
    numVector = VectorAssembler(inputCols=numerical_features, outputCol="numericFeatures")
    
    # Step 5: Scale numerical features
    numScaler = StandardScaler(inputCol="numericFeatures", outputCol="scaledNumericFeatures", withStd=True, withMean=True)
    
    # Step 6: Assemble all features (scaled numerical + encoded categorical)
    featureVector = VectorAssembler(
        inputCols=["scaledNumericFeatures"] + [f"{col}Vec" for col in categorical_features],
        outputCol="features"
    )
    
    # Step 7: Random Forest Algorithm
    rf = RandomForestClassifier(
        labelCol="label", featuresCol="features",
        numTrees=params['numTrees'], maxDepth=params['maxDepth']
    )

    # Step 8: Define the pipeline
    pipeline = Pipeline(stages=catIndexers + catEncoders + [labelIndexer, numVector, numScaler, featureVector, rf])

    # Step 9: Fit the model
    model = pipeline.fit(train)

    # Step 10: Evaluate the model
    prediction = model.transform(test)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(prediction)

    # Hyperopt tries to minimize the objective function, so return the negative accuracy
    return {'loss': -accuracy, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create parameter lists for Random forest, create a trial object and then use Hyperpost objective function to Run Trials to search for best model.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC The trials object in Hyperopt keeps track of the results from each hyperparameter combination tested during optimization. It records the performance based on loss and other relevant information for each trial, allowing for comparison of different hyperparameter settings and helping to identify the best-performing configuration.

# COMMAND ----------

# Define the search space for numTrees and maxDepth
space = {
    'numTrees': hp.choice('numTrees', [50, 100, 150, 200]),  # Number of trees
    'maxDepth': hp.choice('maxDepth', [5, 10, 15, 20])        # Max depth of trees
}

# Use Hyperopt's fmin function to minimize the objective
trials_run = Trials()
argmin = fmin(fn=objective,
              space=space,
              algo=tpe.suggest,
              max_evals=6,
              trials=trials_run)

print("Best hyperparameters:", argmin)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Get details from each trial run

# COMMAND ----------

print("trials:")
for trial in trials_run.trials:
  print("\n", trial)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interpretation

# COMMAND ----------

# MAGIC %md
# MAGIC The trials results suggest that the best configuration was achieved with maxDepth = 3 and numTrees = 1, resulting in the lowest loss value of -0.7954, indicating the optimal parameters in this set for the model.
