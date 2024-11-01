# CASCS506-FinalProject

# Finding best model for identifying cancer markers in genetic material

## 1 Problem/Description of the Project
As computer science students in Boston, we recognize that current data tools are highly effective in the biomedical field, particularly in identifying genomic patterns and mutation frequencies in DNA, allowing for greater scientific advancement. DNA and cancer datasets are closely related, presenting a genomic representation of the potential risk of cancer, and we hope to better understand how to predict the possibility of cancer beforehand by observing how different models perform against the same data. Our primary focus for this project will be to determine a model that is able to most accurately and efficiently identify these markers within the genetic material. We plan on comparing the performance of SVM, XGBoost, and regression models. Additionally, we plan to compress the data using SVD and Latent Semantic Analysis, seeing which will allow for more accurate and efficient processing. By comparing the performance of various models, we can then draw a conclusion regarding which model is most appropriate for further cancer research in identifying cancer markers in DNA.

## 2 Our Goal
While our initial goal is to develop the neural network independently, we will assess our available time to determine if we can also implement the SVM and XGBoost model from scratch. If time constraints arise, we can utilize libraries like Scikit-learn to implement the SVM and XGBoost models instead. After preparing our models, we will train them using cancer genome data and evaluate their accuracy, exploring which model performs best and analyzing the reasons behind their effectiveness implications for future cancer identification.


## 3 Methods - How we plan to model our data
As mentioned previously, our objective is to create a neural network, then compare the performance of our neural network against the performance of SVM and XGboost models.

### 3.1 Support Vector Machine (SVM)
Support Vector Machine (SVM) is a model that finds the most efficient hyperplane or decision
boundary to classify categories of each class. The SVM sets the boundary of the classification
decision. SVM sets this boundary based on support vectors, which are observations found at the
outermost edges of each class.

The distance between the boundary and the position of the support vectors located at the outermost
edges is referred to as the margin. An SVM that tolerates errors within the margin is called a soft
The midterm report and 5-minute presentation should include the following.

Preliminary visualizations of data.
Detailed description of data processing done so far.
Detailed description of data modeling methods used so far.
Preliminary results. (e.g. we fit a linear model to the data and we achieve promising results, or we did some clustering and we notice a clear pattern in the data)
We expect to see preliminary code in your project repo at this point.

Your report should be submitted as README.md in your project GitHub repo.

The 5-minute presentation should be a recording uploaded to YouTube. Please add the video link to the beginning of your report.margin SVM, while an SVM that does not tolerate errors is called a hard margin SVM.

### 3.2 XBGoost 
XGBoost (Extreme Gradient Boosting) is an efficient and powerful machine learning algorithm that enhances traditional gradient boosting methods. It builds an ensemble of decision trees, where each tree is trained to correct the errors of the previous ones, leading to improved predictive accuracy. XGBoost incorporates regularization techniques, such as L1 and L2 regularization, to reduce overfitting and improve model robustness. We will need to look into it more to study the whole algorithm.

### 3.3 Regression Model
A regression model predicts a continuous outcome based on input variables. In our approach, we will explore linear and non-linear regression techniques, seeing what fits best with the data that we have collected. We'll explore linear regression as well as more complex forms of regression, such as polynomial regression, to capture non-linear relationships. This allows the model to fit a wider variety of data patterns, which may be more representative of real-world applications.

### 3.4 Singular Vector Decomposition
Singular Value Decomposition (SVD) is a matrix factorization technique commonly used in data reduction, noise filtering, and recommendation systems. In the context of our model, SVD will be used to reduce the dimensionality of our data while preserving key patterns and relationships. This can enhance model performance by reducing overfitting and computational complexity. By keeping only the top singular values, we can focus on the most significant patterns, allowing the model to generalize better to unseen data.

### 3.5 Latent Semantic Analysis 
Latent Semantic Analysis (LSA) is a natural language processing technique that uses SVD to identify patterns in relationships between terms and documents. It transforms the text data into a lower-dimensional space, capturing the underlying structure in the data. In LSA, documents and words are represented in a latent space, where similar words and documents have similar representations. This helps in reducing the noise in text data while focusing on the core concepts. In our application, we will use LSA to extract relationships between the cancer type and the genes on which markers are present. 

### 3.6 Our Method
The main goal of our project is to 1) determine which model data compression is most suitable for classifying DNA cancer markers and 2) analyze the reasons behind the model's performance and improvements. To achieve these objectives, we will thoroughly investigate each of the models paired with the different data compression techniques. We will first compress the data using the different techniques, train the models on our datasets, and evaluate the results. This approach will allow us to draw meaningful conclusions about the effectivenes of each model in the context of our research. 

In terms of the models themselves, we plan to use packages for each of the models (SVM, XGBoost, Regression models) as well as a package for SVD. However, we plan to implement our own LSA in order to fit it better to the data that we are looking at. 

## 4 Methods - How we plan to visualize our data
To begin, we need to identify the type of regression analysis that is most appropriate for our project. If we were working with only two variables—let's say an attribute of the DNA data (x) and a corresponding classification outcome (y)—we could plot the training data on separate graphs to observe the relationship between each attribute and the classification. However, this approach would not account for the interactions among multiple attributes simultaneously, and isolating them could lead to distorted conclusions.

Moreover, as we incorporate more dimensions, particularly beyond three, visualizing the data becomes increasingly challenging, rendering traditional plotting methods less effective. Therefore, we must consider multi-dimensional regression techniques that can handle the complexity of our datasets without losing the relationships between various factors. This will help us gain a more comprehensive understanding of how the different attributes interact and influence the classification outcomes.

Alternatively, we could explore data compression techniques to make our DNA and cancer datasets more manageable and interpretable. While we know that models like CNNs are effective for compressing image data, our datasets are raw and structured, originating directly from tables. Therefore, we need to investigate other compression methods suitable for our application.

Initially, it may be beneficial to remove less relevant or impactful attributes from the datasets, such as those that do not significantly contribute to classification. Additionally, we might consider reducing the dimensionality of the data to align it with the test datasets we will analyze later. To achieve this, we will experiment with various compression model, including to kernel method determine the most effective approach for our project.

## 5 Dataset and test plan
[newData?](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?select=data.csv)
The data set that we plan on using will be from cBioPortal for Cancer Genomics, a resource that provides publicly available datasets containing useful samples. The link to our specific dataset is [here](https://www.cbioportal.org/study/clinicalData?id=aml_ohsu_2022), which is a dataset of myeloid leukemia samples. For each sample, there is important information that we can use to train our model, such as the number of genetic mutations, what the mutation is, the number of samples, and other factors that can dictate the devlopment of cancer. Additionally, among the sampels are matched normal samples, which are samples of tissue from the same subject that are healthy. We can utilize this data as [art of our training in identifying healthy genetic markers from those that could be cancerous. To actually obtain and collect the data from our dataset website, we can download the data as a TSV, then either extract specific columns (using tools liek pandas/numpy/sklearn), such as age at diagnosis, sex, mutation frequency, or leave all the columns, and use this data for our models. We then plan on compressing this data using SVD and LSA, which we can then run multiple trials of with each of the models. 

We can then train our models on the data. After training the model, we will run it on the test DNA sequence data. At this stage, we will compare the efficiency of all three models: the SVM, XGBoost, and regression models. From our data, we will focus on one cancer type, which is myeloid leukemia, as there are over 900 samples (see the provided link for the specific sample). We will withhold 20% of this data for testing, then use the remaining data to train our models. Since we ae able to access the data, we can control what information and how we train our test our models, in order to ensure best practices.

The main goals of our project are to 1) determine which of the three models is more suitable for classifying cancerous DNA markers, and 2) analyze the reasons behind the model's performance and improvements. To achieve these objectives, we will thoroughly investigate all three models, train them on our datasets, and evaluate the results. This comprehensive approach will help us draw meaningful conclusions about the effectiveness of each model in the context of our research.

<!-- We’re looking to use at least 3 datasets. The first dataset that we’re going to use to train our model
comes from Kaggle. There are 1460 rows of training data and 2919 rows of testing data in this dataset.
It is almost a 1:2 ratio. This dataset is incredibly verbose as it has roughly 80 different columns. We
can analyze the differences in results based on the selected attributes and diversify the combinations
of attributes chosen to increase accuracy.

From there, we still need to use two more datasets which we can use as input to analyze the variation
in prices of homes in different cities. It is possible to change these datasets whether we want to see
different cities or need a more accurate train dataset. However, the datasets we want to use are, firstly,
the Paris Housing Price Prediction , and the Chicago House Price dataset. Although one dataset has
more data sets while the other dataset has fewer data sets than the House Price Dataset, I believe we
can test how the number of dataset affects the accuracy. Also, with 17 and 9 columns respectively,
they have fewer attributes than the House Price Datas. Hence, we should carefully select the attributes
corresponding to the model trained on the House Price Dataset for testing.

Using this House Price Dataset-trained model, we plan to test the two datasets and examine the
differences between predicted and actual house prices in Paris and Chicago. Additionally, we aim to
analyze how house prices of similar specifications vary depending on location. -->

## 6 Work of Each
In order to divide work we were planning on investigating different methods. We will collect the data together and work together to decide how we should compress it. However, we plan to each explore one of the possible tools such as SVM, XGBoost, and regressional models.

<!--
## Proposal (Due 10/1)

The project proposal should include the following:

- Description of the project.
  
- Clear goal(s) (e.g. Successfully predict the number of students attending lecture based on the weather report).
  
- What data needs to be collected and how you will collect it (e.g. scraping xyz website or polling students).
  
  
- How do you plan on visualizing the data? (e.g. interactive t-SNE plot, scatter plot of feature x vs. feature y).
  
- What is your test plan? (e.g. withhold 20% of data for testing, train on data collected in October and test on data collected in November, etc.).

Note that at this stage of the project you should be as explicit as possible on what the goals of the project are and how you plan on collecting the data. You can be a little more vague on the modeling and visualization aspects because that will evolve as you learn more methods in class and see what the data looks like.

Keep in mind that the scope of this project should reflect two months worth of work. Do not be overly simple or ambitious. The course staff will provide feedback on your proposal.

**Please form groups of 1-5 students and create a GitHub repo. Submit your GitHub repo URL here: https://forms.gle/ZswPBRmBXrRyQuLc6.**

Your proposal should be submitted as `README.md` in your project GitHub repo.

--!>
