# CASCS506-FinalProject

# Comparing the performance of the Neural Network model to the XGBoost and SVM models to determine the best model for identifying cancer markers in genetic material

## 1 Problem/Description of the Project
As computer science students in Boston, we recognize that current data tools are highly effective in the biomedical field, particularly in identifying genomic patterns and mutation frequencies in DNA, allowing for greater scientific advancement. DNA and cancer datasets are closely related, presenting a genomic representation of the potential risk of cancer, and we hope to better understand how to predict the possibility of cancer beforehand by observing how different models perform against the same data. Our primary focus for this project will be to implement a neural network from scratch. Once we have successfully created this model, we plan to compare its performance against SVM and XGBoost models. By comparing the performance of various models, we can then draw a conclusion regarding which model is most appropriate for further cancer research in identifying cancer markers in DNA

## 2 Our Goal
While our initial goal is to develop the neural network independently, we will assess our available time to determine if we can also implement the SVM and XGBoost model from scratch. If time constraints arise, we can utilize libraries like Scikit-learn to implement the SVM and XGBoost models instead. After preparing our models, we will train them using cancer genome data and evaluate their accuracy, exploring which model performs best and analyzing the reasons behind their effectiveness implications for future cancer identification.


## 3 Methods - How we plan to model our data
As mentioned previously, our objective is to create a neural network, then compare the performance of our neural network against the performance of SVM and XGboost models.

### 3.1 Neural Network (NN)
Neural Networks (NN) are a class of models inspired by the human brain's structure and function, designed to recognize patterns and make predictions based on input data. A typical neural network consists of layers of interconnected nodes, or neurons, where each connection has an associated weight. First, there is the input layer, which receives the data, then there are one or more hidden layers that perform calculations, and finally there is the output layer.

As illustrated in Figure 1 below, each neuron in a layer applies a linear transformation to its input and then passes the result through a non-linear activation function (such as ReLU or sigmoid) to introduce non-linearity into the model. This allows neural networks to learn complex patterns and relationships from the data.

![Neural Network](image/figure1.png)

In the training process, connections' weights are adjusted using a method called backpropagation to minimize the differences between predicted and actual outputs. This is typically done using an optimization algorithm like stochastic gradient descent.

### 3.2 Support Vector Machine (SVM)
Support Vector Machine (SVM) is a model that finds the most efficient hyperplane or decision
boundary to classify categories of each class. The SVM sets the boundary of the classification
decision. SVM sets this boundary based on support vectors, which are observations found at the
outermost edges of each class.

The distance between the boundary and the position of the support vectors located at the outermost
edges is referred to as the margin. An SVM that tolerates errors within the margin is called a soft
margin SVM, while an SVM that does not tolerate errors is called a hard margin SVM.

### 3.3 XBGoost 
XGBoost (Extreme Gradient Boosting) is an efficient and powerful machine learning algorithm that enhances traditional gradient boosting methods. It builds an ensemble of decision trees, where each tree is trained to correct the errors of the previous ones, leading to improved predictive accuracy. XGBoost incorporates regularization techniques, such as L1 and L2 regularization, to reduce overfitting and improve model robustness. We will need to look into it more to study the whole algorithm.

### 3.4 Our Method
Once we implement the neural network from scratch, we can then train our model on the data. After training the model, we will run it on the test DNA sequence data. At this stage, we will compare the efficiency of all three models: the neural network, SVM, and XGBoost.

The main goals of our project are to 1) study the structure of the neural network, 2) determine which of the three models is more suitable for classifying DNA images, and 3) analyze the reasons behind the model's performance and improvements. To achieve these objectives, we will thoroughly investigate all three models, train them on our datasets, and evaluate the results. This comprehensive approach will help us draw meaningful conclusions about the effectiveness of each model in the context of our research.

## 4 Methods - How we plan to visualize our data
To begin, we need to identify the type of regression analysis that is most appropriate for our project. If we were working with only two variables—let's say an attribute of the DNA data (x) and a corresponding classification outcome (y)—we could plot the training data on separate graphs to observe the relationship between each attribute and the classification. However, this approach would not account for the interactions among multiple attributes simultaneously, and isolating them could lead to distorted conclusions.

Moreover, as we incorporate more dimensions, particularly beyond three, visualizing the data becomes increasingly challenging, rendering traditional plotting methods less effective. Therefore, we must consider multi-dimensional regression techniques that can handle the complexity of our datasets without losing the relationships between various factors. This will help us gain a more comprehensive understanding of how the different attributes interact and influence the classification outcomes.

Alternatively, we could explore data compression techniques to make our DNA and cancer datasets more manageable and interpretable. While we know that models like CNNs are effective for compressing image data, our datasets are raw and structured, originating directly from tables. Therefore, we need to investigate other compression methods suitable for our application.

Initially, it may be beneficial to remove less relevant or impactful attributes from the datasets, such as those that do not significantly contribute to classification. Additionally, we might consider reducing the dimensionality of the data to align it with the test datasets we will analyze later. To achieve this, we will experiment with various compression model, including to kernel method determine the most effective approach for our project.

## 5 Dataset and test plan
The data set that we plan on using will be from cBioPortal for Cancer Genomics, a resource that provides publicly available datasets containing useful samples. The link to our specific dataset is [here](https://www.cbioportal.org/study/clinicalData?id=aml_ohsu_2022), which is a dataset of myeloid leukemia samples. For each sample, there is important information that we can use to train our model, such as the number of genetic mutations, what the mutation is, the number of samples, and other factors that can dictate the devlopment of cancer. Additionally, among the sampels are matched normal samples, which are samples of tissue from the same subject that are healthy. We can utilize this data as [art of our training in identifying healthy genetic markers from those that could be cancerous. To actually obtain and collect the data from our dataset website, we can download the data as a TSV, then either extract specific columns (using tools liek pandas/numpy/sklearn), such as age at diagnosis, sex, mutation frequency, or leave all the columns, and use this data for our models.

Once we implement the neural network from scratch, we can then train our model on the data. After training the model, we will run it on the test DNA sequence data. At this stage, we will compare the efficiency of all three models: the neural network, SVM, and XGBoost. From our data, we will focus on one cancer type, which is myeloid leukemia, as there are over 900 samples (see the provided link for the specific sample). We will withhold 20% of this data for testing, then use the remaining data to train our models. Since we ae able to access the data, we can control what information and how we train our test our models, in order to ensure best practices.

The main goals of our project are to 1) study the structure of the neural network, 2) determine which of the three models is more suitable for classifying DNA images, and 3) analyze the reasons behind the model's performance and improvements. To achieve these objectives, we will thoroughly investigate all three models, train them on our datasets, and evaluate the results. This comprehensive approach will help us draw meaningful conclusions about the effectiveness of each model in the context of our research.

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
In order to divide work we were planning on investigating different methods. The basis of our project
at this point is Neural Network, so we would likely explore that together. However, we would explore other
possible tools such as SVM or XGBoost.

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
