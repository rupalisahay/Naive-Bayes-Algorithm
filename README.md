# Naive-Bayes-Algorithm
Naive Bayes Algorithm

*****Programming language implemented - Python*****
*****Tools - Anaconda (Jupyter notebook) / Pycharm Jetbrains IDE *******


Steps to run the source code:

1. Copy the python file (NaiveBayesAlgorithm.ipynb) and dataset files to your storage of Multinomial Naïve Bayes.
2. Open the file in jupyter notebook, then click on run to compile and run all the funtions.
3. Enter the path of the training and testing dataset. You will be asked to specify the details as follows and write the path specified below:
Enter the path of the training data: 20news-bydate-train
Enter the path of the testing data: 20news-bydate-test


Analysis:
➢ The program is able to read data from 5 random files from the training dataset and creates a naïve Bayes model.

➢ First the model will vectorize the data in the file of the training file and testing file, then it will remove stop words and tokens containing special characters and rest will be stored in the list.

➢ After the vectors are generated, then the running block will train our classifier and vocabulary priors and conditional probabilities are computed.

➢ Then the testing of the dataset is done. Test instance and prediction is obtained using argmax. And then accuracy is displayed.

➢ Libraries used :
  random
  numpy
  math
  ntkl:stopwords
