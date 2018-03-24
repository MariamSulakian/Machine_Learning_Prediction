by Mariam Sulakian

# Using Machine Learning to Predict the 2018 EPL Matches
<br/>Machine learning model, written in Python, to predict the outcome of the 2018 English Premier League (EPL) football matches. Built by training suitable machine learning algorithms on historic results data.

[Write-up](https://mariamsulakian.com/2018/02/01/machine-learning-predicting-the-2018-epl-matches/) available on my [blog](https://mariamsulakian.com/).

## INTRODUCTION
I have built a machine learning model that looks at past EPL game data to predict future games in January 2018. Various attributes were studied including: total goals scored, total goals allowed, discipline (yellow cards, red cards, fouls incurred, total corners), shots per games, shots allowed per game, percentage of games won, defensive statistics (goalie saves, goalie save percentage, ratio of saves), and offensive statistics (scoringPercentage, scoring ratio). Various models were trained using these statistics and each team’s outcome in the past games. KNN produced the highest accuracy with 56%.

## DATA IMPORT
Imported native Python libraries and Scikit-Learn libraries. Update the workspace_id, authorization_token, and endpoint to correspond to the ‘Training.csv’ file in azure, or you can alternatively use the link below to include a file path from your computer. A pathway can be easily produced through azure ‘Generate Data Access Code’.

## DATA TRANSFORMATION AND EXPLORATION
A year column was added to the data, as well as two columns with ‘Winner’ and ‘Loser’ with a none entry in each corresponding to a tie. Data was transformed in excel to produce the various statistics that would be included in the model (code can be found below for each metric). Feature selection was done with recursive feature elimination and feature importance quantification using Extra Trees Classifier to select for top 10 features. However, best results were achieved when all 14 features were included.

## METHODOLOGY OVERVIEW
This model takes in two teams and which year they will be compared in. So for 2018, 2017 data will be used since it is the most current season. The model will then predict the probability that each team will win. Many of the algorithms used require a numerical representation of attributes to conduct statistical analysis. Feature vectors are commonly used in machine learning model since they are n-dimensional vectors composed of numerical inputs. Since these models take in vectors as input, the statistics were transformed into vectors, one for each team, which could then be compared. The simplest way to compare the two vectors is to take the difference between them. The model will then use the resultant vector to predict the probability that each team will win. The model will then be composed of an x component which will be the difference vector and a y component, which will be 1 if team 1 wins, and 0 will be associated with the inverse of the difference. This will allow the model to introduce negative sampling by allowing the model to select against true negatives.

## DATA VISUALIZATION
See [blog post](https://mariamsulakian.com/2018/02/01/machine-learning-predicting-the-2018-epl-matches/)

## FEATURE CREATION
Use a Support Vector Machine (SVM). Plot each data item as a point in a n-dimensional environment, where each feature is a value of a particular coordinate. Find the distance (subtraction) between two vectors.

## MODEL TRAINING
Create training method that takes in a dictionary with with all the teams vectors by year. For each game, the function calculates the difference between between the team vectors for that year. Then, the function assigns a yTrain that is a 1 if the home team wins, and 0 otherwise. The difference vector becomes the input (xTrain) for the model, and a label (yTrain).

## FEATURE SELECTION
Feature selection of the 14 features is done through recursive feature elimination and a ranking of feature importance with extra trees classifier. 10 features of the 14 were determined to be much more influential.

## UPDATED FUNCTIONS TO INCLUDE ONLY TOP 10 FEATURES
See [blog post](https://mariamsulakian.com/2018/02/01/machine-learning-predicting-the-2018-epl-matches/)

## MODEL VALIDATION
Training all 14 features using linear Regression produced the most reliable results. Thus although the function gave the top 10 features to be most influential, all 14 were used to calculate the final results for optimal accuracy

### TESTING MODELS ALL 14 FEATURES
Linear Regression: 62%
SVM Regression: 50.7% (SVC)
SVM Classification: 46.9% (SVR)
Decision Tree (Classifier, Regressor): 49.3%, 52.9%
Logistic Regression: 45.2%
Random Forest Classifier (n = 100): 50.4%
Bayesian Ridge Regression: 49.3%
Lasso Regression: 47.6%
Ridge Regression or Tikhonov regularization (alpha = 0.5): 46.4%
Ada-boost Classifier (n = 100): 49.0%
Gradient Boosting Classifier (n = 100): 50.7%
Gradient Boosting Regressor (n = 100): 47.8%
KNN (n = 60): 56.5%

### TESTING MODELS TOP 10 FEATURES
Linear Regression: 57%
SVM Regression: 58.1% (SVC)
SVM Classification: 56.7% (SVR)
Decision Tree (Classifier, Regressor): 47.6%, 50.4%
Logistic Regression: 49.8%
Random Forest Classifier (n = 100): 44.4%
Bayesian Ridge Regression: 48.0%
Lasso Regression: 41.2%
Ridge Regression or Tikhonov regularization (alpha = 0.5): 41.8%
Ada-boost Classifier (n = 100): 42.8%
Gradient Boosting Classifier (n = 100): 51.6%
Gradient Boosting Regressor (n = 100): 52.2%

## RESULTS
I tested out the above models and selected the one with the highest prediction accuracy (linear regression). I then used this model to calculate the predictions for the 2018 games.

## FINAL PREDICTIONS ON TEST SET
Final predictions are based on the probability that first team (Home Team) wins. The probabilities are as follows:
1) Arsenal Crystal Palace 53.6%
2) Burnley Man United 49.6%
3) Everton West Brom 51.4%
4) Leicester Watford 54.4%
5) Man City Newcastle 77.4%
6) Southampton Tottenham 55.9%
7) Swansea Liverpool 47.3%
8) West Ham Bournemouth 58.96%

## REFERENCES
### Background reading:
* Brucher, Matthieu, et al. “Scikit-Learn: Machine Learning in Python.” Edited by Mikio Braun, Journal of Machine Learning Research, 2011, www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf.
* Demsar, Janez, and Blaz Tupan. From Experimental Machine Learning to Interactive Data Mining. Orange, www.celta.paris-sorbonne.fr/anasem/papers/miscelanea/InteractiveDataMining.pdf.
* Dewey, Conor. “The Hitchhiker’s Guide to Machine Learning in Python.” FreeCodeCamp, FreeCodeCamp, 1 Aug. 2017, medium.freecodecamp.org/the-hitchhikers-guide-to-machine-learning-algorithms-in-python-bfad66adb378/
* Kaufmann, Morgan. “Data Mining: Practical Machine Learning Tools and Techniques.” Edited by Diane Cerra, Research Gate, Nov. 2010, www.researchgate.net/publication/220017784_Data_Mining_Practical_Machine_Learning_Tools_and_Techniques..
* Paruchuri, Vik. “Machine Learning with Python.” Dataquest, Dataquest, 14 Dec. 2017, www.dataquest.io/blog/machine-learning-python/.
* “An Introduction to Machine Learning with Scikit-Learn.” Scikit-Learn, Scikit-Learn Developers, scikit-learn.org/stable/tutorial/basic/tutorial.html.
* Raghavan, Shreyas. “Create a Model to Predict House Prices Using Python.” Towards Data Science, 17 June 2017, towardsdatascience.com/create-a-model-to-predict-house-prices-using-python-d34fe8fad88f.
### Source used to find the common statistics calculated for football teams:
* “Wellington Phoenix.” Soccer Betting Statistics and Results for Wellington Phoenix, Soccer Betting Statistics 2018, 2018, www.soccerbettingstatistics.com/team/wellington-phoenix/2017-2018/a-league/6282/53/1086.
### Feature vectors inspiration:
* Agarwal, Sumeet. Machine Learning: A Very Quick Introduction. 6 Jan. 2013, web.iitd.ac.in/~sumeet/mlintro_doc.pdf.
### Research/background reading on explanations of Word2Vec – representing words with attributes in a vector, then subtracting those vectors to find the difference between them.
* “Why word2vec works.” Galvanize, blog.galvanize.com/add-and-subtract-words-like-vectors-with-word2vec-2/.
* “Vector Representations of Words.” TensorFlow, 2 Nov. 2017, www.tensorflow.org/tutorials/word2vec.
* “Add and Subtract Words like Vectors with word2vec.” Galvanize, blog.galvanize.com/add-and-subtract-words-like-vectors-with-word2vec-2/.
* Critchlow, Will. “A Beginner’s Guide to word2vec AKA What’s the Opposite of Canada?” Distilled, 28 Jan. 2016, www.distilled.net/resources/a-beginners-guide-to-word2vec-aka-whats-the-opposite-of-canada/.
### Feature Selection inspiration:
* “Feature Selection in Python with Scikit-Learn.” Machine Learning Mastery, 21 Sept. 2016, machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/.
Approach inspiration taken from:
* Forsyth, Jared, and Andrew Wilde. “A Machine Learning Approach to March Madness.” Brigham Young University, 2014, axon.cs.byu.edu/~martinez/classes/478/stuff/Sample_Group_Project3.pdf.
