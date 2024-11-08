                                      ___Hate-Speech-Detection-In-Social-Media-In-Python___
Python code to detect hate speech and classify twitter texts using NLP techniques and Machine Learning This project is ispired by the work of t-davidson, the original work has been referenced in the following link. 
This project works on improving the existing work and coming up with new findings and fresh analysis of the changes that occur when new features are introduced to the existing project.
Inspiration Source -https://github.com/t-davidson/hate-speech-and-offensive-language
Description of the project:
In this thesis we start by presenting the existing problem which comes with freedom of speech on the Internet and the misuse of social media platforms like twitter.
These problems become an integral part of the motivation. We conduct a comprehensive and thorough research work by referring to the existing works in this field and coming up with a proposed solution for the problem.
We also identify the gaps present in the existing works and find a way to solve those problems. We make use of a publicly available dataset provided by CrowdFlower and apply NLP techniques to achieve our goal.
We describe the flow of our work which starts with analysis of the dataset followed by performing the text pre-processing to achieve a cleaner dataset that can be used in our next step called feature engineering.
We extract some unique and important features and combine them in different sets for the purpose of comparison and analysis of the performance of various machine learning classification algorithms with regard to each feature set.
Finally we conduct an in-depth analysis of the results obtained and explain the reasons for missclassifications in our model.
Results Obtained:
The logistic regression algorithm works consistently well with all feature sets except for F7 as precision, recall and subsequently the f1-score for “hate” label results in zero which is shown in Fig-23. 
Random Forest classifier works pretty well when it comes to F1 and also shows a significant performance in all other feature sets but its performance is hugely impacted when tf-idf scores are not included in the feature set as shown in Fig-24.
The overall performance of the Naïve Bayes classifier is found to be less significant for the purpose of classifying tweets into hate, offensive or neither labels but it performs significantly better with feature set of F7 compared to other feature sets which is shown in Fig-25.
SVM classifier also seems to be consistent throughout all feature sets except for F4 and F7 as shown in Fig26. From the above graphs we analyse that the most important feature was found to be F1 i.e. the tf-idf scores which helps in better classification of hate speech.
The sentiment scores also prove to be an important feature for the differing of hate speech and offensive language .Doc2vec columns are not found to be very significant in classification purpose as it makes very less difference when it’s removed from the feature set.
On comparing all the graphs above Random Forest is clearly the winner
Summary of project:
We started by collecting data for the formation of our hate speech dataset which is a difficult task because what might be hate speech for someone might be normal text for someone else. 
To remove the unwanted content from the dataset, text pre-processing technique is applied where we remove the punctuations, tokenizing, stopwords removal, stemming, and removal of urls and mention names. 
The processed text is passed further for feature extraction where features like n-gram tf-idf weights, sentiment polarity scores, doc2vec vector columns and other readability scores are extracted and concatenated in different sets to fit into different classification models.
These classification models are evaluated on the basis of accuracy and f1-scores in regards to different feature sets.
The results clearly show that differentiating hate speech and offensive language is a challenging task.
It also indicates the beneﬁts of using the proposed features, and provides a valuable resource for detecting the problem of toxic language on twitter. 
Although a detailed analysis of the features as well as errors could lead to more robust feature extraction methods and also help us in solving the existing challenges in this field.
