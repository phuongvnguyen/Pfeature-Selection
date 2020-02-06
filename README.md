# Pfeature-Selection
-----------------------------------------------------------------------------------------
                Pfeature_select Program was developed by Phuong Van Nguyen 
                      Email: phuong.nguyen@economics.uni-kiel.de
                            Copyright @ Phuong Van Nguyen
---------------------------------Introduction--------------------------------------------


The procedure for feature selection plays an important role in training a Machine Learning
model. Three roles of this procedure are given as follows. First, it removes
irrelevant feature which might worsen the performance of a Machine Learning model.
Second, it makes the data dimensionality reduction. Third, it provides initial ideas about
the relationship between the target feature and explanatory variables. 

To this end, I create three functions that can apply to any type of data, including
categorical and numerical data. These functions are extremely useful for users. 
This is because one just declare several parameters, he/she can produce the four following
vital results

1. How many features are needed to explain the target feature?
2. Visualize the contribution of each feature to the target feature.
3. Visualize top contributors to the target feature.
4. Visualize the cumulative contribution of all explanatory variables to the target feature.
5. Save the results of contribution of all explanatory variables to the target feature.

A wide range of state-of-the-art Machine Learnings are used, such as 
Principal Component Analysis (PCA), Ensemble methods (both bagging and boosting ones)
