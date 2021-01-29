# Support-Vector-Machine-term-paper
# Abstract: 
The main aim of this term paper is to describe the Support Vector Machine Algorithm, a supervised model used for classification and regression. The paper focuses on SVM for classification tasks only. There are linear and non-linear classifiers available for Support Vector machine. The paper  describes the hard margin and the Soft Margin linear Classifier. The non-linear Classifier is not covered in this paper. The paper also provides the formulations for the data which is linearly separable and not separable. It also describes the properties, applications, advantages and disadvantages of Support Vector Machine.
# Introduction
Support Vector Machine algorithm was originally invented by Vladimir N.Vapnik and Alexey Ya. Chervonenkis in 1963.  In the year 1992,Bernhard E. Boser, Isabelle M. Guyon and Vladimir N. Vapnik  recommended a way to create nonlinear classifiers by applying kernel trick to maximum-margin hyperplanes. The binary class Support Vector Machine was extended to the multiclass (where the number of classes are greater than two )by Weston and Watkins in 1999, Platt in  2000
Support Vector machine(SVM)  is one of the most predominant supervised learning algorithms that analyze data and recognize patterns. Support Vector Machines is obtained from Statistical Learning Theory. Support Vector Machine is powerful, yet flexible algorithm used for classification and regression but generally used for classification problems. It is also used for the outlier detection[3]. Full Article
# I.	WHAT IS THE OBJECTIVE OF SVM?
The Support Vector Machine objective is to find a hyperplane with N-1 dimensions separating the  classes in the N-dimensional feature space. This is called linear classifier. To separate the classes there are many possible hyperplanes. Our goal is to find a hyperplane with N-1 dimension  that has maximum margin. The margin is defined as the distance between the separating Hyperplane and the training samples that are closest to this hyperplane, 
which are so called support vectors. 
# II.	HYPERPLANE
Hyperplane is a subspace which is one dimension less than the original vector space. Let us understand it by examples. For 2 dimensions , a hyperplane is a one-dimensional subspace, in other words it is  a line. For 3 dimensions, a hyperplane is a two-dimensional subspace( a plane). So, by generalizing the concept to N- dimensions, a hyperplane is a N-1 dimension subspace. For detailed explanation on Hyperplane, see reference[6] Source
The above concept can be generalized for p dimension
      w0+w1x1+w2x2+w3x3+………….+wpxp= 0
The matrix notation for the above equation
w0+wTx   =0                                                                         
(Application of the property wTx =w.x)

# III.	SUPPORT VECTORS
Support Vectors are  data points closest to the decision boundary (hyperplane).The separating planes are defined using these so called support vectors. The Hyperplane position and orientation depends on support vectors. If the data point is not a support vector, removing it  from the model has no effect. But if the  deleted data point is the support vector, it will change the position of the hyperplane. In the figure below the circles represent support vectors of one class and the square represents the support vector from other class
# IV.	SUPPORT VECTOR MACHINE -CLASSIFICATION
Support Vector Machine is simple and intuitive classifier. There are different types of classifiers available depending upon whether the data is linearly separable or not. If the data is linearly separable by a hyperplane, the maxima marginal classifier also called hard margin Support Vector Machine is used.[8]
A.	Hard Margin SVM:
If the data is linearly separable by a hyperplane then there will be infinite number of  separating hyperplanes. Out of all the hyperplanes, the hyperplane with maximal margin( also known as optimal separating Hyperplane)  farthest from the training observations is chosen.
What is maximal margin?
The perpendicular distance from each  observation in the training dataset to the separating hyperplane is calculated. The smallest such distance is the minimal distance from the observations to the separating hyperplane and is called margin. The maximal margin hyperplane is the one for which the margin is largest. The figure below shows the small margin and large margin with circles representing one class and squares representing another class. Support vectors are the data points used for defining the separating hyperplanes.Once the optimal separating hyperplane (decision boundary) is chosen, we can then classify the test record depending on which side of the hyperplane it lies.
B.	Soft Margin SVM
The Soft Margin SVM is the extension of the hard margin SVM for the data which is not separable. SVM, in this case is not looking for maximizing hard margin, instead soft-margin SVM tries to classify most of the data correctly while allowing the misclassification of few points under the appropriate cost penalization 

# MULTICLASS SVM
The Support Vector Machine is extended to multi class classification where the number of classes are more than two. There are different strategies used to address the multiclass classification, they are one versus the rest and pair-wise classification .
# SVM REGRESSION
Support Vector Machine is also useful for regression. It supports linear regression as well as non-linear regression. The principle involved in SVM regression is almost similar to the principle in classification with small differences. In regression , consider two lines at a certain distance one  above and the other  below the  hyperplane. The distance between these lines  is the margin of tolerance(epsilon) or maximum error.

