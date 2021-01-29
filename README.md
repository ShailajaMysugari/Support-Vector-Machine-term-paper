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
B.	 Construction of the Hard Margin SVM:
For our convenience, let us consider the training data with two classes positive and negative separated by a decision boundary. To get the margin maximization let us look at the positive hyperplanes and negative hyperplanes which can be expressed as follows
Equation 1.0 defines parallel hyperplanes of positive class
b+wT xneg  ≤  ─ 1  -----------------------[1.1]
Equation 1.1 defines parallel hyperplanes of negative class
The xpos represents the support vectors for positive class(points on the hyperplane of positive class) and xneg represents the support vectors for negative class(points on the hyperplane of negative class).  wT represents the weight vector, b represents the bias or offset

The above equations are based on the concept of signed distance of dot product. The equation [1.0] represents the positive distance and the equation [1.1] represents the signed negative distance from the decision boundary (Hyperplane).
In the above equations we are assuming the samples are classified correctly
Now let us consider label y(i) such that
y(i) is +1 for the positive class
y(i) is -1 for the negative class
The equations [1.0] and [1.1] are multiplied with the label y(i).
Then the final  equation 
y(i)( b+wTx(i)) ≥ 1………………………[1.2]
y(i)( b+wTx(i)) is called as agreement.
From the above equation we can say that the prediction is correct when y(i) and b+wTx are in same direction.
(For information: The x is representation of support vectors in matrix notation)
Our main aim is to find the maximal margin between two margin boundaries. For that we subtract the equation [1.0] from equation [1.1]  to obtain the width (distance between the two parallel hyperplanes)we get
( xpos - xneg) = 2/(∥w∥ )
The left-hand side of the above equation is the distance between the positive and negative hyperplanes which is  called margin, the objective to be maximized. Now, the objective is to maximize the 2/(||w||). In order to maximize it we need to minimize the 1/2||w||2  which can be solved by using quadratic programming.
For detailed information about derivation, see reference[7].
The Hard Margin SVM classification is useful only if there is a linear separation of data. But in reality, most of the classification data is not completely separable, so the SVM is extended to Soft margin SVM by introducing a function called Hinge loss function. 
	Soft Margin SVM
The Soft Margin SVM is the extension of the hard margin SVM for the data which is not separable. SVM, in this case is not looking for maximizing hard margin, instead soft-margin SVM tries to classify most of the data correctly while allowing the misclassification of few points under the appropriate cost penalization 
The  data is not separable in this case, a slack variables ξ(i)  is used in SVM objective function to allow error in the classification.(to allow misclassification)
	Construction of Soft Margin SVM
The equations for the soft margin SVM are obtained by applying slack variable ξ(i)  to the linear constraints. The equations would be same as Hard Margin Support Vector Machine. But we introduce a Hinge loss , allowing contraints to be violated and integrate actual loss value. Let us consider
z= y(i)( b+wTx(i))………………………..[1.2]
The Hinge loss Function is given by
h(z)  =   0 if   z ≥1
              1 – z if z<1…………………..….[1.3]
h(z)   is called as Hinge loss.
z greater than or equal to 1 represents the record has been classified correctly and so the loss term is zero. When the z is between 0 and 1 the records are classified correctly with hinge loss less than 1. When the z is less  than 1 there will be misclassification, so in this case loss function is given by equation [1.3]
Our goal is to find the maximal hyperplane by allowing some amount of misclassification. So, let us take an example of a dataset with size n and m features. There are two classes in the dataset positive and negative class. The SVM classifier will try to minimize the following function
ƛ/2||w^2 ||+1/n[∑_(i=1)^n▒〖(1 - y(i)( b+w^T x(i)))〗   ]
The above function can be divided in to two parts
ƛ/2||w^2 ||  is a regularization term and ƛ is the regularization parameter.
1/n[∑_(i=1)^n▒〖(1 -( y(i)( b+w^T x(i))〗   ] is the loss term 
For convention lambda is removed from regularization term and the loss term is multiplied by variable C ( a misclassification 
parameter) and  term (1 - y(i)( b+w^T x(i)) is replaced by h(z)  (hinge loss function)

Then our new optimization problem is to minimize the function
1/2 |(|w|)|^2+C/n  [∑_(i=1)^n▒〖h(〗  y(i)(b+w^T x(i)))]
Subject to constraint 
y(i)( b+wTx(i)) ≥ 1-ξ(i)              ξ(i)≥0   
i=1,2,3……..n
      
The relation between lambda and C is ƛ = 1/C
The variable C is to control the penalty for the misclassification error. Large values of C correspond to the large error penalties whereas for small values of C corresponds to small error penalties. In other words, small value of C means we are less strict about misclassification errors.  The value of parameter C  is used to control the width of the margin and the bias-variance tradeoff. Increasing the  C value decreases the training error. For detailed information ,see reference [7]
Using Gardient Descent and Stochastic Gradient Descent to find optimised solution for the objective function of Support Vector Machine.
The cost Function is given by 
J(w,b)= 1/2 〖||w||〗^2 +   C/n  [∑_(i=1)^n▒〖h(〗  y(i)(b+w^T x(i)))]
The weight is updated using the Gradient     Descent algorithm as follows.
w=w-η/n  dJ/dw
     η is the learning rate.
     For Stochastic Gradient descent a point is                                                                   
     selected randomly from the points i=1,2…..n
     The weight is updated using Stochastic Gradient     
     descent algorithm as follows.
     
w=w-η/n  d[1/2 〖||w||〗^2+ C/n  h( y(i)(b+w^T x(i)))]/dw
    Thus the optimised solution is obtained using the     
    Gradient descent, Stochastic Gradient descent. 
       Using Lagrange Multipliers
Introducing the Lagrange Multipliers αi for the constraints to obtain optimal solution ,          i=1,2,3………………………………N
Minimize Lp(w,b,α) =     1/2 〖||w||〗^2+C/n  [∑_(i=1)^n▒〖h(〗  y(i)(b+w^T x(i)))]
〖||w||〗^2/2+∑_(i=1)^n▒〖αi[1-( y(i)( b+w^T x(i))]〗 
Subject to αi>0 i=1,2,3,………..n
This is called the primal problem also referred as Primal Lagrangian. For detailed information see information[8]
Formulating the dual objective
Take the partial derivatives of Lp with respect to w set to zero
     ∂Lp/∂w=0=>w=∑_(j=1)^n▒〖 αjy(j)x(j)〗
Take the partial derivatives of Lp with respect to b set  to zero
∂Lp/∂b=0=>∑_(j=1)^n▒  αjy(j)=0
Substituting them in the primal Lagrangian Lp gives the dual Lagrangian
LD(w,b,α) = 
∑_(i=1)^n▒  αi -  1/( 2) ∑_(i,j=1)^n▒〖 αiαjyiyj〗 xj^T.xi
subject to constraint  
∑_(j=1)^n▒  αjy(j)=0,           0≤ αi≤ C 
 
This is called dual objective function and this function must be maximized for positive values of alpha to find the optimum Hyperplane[8]. .Full Article
Note: The above two are linear classifiers, there are nonlinear classifiers available in Support Vector Machine using kernel Function. They are popular classifiers ,useful when the data is not linearly separable in input space.

