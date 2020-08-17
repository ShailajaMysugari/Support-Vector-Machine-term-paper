import numpy as np
from scipy.special import comb

def cumulative_comb_with_repetition(n, k):
    """
    Compute the number of possible non-negative, integer solutions to
    x1 + x2 + ... + xk <= n.
    
    We will use this function to compute the dimension 
    of order k polynomial feature vector

    Args:
        n: integer, the number of "balls" or "stars"
        k: integer, the number of "urns" or "bars"
        
    Returns: the total number of combinations, integer.
    """
    # your code below
    for i in range(1,k):
        result = int(comb((n+k),k, exact = False,repetition=False))
     
    return result

    # your code above
    raise NotImplementedError


def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    # Your code below
    z = label *(np.dot(theta, feature_vector) + theta_0)
    loss = max(0.0, 1-z)
    return loss
  
    raise NotImplementedError

def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    # Your code below
    loss_full=0
    n=len(feature_matrix)
    for i in range(len(feature_matrix)):
        loss_full = loss_full + hinge_loss_single(feature_matrix[i], labels[i], theta, theta_0)
    return loss_full/n


    # your code above 
    raise NotImplementedError

def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code below
    if label * (np.dot(current_theta, feature_vector) + current_theta_0)<= 0:
        current_theta += label * feature_vector
        current_theta_0 += label
    return (current_theta, current_theta_0)


    # your code above
    raise NotImplementedError

def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """


    current_theta = np.zeros((feature_matrix.shape[1],))
    current_theta_0 = 0

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # Your code below
            
            current_theta, current_theta_0 = perceptron_single_step_update(
            feature_matrix[i], labels[i], current_theta, current_theta_0)
    



            # Your code above
             
    return (current_theta, int(current_theta_0))
    raise NotImplementedError

def gradient_descent(feature_matrix, label, learning_rate = 0.05, epoch = 10):
    """
    Implement gradient descent algorithm for regression.
    
    Args:
        feature_matrix - A numpy matrix describing the given data, with ones added as the first column. Each row
        represents a single data point.
        
        label - The correct value of response variable, corresponding to feature_matrix.
        
        learning_rate - the learning rate with default value 0.5
        
        epoch - the number of iterations with default value 1000

    Returns: A numpy array for the final value of theta
    """
    n = len(label)
    theta = np.zeros(feature_matrix.shape[1])    # initialize theta to be zero vector
    for i in range(epoch):
        # your code below
        feature_matrix_Transpose = feature_matrix.transpose()
        theta_old=theta
        hypothesis = np.dot(feature_matrix,theta)
        loss = hypothesis - label
        # compute (average) gradient below
 
        gradient = np.dot(feature_matrix_Transpose, loss) / n
        
        # update theta below 

        theta = theta_old - (learning_rate*gradient)
        # compute the value of cost function
        cost = np.sum(loss ** 2) / (2 * n) 
        #print(i, theta, cost)
        # It is not necessary to comput cost here. But it is common to use cost 
        # in the termination condition of the loop
        if(theta_old==theta).all() or cost==0:
            break              
        # your code above 
        # test
        # print(i, theta, cost)
        
    return theta
    raise NotImplementedError




def stochastic_gradient_descent(feature_matrix, label, learning_rate = 0.05, epoch = 1000):
    """
    Implement gradient descent algorithm for regression.
    
    Args:
        feature_matrix - A numpy matrix describing the given data, with ones added as the first column. Each row
        represents a single data point.
        
        label - The correct value of response variable, corresponding to feature_matrix.
        
        learning_rate - the learning rate with default value 0.5
        
        epoch - the number of iterations with default value 1000

    Returns: A numpy array for the final value of theta
    """
    n = len(label)
    theta = np.zeros(feature_matrix.shape[1])    # initialize theta to be zero vector
    for i in range (epoch):
         
        theta_old = theta
        # your code below 
        # generate a random integer between 0 and n
        rand_ind = np.random.randint(0,n)
        x_k = feature_matrix[rand_ind,:].reshape(1,feature_matrix.shape[1])
        y_k = label[rand_ind].reshape(1,1)
        prediction = np.dot(x_k,theta)
     
        # compute gradient at this randomly selected feature vector below
        gradient = (1/n)* np.dot(x_k.T, prediction-y_k)
      
        # update theta below 
        theta = theta_old - learning_rate * gradient
           
        # compute average squared error or empirical risk or value of cost function
        cost = np.sum(np.square(prediction-y_k))/(2*n)
        
        # It is not necessary to comput cost here. But it is common to use cost 
        # in the termination condition of the loop
        

        # your code above 
        # test
        #print(i, theta, cost)
        if cost == 0 or (theta == theta_old).all():
            break
    return theta
    raise NotImplementedError


def kmeans_assignment(X, z):
    """
    Assign each instance to a cluster based on the shortest distance to all centroids. 
    Clusters are integers from 0 to K - 1.

    No loops allowed.
    
    Args:
        X: (n, d) NumPy array, each row is an instance of the data set
        z: (K, d) NumPy array, each row is the coordinate of a centroid.
        
    Returns:
        c: (n, ) NumPy array, the assignment of each instance to its closest centroid.
    """
    
    # Your code below
    distances = np.sqrt(((X[:, np.newaxis] - z)**2).sum(axis=2))
    return np.argmin(distances, axis=1)

    # step 1: compute squared l2 distance matrix D (n, K)
    # the squared l2-distance between each row of X and each row of z


    
    
    # step 2: find the minimum value of each row of D and return its position.



    # your code above 
    raise NotImplementedError
    
    
def kmeans_update(X, c, K):
    from mlxtend.preprocessing import one_hot
    """
    Given the data set and cluster assignment, find the updated coordinates of all centroids.
    
    No loops allowed

    Args:
        X: (n, d) NumPy array, each row is an instance of the data set
        c: (n, ) NumPy array, the assignment of each instance to its closest centroid.
        K: scalar, the number of clusters
    Returns:
       z: (K, d) NumPy array, each row is the updated coordinates of a centroid.
    """
    
    # Your code below. (hint: use one-hot encoding)
    #n=one_hot(c,num_labels=K)
    n = np.eye(K)[c]
    new_centroids=np.dot(X.T, n)/np.sum(n,axis=0)
    return new_centroids.T


    # your code above
    raise NotImplementedError




def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        np.random.seed(1)
        indices = list(range(n_samples))
        np.random.shuffle(indices)
        return indices
