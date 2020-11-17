from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import random as rand
import math

# ----------------------------- OBJECTS ---------------------------------- #
#Classes to store data and clusters
class Data:
    def __init__(self,points,attributes):
        self.points = points
        self.attributes = attributes

class Cluster:
    def __init__(self,points,attributes,means):
        self.points = points
        self.attributes = attributes
        self.means = means

# ----------------------------- STORING DATA ------------------------------ #
# Opens and parses data and returns an array of points and properties
def getData(chosen):
    axes_choices = ["Sepal Length (cm)", "Sepal Width (cm)", "Petal Length (cm)", "Petal Width (cm)"]
    #Get axes needed for plotting later
    axes=[axes_choices[int(i)-1] for i in chosen]

    points,attributes = [],[]
    # Open the dataset and store it as a list
    with open('iris_data.txt') as f:
        dataset = list(f)
    rand.shuffle(dataset)
    # Go through each item in the list and obtain the required features of each point and its attribute
    for item in dataset:
        temp_point = []
        point = item.rstrip("\n").split(",")
        if len(point) <= 1:
            continue
        for i in range(len(chosen)):
            temp_point.append(float(point[int(chosen[i])-1]))
        # Push the chosen feature values and attribute of the point
        points.append(temp_point)
        attributes.append(point[4])
    return points, attributes, axes

# Ask user to enter select properties to plot - and returns a list of the properties they chose
def askProp():
    # Loops until tbe user enters valid input
    while True:
        print('Please enter the number(s) of the category(ies) you would like to plot. The options are the following:')
        print('1) Sepal Length')
        print('2) Sepal Width')
        print('3) Petal Length')
        print('4) Petal Width')
        chosen = input("Enter the number(s) separated by the ' ' or SPACE character eg.(3 1 4): ").split(' ')
        # Setting the global no. of choices variable
        global no_choices
        no_choices = len(chosen)
        # Check if arguments are valid
        if checkValid(chosen) == False:
            continue
        break
    return chosen

#Extra Validation
def checkValid(chosen):
    # Check if extra space was entered.
    if any(i is '' for i in chosen):
        print('Please input choices without extra space.\n')
        return False
    # Check if given option is not 1,2,3,4
    for i in chosen:
        if(i.isnumeric() == False):
            print('Please enter 1,2,3 or 4 as options.\n')
            return False
        i = int(i)
        if not(i in [1,2,3,4]):
            print("Please enter 1,2,3 or 4 as options.\n")
            return False
    # Validation so that the right amount of properties are used.
    if len(chosen) < 1 or len(chosen) > 3:
        print('Please enter 1-3 categories.\n')
        return False

# ----------------------------- PLOTTING ---------------------------------- #
#Return X, Y, Z values and attributes for an object
def getCoord(object):
    x,y,z=[],[],[]
    for point in object.points:
        if(no_choices >= 1):
          # Add the first feature value of point to X array
          x.append(point[0])
        if(no_choices >= 2):
          # Add the second feature value of point to Y array
          y.append(point[1])
        if(no_choices == 3):
          # Add the third feature value of point to Z array
          z.append(point[2])
    return x,y,z,object.attributes

#Return X, Y, Z values and attributes for a cluster
def getCoordCluster(clusters):
    x,y,z,attr = [],[],[],[]
    for cluster in clusters:
       # Get the x, y, z and attr for the cluster
       temp_x,temp_y,temp_z,temp_attr = getCoord(cluster)
       # Merge values
       x = x + temp_x
       y = y + temp_y
       z = z + temp_z
       attr = attr + temp_attr
    return x,y,z,attr


#Plot points given x,y,z values and attribute
def plotPoints(x,y,z,attr,title):
    global global_axes
    #Get colours for data points from attribute array
    color = [('red' if i == 'Iris-setosa' else 'blue' if i == "Iris-versicolor" else 'green') for i in attr]
    #Custom Legend
    options = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    custom_lines = [Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='green', lw=4)]
    # Plotting a 1D scatter plot using just x coordinates
    if(no_choices == 1):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x, [0]*len(x), color=color)
        ax.legend(custom_lines, options)
        ax.set_xlabel(global_axes[0])
        # Hide the values on the y-axis.
        ax.get_yaxis().set_visible(False)
    # Plotting a 2D scatter plot using x and y co-ordinates
    elif(no_choices == 2):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x, y, color=color)
        ax.legend(custom_lines, options)
        ax.set_xlabel(global_axes[0])
        ax.set_ylabel(global_axes[1])
    # Plotting a 3D scatter plot using x, y and z co-ordinates
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, color=color)
        ax.legend(custom_lines, options)
        ax.set_xlabel(global_axes[0])
        ax.set_ylabel(global_axes[1])
        ax.set_zlabel(global_axes[2])
    # Adding title and displaying graph
    plt.title(title)
    plt.show()

# ------------------------------ K MEANS CLUSTERINNG ----------------------- #
def getMinAndMax():
    #Create a temp array to store all values of a feature
    temp = [[] for i in range(no_choices)]
    #Access each point
    for point in data.points:
        #Access each feature
        for i,value in enumerate(point):
            #Add the points feature value to the appropriate array
            temp[i].append(value)
    #Returns an array of minimum/maximum values of each feature
    temp_min = [min(temp[i]) for i in range(no_choices)]
    temp_max = [max(temp[i]) for i in range(no_choices)]
    return temp_min, temp_max

#Creates a data point of a random feature values
def getNewMean():
    temp_min, temp_max = getMinAndMax()
    means = [None] * no_choices
    # Find a random value for each chosen feature
    for i in range(no_choices):
        means[i] = round(rand.uniform(temp_min[i]+1, temp_max[i]-1),1)
    return means

def updateMeans(cluster,point):
    # Updates the mean point/centroid for a given cluster
    n = float(len(cluster.points))
    # Access each feature in the old centroid and updating it
    for i,mean in enumerate(cluster.means):
        cluster.means[i] = (mean*(n-1)+point[i])/float(n)
    return

#Eucledian distance where A and B are arrays of x,y,z co-ordinates (feature values)
def euclidDistance(A, B):
    if (no_choices == 1):
        return math.sqrt(math.pow(A[0] - B[0], 2))
    if (no_choices == 2):
        return math.sqrt(math.pow(A[0] - B[0], 2) + math.pow(A[1] - B[1], 2))
    if (no_choices == 3):
        return math.sqrt(math.pow(A[0] - B[0], 2) + math.pow(A[1] - B[1], 2) + math.pow(A[2] - B[2], 2))

#Finds the nearest cluster of a given point and returns its index.
def getAssignedCluster(point,clusters):
    min_value,position = 0, 0
    #Iterate through each cluster
    for i,cluster in enumerate(clusters):
        #Find the distance between the point and current cluster
        euclid_distance = euclidDistance(point,cluster.means)
        #Set an initial minimum value
        if i==0:
            min_value = euclid_distance
        #Find a new minimum
        if euclid_distance < min_value:
            min_value = euclid_distance
            position = i
    return position

#Find the most common attribute in a cluster and return an array of it
def getCommonAttr(attributes):
    words = Counter(attributes)
    words.most_common(1)
    return [words.most_common(1)[0][0]] * len(attributes)

#Returns k Clusters created by K-Means Clustering
def createKMClusters():
    #Create k new clusters of random means/centroids
    clusters = [Cluster([], [], getNewMean()) for i in range(k)]
    #Access each item in data and assign it to appropriate cluster.
    for j,item in enumerate(data.points):
        #Get the index of the nearest cluster.
        i = getAssignedCluster(item,clusters)
        #Add the point to the array of points in given cluster.
        clusters[i].points.append(item)
        #Add attribute
        clusters[i].attributes.append(data.attributes[j])
        #Update cluster mean
        updateMeans(clusters[i],item)
    #Update attributes of cluster to most common
    for i,cluster in enumerate(clusters):
        if(len(cluster.attributes) != 0):
            clusters[i].attributes = getCommonAttr(cluster.attributes)
        else:
            print('Error - cluster has no assigned points.\n')
    return clusters

# ------------------------------ K NN CLUSTERINNG ------------------------ #
#Roughly splits the data according to a percentage - returns training and evaluation data object.
def splitData(data,percentage):
    training,evaluation = Data([],[]),Data([],[])
    #Calculate how many elements are going into training set
    split = int(round(percentage * len(data.points)))
    for i,point in enumerate(data.points):
        if i < split:
            training.points.append(point)
            training.attributes.append(data.attributes[i])
        else:
            evaluation.points.append(point)
            evaluation.attributes.append(data.attributes[i])
    return training, evaluation


def findNeighbours(test_point,training,k):
    distances = []
    #Making a list of distances between point and training points.
    for i,point in enumerate(training.points):
        euclid_distance=euclidDistance(test_point,point)
        #Add distance and Index of point in training set
        distances.append([euclid_distance,i])
    #Sort the array by the elements Euclidian Distance in ascending order
    distances.sort(key=lambda x: x[0])
    points,attributes = [],[]
    #Get points and attributes of k closest training points
    for i in range(k):
        points.append(training.points[distances[i][1]])
        attributes.append(training.attributes[distances[i][1]])
    return points, attributes

def testAccuracy(predictions, evaluation):
    matched = 0
    #Check if the data was predicted correctly
    for i,attr in enumerate(evaluation.attributes):
        if predictions[i] == attr:
            matched += 1
    return round((matched/float(len(evaluation.attributes))) * 100.0 , 2)

def KNNClustering(k,percentage):
    #Split the data into a training and evaluation set
    training,evaluation = splitData(data,percentage)
    predictions = []
    #For each point in the evaluation set
    for i,point in enumerate(evaluation.points):
        #Find the closest points and their attributes from the training set
        neighbours, attributes = findNeighbours(point,training,k)
        #Find the most common attribute and add it to predictions
        attr = getCommonAttr(attributes)[0]
        predictions.append(attr)
    #Test the accuracy of the predictions and return
    accuracy = testAccuracy(predictions,evaluation)
    print('\tFor k = ' + repr(k) + ' and Training Percentage = ' + repr(percentage) + ' : Accuracy = ' + repr(accuracy) + '%')
    return accuracy

#Does KNN Clustering for multiple values of k and tries to find best k and training percentage
def doKNNClustering():
    max_acc, max_k, max_train = 0.0, 0, 0.0
    print('\n\t - - - - - - - - KNN Clustering - - - - - - - - ')
    # Try a bunch of K's
    for i in range(3, 10):
        temp_acc, temp_k, temp_train = 0.0, 0, 0.0
        # Try a bunch of percentages
        for j in range(60, 81, 5):
            accuracy = KNNClustering(i, float(j) / 100)
            if (accuracy > temp_acc):
                temp_acc, temp_k, temp_train = accuracy, i, float(j) / 100
        print('\tMaximum for current K =', i, 'Max Accuracy = ', temp_acc, '%')
        if (temp_acc > max_acc):
            max_acc, max_k, max_train = temp_acc, temp_k, temp_train
    # Print Best Results
    print('\nThe best accuracy (', max_acc, '%) of results by KNN Clustering came from the following values:')
    print('K = ', max_k, 'Training Percentage = ', max_train)


# ----------------------------- RUNNER ----------------------------- #
def start():
    #Create a normal graph with given points.
    x,y,z,attr = getCoord(data)
    plotPoints(x, y, z,attr,'Actual Data')

    #Do K-Means Clustering
    clusters = createKMClusters()
    x, y, z, attr = getCoordCluster(clusters)
    plotPoints(x, y, z, attr,'K-Means Clustering')

    # Do K-NN Clustering
    doKNNClustering()

# ---------------------- GLOBAL VARIABLES -------------------------- #
no_choices = 0 #no. of features the person chose - defined in askProp()
k = 3 #K value - remains constant
#Creating a global variable data
global_points, global_attributes, global_axes = getData(askProp())
data = Data(global_points, global_attributes)

#Start program
start()