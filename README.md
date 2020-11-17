# ICS2207-Assignment - Experimenting with k-Means and KNN

In this assignment, students were asked to download the <a href="http://archive.ics.uci.edu/ml/datasets/Iris">Iris Dataset</a> and use it to implement the following three artifacts:
<ol>
<li>Write a program which loads the dataset, asks you to select either 1, 2 or 3 features and then plots them as a graph. Each instance in the graph should be labelled either red, green or blue depending on which class the instance belongs to for visually determine whether there are any 'obvious' clusters.</li>
<li>Implement the k-Means clustering algorithm (for k=3) <b>WITHOUT</b> using a library, to cluster the data into its three possible classes. Similar to (1) allow the user to select 1, 2, or 3 features and plot all the instances of a cluster as a particular colour.</li>
<li>Implement the k-NN algorithm by yourself, <b>WITHOUT</b> using a library. Train using a random % of the data and the remaining % for evaluation. Identify the optimal splits and which k to use.</li>
</ol>

## Implementation
All artifacts of this assignment were completed and a report has been included in the repo describing my implementation process of this assignment in detail. The program plots chosen features of a data set and then uses an implementation of the K- Means Clustering algorithm and the KNN Clustering Algorithm. At the start of the program the user chooses 1-3 features from the Iris data set which they want to plot as shown in Figure 1 below. 

<p align="center">
  <img src="https://github.com/valerija-h/ICS2207-Assignment/blob/main/images/Example.png" width="80%"/>
</p>
<p align="center"><b>Figure 1</b> - Example of asking the user to choose which features to plot.</p>

A scatter plot graph called 'Actual Data' of appropriate dimensions will display the data points of these chosen features. The graph in Figure 2 on the left is an example of the chosen features being plotted. Next, the K-Means Clustering Algorithm will attempt to classify each point in the dataset to one of 3 clusters (each of which will represent a particular plant class). The clusters will be plotted in a graph called 'K-Means Clustering' (Figure 2 on the right) and displayed to the user to compare the results to the 'Actual Data' graph.

<p align="center">
  <img src="https://github.com/valerija-h/ICS2207-Assignment/blob/main/images/Graphs.png" width="60%"/>
</p>
<p align="center"><b>Figure 2</b> - The user's chosen features plotted and the plotted results of the K-Means Clustering algorithm for the data chosen in Figure 1</p>

In the end, the program will under-go KNN Clustering for different k-values and varying training split percentages and display their accuracy. The k-value and training split that gives the highest accuracy is displayed at the end of the program. For a more detailed description on how these algorithms were implemented please refer to the report.

<p align="center">
  <img src="https://github.com/valerija-h/ICS2207-Assignment/blob/main/images/Output.png" width="40%"/>
</p>
<p align="center"><b>Figure 3</b> - KNN Clustering Algorithm for different k-values and training splits for the same data chosen in Figure 1.</p>
