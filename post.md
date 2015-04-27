
You are a home builder looking to figure out places to build your next venture in California. You want to understand similar places & clusters for which the economics work for us. We have a dataset of 1400 places in California with the following variables:


![Variables](http://vozag.com/uploads/blog/azureml/variables_table.jpg)

We want to use a K-Means clustering algorithm to group similar places into different buckets. In the old days, we would have installed our R package, done a bunch of transformations & got the answer.

In the new world, we can use [Microsoft Azure Machine Learning Studio](https://studio.azureml.net/) to do the same in a fraction of the time. 

From Azure ML’s documentation, here is a brief explanation on [K-Means clustering](https://msdn.microsoft.com/en-us/library/azure/dn905944):
In general, clustering uses iterative techniques to group cases in a dataset into clusters that contain similar characteristics. These groupings are useful for exploring data, identifying anomalies in the data, and creating predictions. Clustering models also can help you identify relationships in a dataset that you might not logically derive through casual observation. For this reasons, clustering is often used in the early phases of machine learning task to explore the data and discover unexpected correlations.
The K-means algorithm places data points into the specified number of clusters by minimizing the within-cluster sum of squares. The K-means algorithm begins with an initial set of centroids, which are like central starting points for each cluster, and then uses Lloyd's algorithm to iteratively refine the locations of the centroids. The algorithm stops building and refining clusters when it meets one or more of these conditions:
The centroids stabilize, meaning that cluster assignments for individual points no longer change and therefore the algorithm has converged on a solution.
The algorithm completed running the specified number of iterations. 


##Step 1: Start Screen
To get going on Azure ML, you need a Microsoft Account. Once you have signed, here is the start screen that you see.

![start screen](http://vozag.com/uploads/blog/azureml/start_screen.png)

##Step 2: Upload Data Set
The second step is to create a data set which you can use the upload option
![create dataset](http://vozag.com/uploads/blog/azureml/create_dataset.png)

##Step 3: Create Experiment
Once the process is done, you will see the available data sets. Now you need to get going on the experiment by clicking on the experiment icon (the experiment flask) & then click New -> Experiment -> Blank Experiment.

Click on the saved data set & choose the file and click on the data set. 
or you can also search for the dataset. Just drag and drop into the canvas
![canvas](http://vozag.com/uploads/blog/azureml/canvas.png)

##Step 4: Choose the Machine learning module
We have chosen the K-Means clustering for this case study but you can choose any algorithm that suits your problem. Here is a brief excerpt from [Azure ML documentation on choosing algorithms](https://msdn.microsoft.com/en-us/library/azure/dn905812.aspx). 

Azure Machine Learning Studio provides many different state-of-the art machine learning algorithms to help you build analytical models. First, identify the general type of machine learning task you are performing, as the algorithms grouped in each category are tailored to specific predictive tasks.
After you have chosen an algorithm and configured its parameters, you can then use one of the training modules to run data through the chosen algorithms, or you can use Sweep Parameters to iterate over all possible parameters and determine the optimal configuration for your task and data. Use the modules in this section to configure one of the many machine learning algorithms provided by Studio, train the model, generate scores, and evaluate your model’s performance.

For our task & to initialize the model, follow: 
Machine Learning -> Initialize Model -> Clustering -> K-Means Clustering

![choose k-means clustering](http://vozag.com/uploads/blog/azureml/choose_kmeans_clustering.png)
This module Configures and initializes a K-means clustering model. We will use 5 centroids and metric is cosine and leave the default 100 initializations

##Step 4: Train the clustering Model

[The Train Clustering Model](https://msdn.microsoft.com/en-us/library/azure/dn905873.aspx) module takes an untrained clustering model, such as that produced by K-Means Clustering, and an unlabeled data set. It returns a trained clustering model that can be passed to Assign to Clusters. It also returns labels for the training data.

To set up the training, go to Under Machine learning -> Train -> train clustering models
drag and drop

Connect the Initialize K-means clustering module and dataset to the train clustering model at the appropriate places. We have to choose the columns we want to consider. For this experiment we consider all the columns except GEOID and place name. Check the check box to get the assignments 

![Train clustering model](http://vozag.com/uploads/blog/azureml/train_clustering_model.png)

##Step 5: Convert to CSV Output

We can choose the final output of the algorithm by adding a “Convert to CSV” module. To download to local we need to use a module which will convert the dataset into a comma separated value file. Convert to csv module does that.

##Step 6: Run the experiment & get the data
You can run the experiment at this stage by clicking on the run button at bottom. To download the results of the cluster, Click on the bubble of the convert to CSV box and you will get each place in California & its assignment to the various five clusters. 

![Run Experiment](http://vozag.com/uploads/blog/azureml/run_experiment.png)

When we ran the experiment considering only median home value we got the following example clusters

![Output](http://vozag.com/uploads/blog/azureml/output.jpg)

##This whole analysis took us less than 1 hour from the start to the finish without ever doing a K-Means cluster analysis on Azure ML before. 