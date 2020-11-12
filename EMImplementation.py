import pandas as pd
import numpy as np
import statistics
import math
from tqdm import tqdm
from scipy.stats import poisson
from mpmath import *

# Constants to be used in the program
number_of_data = 7
mu_of_poisson = 15
number_of_data_generated = 8
tolerance = 0.5
e_limit = -15
cost = 4
price = 10
error = 0.25

# Initialization phase. There are four arrays to store the results
# Array of clusters : Stores the clusters as the result of simulation
# Array of average to converge : Stores the number of iterations needed to converge
# Array of BIC : Stores the BIC values of the clusters
# Array of difference : Stores the smallest number of difference between cluster values
array_of_clusters = []
array_of_average_to_converge = []
array_of_BIC = []
array_of_difference = []

for _ in tqdm(range(1000)):
    # Generating random number of lambdas
    lambda_array_of_daily_demands = np.random.poisson(mu_of_poisson, number_of_data)
    # Change to array of lambda
    
    # Built-in function of pdf count since there is no PDF function in Poisson
    # Input : k (value to be analyzed), mu (the mean)
    # Output : Probability Density Function from 0 to 1
    def pdf_count(k, mu):
        return (math.exp(-mu)*pow(mu, k))/(math.factorial(k))
    
    # Built-in function to count Conditional Expectation to infinity
    # Input : n (value to be analyzed), mu (the mean)
    # Output : Conditional Expectation of n, from 1 to infinity if the mean is mu
    def condExp(n, mu):
        return nsum(lambda x: (x * pdf_count(x, mu))/(1 - poisson.cdf(int(n-1), mu)), [max(1, int(ceil(n))), math.inf])
    
    # Generating hourly demand based on the array of daily demands
    demands = []
    for i in range(0, len(lambda_array_of_daily_demands)):
        demands.append(list(np.random.poisson(lambda_array_of_daily_demands[i], number_of_data_generated)))
    
    # Reshaping the demands to be processed easier
    inverted_demands = list(map(list, zip(*demands)))

    # Initiate an array of daily stock
    stock = []
    for i in range(0, len(lambda_array_of_daily_demands)):
        stock.append(list(np.random.uniform(lambda_array_of_daily_demands[i] + 25, lambda_array_of_daily_demands[i] - 25, number_of_data_generated).astype(int)))

    # Change shape of demands to be processed easier (list -> dataframe)
    new_demands = []
    new_stock = []
    for i in range(number_of_data):
        for j in range(number_of_data_generated):
            new_demands.append(demands[i][j])
            new_stock.append(stock[i][j])

    # Change shape of array of demands
    df = pd.DataFrame(new_demands)
    df_onhand = pd.DataFrame(new_stock) 

    # Change shape of array of demands
    df.columns = ['demand']
    df_onhand.columns = ['stock']

    df = pd.concat([df, df_onhand], axis=1, ignore_index = True)
    df.columns = ['demand', 'stock']

    # Processing the number of sales, comparing bigger value between demand and stock
    df['sales'] = df[['demand', 'stock']].min(axis=1)
       
    # Processing important parameters in newsvendor: Cost, Turnover, Profit
    df['cost'] = cost * df['sales']
    df['turnover'] = price * df['sales']
    df['profit'] = df['turnover'] - df['cost']
    
    # Determining the stockout = Stock = Sales
    df['stockout'] = (df['stock'] == df['sales'])

    # Separating the data between censored demands and uncensored demands
    cens = df.loc[df['stockout'] == 1]
    cens_checkstock = cens['sales'].values.tolist()
    noncens = df.loc[df['stockout'] == 0]
    noncens_checkstock = noncens['sales'].values.tolist()

    # Setting a default value for a lambda candidate
    lambdacand = statistics.mean(cens_checkstock)

    # Run the experiment from 1 to 7 clusters
    
    # Variable to contain all the cluster values
    # One to seven clusters = Storing seven cluster values in a single array
    # Array of convergency = Storing convergence that happens in a cluster
    one_to_seven_clusters = []
    array_of_convergency = []
    
    for number_of_clusters in tqdm(range(1, 8)):
        # Variable to contain the cluster for each number of clusters e.g. 2 [2, 3]
        mu = []
        
        # Variable to count the number of iteration needed for convergence
        number_of_convergence = 0
        
        for i in range(0, number_of_clusters):

            # Set a random set of mu
            # Input : []
            # Output : [0, n], where n is the total number of clusters
            # Example : If the number of clusters is 3, the output table is [0, 15], [0, 14], [0, 19] (3 random numbers on Uniform Distribution)
            array_of_lambdacand = [0]
            new_uniform = np.random.uniform(lambdacand + 25, lambdacand - 25, 1)
            array_of_lambdacand.append(new_uniform[0])
            
            # Check the items that is stock out
            cens = df.loc[df['stockout'] == 1]

            # Continue to iterate while the difference between the candidates is more than the error
            while(abs((array_of_lambdacand[-1] - array_of_lambdacand[-2])) > error):
                # Implement the conditional expectation on the censored demand and the average demand
                cens['sales'] = cens['sales'].apply(lambda x : condExp(x, array_of_lambdacand[-1])).astype(float)
                
                # Combine the conditional expecation with the uncensored demand
                array_of_lambdacand.append((sum(cens['sales']) + sum(noncens['sales']))/(len(cens['sales']) + len(noncens['sales'])))
                
                # Add the convergence if the iteration continues
                number_of_convergence += 1
                
            # Append final the result to array of lambda candidates
            mu.append(array_of_lambdacand[-1])

            # Return the 'cens' demand to its default value
            cens = df.loc[df['stockout'] == 1]
        
        # Append the result to the array of clusters
        one_to_seven_clusters.append(mu)
        
        # Find the average of iteration needed to converge
        array_of_convergency.append(number_of_convergence/number_of_clusters)
    
    # Append cluster to array of clusters
    array_of_clusters.append(one_to_seven_clusters)
    
    # Append number of convergence to array of convergency
    array_of_average_to_converge.append(array_of_convergency)

    # Finding the Least Difference Between Clusters

    # In this process, the least difference between clusters will be counted
    # Input : The array of clusters from 1 to 7
    # Output : The least difference between clusters 1 to 7
    
    # Variable to store array differences
    array_of_differences = []
    for i in range(0, len(one_to_seven_clusters)):
        
        # Convert and sort the shape of the clusters
        one_to_seven_clusters[i] = list(dict.fromkeys(one_to_seven_clusters[i]))
        one_to_seven_clusters[i].sort()
        
        # Initialization
        diff = one_to_seven_clusters[0][0]
        if(len(one_to_seven_clusters[i]) > 1):
            for j in range(0, (len(one_to_seven_clusters[i]) - 1)):
                if((one_to_seven_clusters[i][j + 1] - one_to_seven_clusters[i][j]) < diff):
                    # The difference is the least difference between each cluster member
                    diff = one_to_seven_clusters[i][j + 1] - one_to_seven_clusters[i][j]
        differences = [diff, i + 1]
        array_of_differences.append(differences)
    
    # Sort the least differences
    array_of_differences.sort()
    least_difference = array_of_differences[0]
    array_of_difference.append(least_difference)

    # Finding the Log-likelihood 

    # In this process, the BIC values of the final mu will be counted
    # Input : The mu, the average values
    # Output : The BIC numbers
    
    # Variable to store optimal cluster
    candidates_of_optimal_cluster = []
    for i in range(0, len(one_to_seven_clusters)):
        
        # Variable to store variances (L)
        list_of_variances = []
        
        for j in range(0, len(one_to_seven_clusters[i])):
            
            # Variable to store sum of conditional expectations
            temp_results = 0
            
            for k in range(0, len(new_demands)):
                # Add all the sum of conditional expectation
                temp_results += float(condExp(int(new_demands[k]), one_to_seven_clusters[i][j]))
            
            # Add all results to an array
            list_of_variances.append(temp_results)
        
        # Count the average of variances average
        loglikelihood = -2 * (statistics.mean(list_of_variances)) + i * math.log(len(new_demands))
        candidates = [loglikelihood, i+1]
        candidates_of_optimal_cluster.append(candidates)
    
    # Sort and take the most optimal candidates
    candidates_of_optimal_cluster.sort()
    most_optimal_bic = candidates_of_optimal_cluster[0]
    array_of_BIC.append(most_optimal_bic)

# Export the data to dataframe
data_of_clusters = pd.DataFrame(array_of_clusters, columns=['1', '2', '3', '4', '5', '6', '7'])
data_of_average_to_converge = pd.DataFrame(array_of_average_to_converge, columns=['1', '2', '3', '4', '5', '6', '7'])
data_of_BIC = pd.DataFrame(array_of_BIC, columns=['value', 'cluster'])
data_of_difference = pd.DataFrame(array_of_difference, columns=['value', 'cluster'])

# Export the data to CSV
data_of_clusters.to_csv('data_of_clusters_3.csv')
data_of_average_to_converge.to_csv('data_of_average_to_converge_3.csv')
data_of_BIC.to_csv('data_of_BIC_3.csv')
data_of_difference.to_csv('data_of_difference_3.csv')
