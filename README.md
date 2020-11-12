# IMPROVING DEMAND FORECASTS BY USING HIGH-FREQUENCY SALES DATA

## Abstract

As one of the most classic problems in the world of supply and demand, there have been many suggestions to solve the high-frequency sales data. Predicting the uncertainty of the demand is very helpful for retail business, mainly the ones that cannot be stored like newspaper or bread. There are two possibility that can happen in this case, an uncensored case where the demand is lower than the stock, and censored case where the demand is bigger than the stock. 

In this thesis, a new method is offered by using the Expectation-Maximization (EM) Algorithm to predict the demands based on the censored and the uncensored demands. This Algorithm tries to predict the demand based on past uncensored and censored data. The model generated from EM Algorithm will be tested by Bayesian Information Criterion (BIC). At the end, it is proved that EM Algorithm can successfully build a parsimonious model for Newsvendor problem.

## Background

Newsvendor problem or single-period problem is a problem in supply chain management to find a product’s order quantity that maximizes the expected profit under probabilistic demand (Khouja, 1999). It is a major challenge in supply chain management practice and research (Huber et al., 2019a). Newsvendor problems assumes that a company buys stocks at the beginning of a time period and sells them in a certain period. At the end of the day, a company should consider two costs: holding costs (the cost of discarding unsold goods) and shortage cost (costs if a company runs out of the goods in the middle of a period). (Oroojlooyjadid, Snyder and Takáč, 2017).

Because the importance of this problem, many companies try to predict the order quantity. One of the main examples of this is Khouja (1999) where the writer offers 11 ways to extend the newsvendor problem. Recently, Fildes, Ma and Kolassa  (2019) also offers the methods to forecast newsvendor problems, but different from Khouja, they shows the solution on the perspective of management side. Anupindi, Dada and Gupta (1998) also tries to solve this problem by using an application of a Vending Machine.

Nowadays, Big Data is used to solve some problems of uncertainty, including Newsvendor problem. Big Data usage can be tracked in 2005, where Bertsimas and Thiele (2005) used a Data-Driven approach to tackle Newsvendor problems. This approach also done by Sachs and Minner (2014) that expands the scope to a censored demand observation, and Huber et al., (2019) the most recent publication that already used a Machine Learning Approach.

Oroojlooyjadid, Snyder and Takáč, (2017) approach on Newsvendor Problem used a Deep Learning Method, a type of Machine Learning Algorithm that aims to build a model between inputs and outputs but have more wider usage, such as in time series analysis and weather prediction. In a larger scope, Jain, Rudi and Wang, (2015) considers the sales quantity, occurrence, and demand occurrence to solve the Newsvendor Problem. Finally, Rudin and Vahn (2014) provides the newsvendor problem using Machine Learning, considering the features of the data, not only the demand itself. 

In the present time, a demand that happens in a single day have some certain features that shows the reason why a demand happens so. For example, a certain demand d, on an ice cream store can be large if it happens on a summer day. This “summer” information is called the feature. This feature-and-demand relation of Newsvendor Problem is already analyzed on Rudin and Vahn (2014), using a regression approach.

One of the Machine Learning algorithms that specifically analyzes not just the demand is the Expectation-Maximization algorithm. In this algorithm, beside analyzing the demand, it can also analyze the relevant features of a demand. For instance, if a certain demand shows, are there some specific features that affects that number? Because of this consideration, this project will try to solve the Newsvendor problem with the EM Algorithm by creating a Parsimonious Model for a High-Frequency Data.

## Review of Literature

### Newsvendor Problem

Demand Forecast or Newsvendor Problem is a problem to find a product’s quantity of order that ensures companies gets the maximum profit (Stevenson, 2012), firstly discussed by Morse and Kimball (1951). It assumes that on a single period, the product is bought at the beginning and sold by the end of it.

The Newsvendor Problem can be described as follows. Consider a vendor buys y items for price c to be sold with price p in a single day, with D items as the demand number. At the end of the day, the unsold goods (if available) are sold with price s. This situation creates three possible condition:

	Demand is bigger than the order size (D > y),  that results in “underage/shortage costs” = p – c = cp
	Demand is less than the order size (D < y), that results in “overage/holding costs” = c – s = ch 
	Demand is equal to the order size. Since both underage costs and overage costs reduces the profit, this condition is desired to ensure companies gets the maximum profit.

Newsvendor problem approach is to minimize both shortage costs and holding costs, since it is almost impossible in the real-world condition to make it zero, because the uncertainty of the demand. (Stevenson, 2012), (Chhajed and Lowe, 2008)

The mathematical formulation of the newsvendor problem is finding the cost between the lesser between demand and order size. If the demand is less than the order size, the difference will be multiplied by the shortage costs, and if the order size is less than the demand, the difference will be multiplied by the holding costs, with the formula shown below (Khouja, 1999):

min┬y⁡〖C(y)〗=E_D |c_p (D-y)^++c_h (y-D)^+ |

Where (α)+:= max {0,a}. 

The first way to find the high-frequency demands is to determine the distribution of the demand. In newsvendor problems, this demand is often simulated into distributions, mainly using normal or Poisson distribution. After determining the demand, the order quantity will be set based on the demand so it will reduce both shortage and holding costs.

## Results

EM Algorithm is known from its guarantee for it to reach a convergent value, although the starting point is random. In this EM Algorithm implementation, all the values lead to a single value, where the iteration to another value is terminated when it reached a single value.

With a default λ of 15, it can be assumed that all the cluster value will converge to 15. While this hypothesis is not entirely false, the graph shows that most of the cluster values of the EM Algorithm lies between 11 and 14, with most of the values lies around 12 and 13. 

This trend did not change, although the number of average values is changed, and the number of repetitions is added. If the cluster value with the default λ is set to 20, most of the clusters will lie between 17 and 19.

The reason that this happens is because the simulation of data. After the Poisson distribution determines the total number of average daily demands, and each demands determines another hourly viewed demands using another Poisson Distribution, the original average might be distorted and the EM Algorithm only captures the value around the hourly view demand on a single day.

For instance, consider the daily viewed in 3 days, still with the same average value, the average daily demands are 11, 13, and 14 days. These numbers are totally normal to be generated in Poisson distribution, since Poisson distribution will randomly generate the value around the average with the standard deviation is equal to the mean. Furthermore, these daily demands view of 11, 13, and 14 can yield another result that could be less than 15. 

Another reason for this condition is because the higher demand frequency (more than 15) can have a higher probability of stockout. As these demands have a higher probability of stockout, they will also have a higher probability of being censored. The censored data is processed in the EM Algorithm, and since the EM Algorithm processed the censored data to a smaller value to the be counted in the clusters, the clusters generated a new, smaller result that will be smaller than the default average (15). 

Nevertheless, all the clusters value could successfully differ the different daily demands to multiple clusters. This part will be explained more in the next parts of this dissertation.

## Conclusions

The High Frequency Data will be processed using EM Algorithm to create a model that shows multiple clusters that can be chosen to purchase a daily demand. The High Frequency Data in this thesis represents an hourly demand noted in a week. The EM Algorithm will analyse the best number of clusters that represents the model, verified by the Bayesian Information Criterion. There are seven possible clusters, representing the number of days in a single week.  

The model shows that using one or two clusters is enough to create a model that can fit into a demand data. This is shown by the BIC values that mostly shown an optimal value of one or two clusters and the number of iterations that mostly shown the best value between 1 to 3 values.
