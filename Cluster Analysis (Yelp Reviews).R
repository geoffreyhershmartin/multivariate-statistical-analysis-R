# This code runs different cluster analyses methods on the Yelp Reviews dataset. Here, 
# the different clustering methods I try are Ward's Method, the average method and K-means
# clustering. For each method, I also employ different distance measures namely:
# euclidean, manhattan and maximum distance measures. Finally, I run KMeans clustering on 
# the data and cross-check the results with our hierarchical clustering methods. Ultimately,
# our goal is to see if there are groups of users that review in similar ways.

library(cluster)
library(xtable)
yelp <- read.csv("Data/yelp_cleaned.csv")

# Standardize and Sample
yelp_scaled <- scale(yelp[])
set.seed(1)
index <- sample(1:nrow(yelp_scaled), 100)
yelp_sample <- yelp_scaled[index, ]

# Calculate Distance Matrix
euclidean <- dist(yelp_sample,method="euclidean") # Euclidean
manhattan <- dist(yelp_sample,method="manhattan") # Manhattan
maximum <- dist(yelp_sample,method="maximum") # Manhattan

# Perform clustering using Ward's method
euc_ward <- hclust(euclidean, method="ward.D") # Euclidean distance, Ward's method
man_ward <- hclust(manhattan, method="ward.D") # Manhattan distance, Ward's method
max_ward <- hclust(maximum, method="ward.D") # Maximum distance, Ward's method

# Perform clustering using average method
euc_complete <- hclust(euclidean, method="complete") # Euclidean distance, average method
man_complete <- hclust(manhattan, method="complete") # Manhattan distance, average method
max_complete <- hclust(maximum, method="complete") # Maximum distance, average method

# Plots
plot(euc_ward,labels= yelp[index,1], cex=0.3, xlab="",
     ylab="Euclidean Distance",
     main="Clustering using Euclidean distance & Ward's method")
plot(man_ward,labels= yelp[index,1], cex=0.3, 
     xlab="",ylab="Manhattan Distance",
     main="Clustering using Manhattan distance & Ward's method")
plot(max_ward,labels= yelp[index,1], cex=0.3, 
     xlab="",ylab="Manhattan Distance",
     main="Clustering using Maximum distance & Ward's method")

plot(euc_complete,labels= yelp[index,1], cex=0.3, 
     xlab="",ylab="Euclidean Distance",
     main="Clustering using Euclidean distance & Complete Linkage")
rect.hclust(euc_complete, k = 6)
plot(man_complete,labels= yelp[index,1], 
     cex=0.3, xlab="",ylab="Manhattan Distance",
     main="Clustering using Manhattan distance & Complete Linkage")
rect.hclust(euc_complete, k = 6)
plot(max_complete,labels= yelp[index,1], 
     cex=0.3, xlab="",ylab="Manhattan Distance",
     main="Clustering using Maximum distance & Complete Linkage")
rect.hclust(euc_complete, k = 6)

# The dendrograms we have seen suggest that there are approximately
# 5 – 6 well defined groups, with Observation 36 being an outlier with its own group. 
# It is likely that these groups are clustered by the number of stars that that 
# review gave. However, KMeans clustering should give us a better idea as to how 
# many clusters might actually be present.

# K-means Clustering
# Calculate the within groups sum of squared error (SSE) for the number of cluster solutions selected by the user
kdata=yelp_sample
n.lev=15  #set max value for k

# Calculate the within groups sum of squared error (SSE) for the number of cluster solutions selected by the user
wss <- rnorm(10)
while (prod(wss==sort(wss,decreasing=T))==0) {
  wss <- (nrow(kdata)-1)*sum(apply(kdata,2,var))
  for (i in 2:n.lev) wss[i] <- sum(kmeans(kdata, centers=i)$withinss)}

# Calculate the within groups SSE for 250 randomized data sets (based on the original input data)
k.rand <- function(x){
  km.rand <- matrix(sample(x),dim(x)[1],dim(x)[2])
  rand.wss <- as.matrix(dim(x)[1]-1)*sum(apply(km.rand,2,var))
  for (i in 2:n.lev) rand.wss[i] <- sum(kmeans(km.rand, centers=i)$withinss)
  rand.wss <- as.matrix(rand.wss)
  return(rand.wss)
}

rand.mat <- matrix(0,n.lev,250)

k.1 <- function(x) { 
  for (i in 1:250) {
    r.mat <- as.matrix(suppressWarnings(k.rand(kdata)))
    rand.mat[,i] <- r.mat}
  return(rand.mat)
}

rand.mat <- k.1(kdata)

xrange <- range(1:n.lev)
yrange <- range(rand.mat,wss)
plot(xrange,yrange, type='n', xlab="Cluster Solution", ylab="Within Groups SSE", main="Cluster Solutions against SSE")
for (i in 1:250) lines(rand.mat[,i],type='l',col='red')
lines(1:n.lev, wss, type="b", col='blue')
legend('topright',c('Actual Data', '250 Random Runs'), col=c('blue', 'red'), lty=1)

# Here, I ran KMeans clustering with a range of cluster solutions (1 15). With our results 
# from KMeans clustering, it would seem as though our initial suspicions and dendrogram
# interpretations were accurate. Indeed, the scree plot, obtained by plotting 
# the within groups SSE against the number of clusters specified, seems to indicate that the 
# within groups SSE ‘elbows’ when there are 4 clusters. Thus, this suggests that there are 
# about 4 groups since the addition of extra clusters from then on does not improve within 
# groups SSE by much. Since 4 clusters seems to be the optimal cluster solution, we then 
# performed further analysis using 4 clusters in KMeans clustering. First, when our data 
# was plotted in the first 2 dimensions of maximum variability, i.e., Principal Component 1 
# and Principal Component 2, and specifying 4 clusters, we see that there is heavy overlap 
# between clusters 2 and 3 while clusters 1 and 4 seem to be fairly distinct.

clust.level <- 4
fit <- kmeans(kdata, clust.level)
aggregate(kdata, by=list(fit$cluster), FUN=mean)
clust.out <- fit$cluster
kclust <- as.matrix(clust.out)
kclust.out <- cbind(kclust, yelp_sample)
clusplot(kdata, fit$cluster, shade=F,
         labels=2, lines=0, color=T, lty=4, main='Principal Components plot showing K-means clusters')
d <- cbind(fit$centers, fit$size)
colnames(d)[20] <- "cluster_size"

print(xtable(d[, 1:8]), comment = FALSE)
print(xtable(d[, 9:14]), comment = FALSE)
print(xtable(d[, 14:20]), comment = FALSE)

# Groups 2 and 3 seem to have similar coefficient values for the ‘stars’ variable while 
# Groups 1 and 4 seem to have much higher and lower coefficients respectively. Intuitively, 
# we would expect such results since we would expect most reviews to rate an establishment 
# moderately whereas reviews that rate establishments exceptionally high or low are likely 
# to be out of the ordinary. Other interesting and noteworthy coefficients are the 
# coefficients to the ‘isWeekend’ and ‘words’ variables. Reviews in the cluster with higher 
# star ratings (Group 1) tend to have fewer words and are more likely to be written over the 
# weekend whereas reviews in the cluster with lower star ratings (Group 4) tend to have many 
# words and are more likely to be written on a weekday. All in all, the combined cluster 
# analysis from KMeans clustering and Hierarchical clustering seems to suggest that there 
# are approximately 5 well defined groups of reviews, grouped according to the number of 
# stars that that review gave to the establishment it was reviewing. For further analysis 
# though, we could explore the data using the Average Linkage agglomeration method and 
# permutate our linkage methods with the maximum distance metric.
