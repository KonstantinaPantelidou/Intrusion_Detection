from sklearn import datasets, metrics, model_selection, ensemble
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import datasets
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from itertools import cycle
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.decomposition import PCA

train_data = pd.read_csv("NSL-KDDTrain.csv")
test_data = pd.read_csv("NSL-KDDTest.csv")

print("train data shape1")
print(train_data.shape)
print()
#(125973, 41)
print("test data shape1")
print(test_data.shape)
print()
#(22544, 42)

#removal of columns from train_data that are duplicates
train_data = train_data.drop_duplicates()
#check and manage NULL values in a data frame
print(train_data.isnull().sum())
print()
print("train data shape3")
print(train_data.shape)
print()
#(125957, 41)

#checking the cardinality of categorical columns
sns.countplot(x='protocol_type', data=train_data)
plt.xticks(rotation=90)
plt.show()


sns.countplot(x='service', data=train_data)
plt.xticks(rotation=90)
plt.show()


sns.countplot(x='flag', data=train_data)
plt.xticks(rotation=90)
plt.show()

train_data['flag'].unique()
train_data['flag'] = np.where(train_data['flag']=='S0','S1',  train_data['flag'])
train_data['flag'] = np.where(train_data['flag']=='S0','S2',  train_data['flag'])
train_data['flag'] = np.where(train_data['flag']=='S0','S3',  train_data['flag'])
train_data['flag'] = np.where(train_data['flag']=='S2','S3',  train_data['flag'])
train_data['flag'] = np.where(train_data['flag']=='S1','S3',  train_data['flag'])
train_data['flag'] = np.where(train_data['flag']=='RSTO','RSTOS0', train_data['flag'])

sns.countplot(x='flag', data=train_data)
plt.xticks(rotation=90)
plt.show()

train_data['service'].unique()
train_data['service'] = np.where(train_data['service']=='http_443','http_8001',  train_data['service'])
train_data['service'] = np.where(train_data['service']=='http_443','http_2784',  train_data['service'])
train_data['service'] = np.where(train_data['service']=='http_8001','http_2784',  train_data['service'])
train_data['service'] = np.where(train_data['service']=='pop_2','pop_3',  train_data['service'])
train_data['service'] = np.where(train_data['service']=='ftp_data','ftp',  train_data['service'])


sns.countplot(x='service', data=train_data)
plt.xticks(rotation=90)
plt.show()

print("train data shape3")
print(train_data.shape)
print()


#Converting categorical variable into dummy/indicator variables
df= pd.get_dummies(train_data,  columns=['protocol_type', 'service', 'flag'])

print(df)
print("meta to dummies")
print(df.dtypes)
print(df.shape)


for col in df.columns: 
    print(col) 

#drop columns that do not exist in test_data
df = df.drop(['service_aol', 'service_harvest',
	'service_red_i', 'service_urh_i'], axis=1)

print("df data shape")
print(df.shape)
#(125957, 109)

#filling NA/NaN values
df=df.fillna(0)
print(df)
print("df after fillna")
print(df.dtypes)
print(df)
print(df.shape)



#remove outliers
#find Q1, Q3, and interquartile range for each column
Q1 = df.quantile(q=.25)
Q3 = df.quantile(q=.75)
IQR = df.apply(stats.iqr)

#only keep rows in dataframe that have values within 1.5*IQR of Q1 and Q3
df = df[~((df < (Q1-1.5*IQR)) | (df > (Q3+1.5*IQR))).any(axis=1)]

#find how many rows are left in the dataframe 
print('df after outlier removal')
print(df.shape)
print(type(df))
#(7339, 109)


#scaling the data
df = StandardScaler().fit_transform(df)
print("df after scaling")
print(df)
print(df.shape)

print("test data shape")
print(test_data.shape)
print(test_data.dtypes)
#(22544, 42)

#removal of columns from train_data that are duplicates
test_data = test_data.drop_duplicates()
#check and manage NULL values in a data frame
print(test_data.isnull().sum())
print("test data after drop duplicates")
print(test_data.shape)
print(test_data.dtypes)
#(22541, 42)

#checking the cardinality of categorical columns
sns.countplot(x='protocol_type', data=test_data)
plt.xticks(rotation=90)
plt.show()

sns.countplot(x='service', data=test_data)
plt.xticks(rotation=90)
plt.show()


sns.countplot(x='flag', data=test_data)
plt.xticks(rotation=90)
plt.show()


test_data['flag'].unique()
test_data['flag'] = np.where(test_data['flag']=='S0','S1',  test_data['flag'])
test_data['flag'] = np.where(test_data['flag']=='S0','S2',  test_data['flag'])
test_data['flag'] = np.where(test_data['flag']=='S0','S3',  test_data['flag'])
test_data['flag'] = np.where(test_data['flag']=='S2','S3',  test_data['flag'])
test_data['flag'] = np.where(test_data['flag']=='S1','S3',  test_data['flag'])
test_data['flag'] = np.where(test_data['flag']=='RSTO','RSTOS0', test_data['flag'])

sns.countplot(x='flag', data=test_data)
plt.xticks(rotation=90)
plt.show()

test_data['service'].unique()
test_data['service'] = np.where(test_data['service']=='pop_2','pop_3',  test_data['service'])
test_data['service'] = np.where(test_data['service']=='ftp_data','ftp',  test_data['service'])


sns.countplot(x='service', data=test_data)
plt.xticks(rotation=90)
plt.show()



print("test data shape3")
print(test_data.shape)
#(22541, 42)


#Converting categorical variable into dummy/indicator variables
test_data = pd.get_dummies(test_data,  columns=['protocol_type', 'service', 'flag', 'target'])

print(test_data)
print("test data meta to dummies")
print(test_data.shape)
print()
for col in test_data.columns: 
    print(col) 


#filling NA/NaN values
test_data=test_data.fillna(0)
print(test_data)
print("test data meta to fillna")
print(test_data.shape)
#(22541, 112)

print(test_data['target_attack'])


print("test data")
print(test_data.shape)
#find how many rows are left in the dataframe 
print(test_data.shape)
print(test_data.dtypes)
print(test_data)


print('test_data.shape')
print(test_data.shape)
#(22541, 112)


#converting test_data to DataFrame
test_data = pd.DataFrame(test_data) 

##1=attack and 0=normal
y_test = test_data['target_attack']
print(y_test)
#1=attack k 0=norm


#removal of 'target_attack' and 'target_normal' from test_data after get_dummies
test_data = test_data.drop(['target_attack', 'target_normal'], axis=1)
print(test_data.shape)
#(22541, 110)

#scaling data
test_data = StandardScaler().fit_transform(test_data)
print("teest data after scaling")
print(test_data)
print(test_data.shape)
#(22541, 110)
print('y_test.shape')
print(y_test.shape)

#applying PCA for dimensionality reduction to 2 components
pca = PCA(n_components=2)
df = pca.fit_transform(df)
df = pd.DataFrame(df)
print('df shape')
print(df.shape)

pca = PCA(n_components=2)
test_data = pca.fit_transform(test_data)
test_data = pd.DataFrame(test_data)

#applying k-Means Clustering Algorithm
kmeans = KMeans(n_clusters=2)  
kmeans=kmeans.fit(df.iloc[:, :10])  
y_predicted = kmeans.predict(test_data)
print(y_predicted)
print(' y predicted shape')
print(y_predicted.shape)

print("y_test shape")
print(y_test.shape)
print(y_test)

print("y_predicted shape")
print(y_predicted.shape)
print(y_predicted)

#y_test = y_test.iloc[:-21489]
print(y_test.shape)
print(type(y_test))
print(y_predicted.shape)

#calculation of silhouette score
score = silhouette_score(df, kmeans.labels_, metric='euclidean')
print('Silhouette Score - k-Means Clustering:')
print(score)
#calculation of accuracy using test data
accuracy = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy - k-Means Clustering:')
print(accuracy)
print("Recall - k-Means Clustering:")
print(metrics.recall_score(y_test, y_predicted, average='macro', zero_division='warn'))
print("Precision - k-Means Clustering:")
print(metrics.precision_score(y_test, y_predicted, average='macro'))
print("F1 - k-Means Clustering:")
print(metrics.f1_score(y_test, y_predicted, average='macro', zero_division='warn', labels=np.unique(y_predicted)))

#calculation of rand score
#Similarity measure between two clusterings by considering all pairs of samples and 
#counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.
rand=metrics.rand_score(y_test, y_predicted)
print(rand)

#The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples and 
#counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.
rand=metrics.adjusted_rand_score(y_test, y_predicted)
print(rand)

#Adjusted Mutual Information (AMI) is an adjustment of the Mutual Information (MI) score to account for chance. 
#It accounts for the fact that the MI is generally higher for two clusterings with a larger number of clusters, 
#regardless of whether there is actually more information shared
rand=metrics.adjusted_mutual_info_score(y_test, y_predicted)
print(rand)


print(type(y_predicted))
y_predicted= pd.DataFrame(y_predicted)
print(type(y_predicted))

print(kmeans.labels_)
print(y_test)
centroids = kmeans.cluster_centers_
print(centroids)

# calculating distortion for a range of number of clusters
distortions = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(df)
    distortions.append(km.inertia_)

# plot
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

print(classification_report(y_test, y_predicted))


#applying Agglomerative Clustering Algorithm

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(df)

print(cluster.labels_)
predictedlabels = cluster.labels_
print(predictedlabels.shape)
print(type(y_test))
predictedlabels = pd.DataFrame(predictedlabels)
predictedlabels = predictedlabels.iloc[:-5895]

score = silhouette_score(df, cluster.labels_, metric='euclidean')
print('Silhouette Score - Agglomerative Clustering:')
print(score)
print()


#applying MeanShift Clustering Algorithm
cluster = MeanShift(bandwidth=2).fit(df)

score = silhouette_score(df, cluster.labels_, metric='euclidean')
print('Silhouette Score - MeanShift Clustering:')
print(score)

y_predicted = cluster.predict(test_data)

accuracy = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy - MeanShift Clustering:')
print(accuracy)
print("Recall - MeanShift Clustering:")
print(metrics.recall_score(y_test, y_predicted, average='macro', zero_division='warn'))
print("Precision - MeanShift Clustering:")
print(metrics.precision_score(y_test, y_predicted, average='macro'))
print("F1 - MeanShift Clustering:")
print(metrics.f1_score(y_test, y_predicted, average='macro', zero_division='warn', labels=np.unique(y_predicted)))

print(classification_report(y_test, y_predicted))

#applying AffinityPropagation Clustering Algorithm
print('test_data.shape')
print(test_data.shape)
print('y_test.shape')
print(y_test.shape)
clustering = AffinityPropagation(random_state=5).fit(df)  


y_predicted = clustering.predict(test_data)

print(' y predicted shape')
print(y_predicted.shape)

print("y_test shape")
print(y_test.shape)
print(y_test)

print("y_predicted shape")
print(y_predicted.shape)
print(y_predicted)

print(y_test.shape)
print(type(y_test))
print(y_predicted.shape)

#calculation of silhouette score
score = silhouette_score(df, clustering.labels_, metric='euclidean')
print('Silhouette Score - AffinityPropagation Clustering:')
print(score)
#calculation of accuracy using test data
accuracy = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy -AffinityPropagation Clustering:')
print(accuracy)


