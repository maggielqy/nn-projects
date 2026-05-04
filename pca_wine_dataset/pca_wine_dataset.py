import torch
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import sklearn.metrics
import matplotlib.pyplot as plt

# load the wine dataset by uncommenting these lines
wine = datasets.load_wine(as_frame=True)
wine_df = wine.frame
wine_df.head()

# check data types that all are numerical (rather than numbers that are actually strings, for example)
wine_df.dtypes

# get the shape of the dataset
wine_df.shape


# check how many target groups there are in the last column
wine_df['target'].value_counts()



X = torch.tensor((wine_df.drop('target',axis=1)).values)
Y = torch.tensor(wine_df['target'].values)

# Make data zero-centered before applying PCA (if you're wondering why we need to do that, refer back to Question 1.1 :) )



M = M = torch.mean(X, dim=0)
std_dev = torch.std(X, dim=0)
X_torch = (X - M)/ std_dev

# perform PCA on  using the covariance matrix


cov_mat = torch.matmul(X.T,X)

# use torch.linalg.eig to get the eigenvalue decomposition of the covariance matrix

eigenvalues, eigenvectors = torch.linalg.eig(cov_mat)
eigenvalues = eigenvalues.real
eigenvectors = eigenvectors.real
eigenvalues




idx = torch.argsort(eigenvalues,descending=True)[:2]
eig_val = eigenvalues[idx]
eig_vec = eigenvectors[:,idx]
X_pca = torch.matmul(X_torch,eig_vec)


plt.scatter(X_pca.numpy()[:,0],X_pca.numpy()[:,1],c=Y.numpy())
plt.show()
# it's possible to differentiate the numbers in two dimensions.



total_var = torch.sum(eigenvalues)
var1 = eig_val[0]/total_var
var2 = eig_val[1]/total_var
print(f'PC1 explains variance: {var1.item()}')
print(f'PC2 explains variance: {var2.item()}')


from sklearn.preprocessing import StandardScaler # documentation here: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html 
from sklearn.decomposition import PCA # documentaiton here: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA


scaler = StandardScaler()
X_s = scaler.fit_transform(X)

pca = PCA(2)
X_pca_s = pca.fit_transform(X_s)
# X_pca_s

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.datasets import fetch_california_housing



housing = fetch_california_housing()
housing_x = housing.data
housing_y = housing.target
housing # gives feature names and other information
print(housing_x[0])
print(housing_y[0])

# housing

print("Feature names:", housing.feature_names)


lr = LinearRegression()
x_med = housing_x[:,0].reshape(-1,1)

lr.fit(x_med,housing_y)


coef = lr.coef_
inte = lr.intercept_
print(coef)
print(inte)


y_hat = lr.predict(x_med)
MSE = np.sum((y_hat - housing_y)**2)/len(housing_y)
MSE


plt.scatter(x_med,housing_y,alpha=0.1)
plt.plot(x_med,y_hat,c='red', alpha=0.5)
plt.show()
#median income is a good predictor for the price of a home, but it does not explain the price very well. other features should be considered.


#add 'HouseAge'
lr_new = LinearRegression()
x_3 = housing_x[:,[0,1]]

lr_new.fit(x_3,housing_y)

y_p = lr_new.predict(x_3)
mse_new = mean_squared_error(y_p,housing_y)
mse_new
# Adding the variables 'HouseAge' improves the model, as the mse gets smaller.


w0 = lr_new.intercept_ 
w1 = lr_new.coef_[0] 
w2 = lr_new.coef_[1] 

x1_min, x1_max = x_3[:, 0].min(), x_3[:, 0].max()
x2_min, x2_max = x_3[:, 1].min(), x_3[:, 1].max()

x1_range = np.linspace(x1_min, x1_max, 20)
x2_range = np.linspace(x2_min, x2_max, 20)

X1, X2 = np.meshgrid(x1_range, x2_range)

Y_plane = w0 + w1 * X1 + w2 * X2

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

scatter = ax.scatter(x_3[:, 0], x_3[:, 1], housing_y, 
                    cmap='viridis', s=15, alpha=0.3)

ax.plot_surface(X1, X2, Y_plane, 
                alpha=0.5, 
                color='red')


plt.show()
