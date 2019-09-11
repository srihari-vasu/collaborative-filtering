import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error 
import warnings
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")

#Open the CSV file with the userid and movieid along with their ratings
ratings = pd.read_csv("ratings.csv")

#calculate the number of users and movies
n_users = ratings.userId.unique().shape[0]
n_movies = ratings.movieId.unique().shape[0]

#create a list of users and movies with their id and use their index to idnetify them
utag = list(set(ratings.userId))
mtag = list(set(ratings.movieId))
utag.sort()
mtag.sort()

#create a rating matrix between the user and the movies and populat it with the ratings
rate_matrix = np.zeros((n_users, n_movies))
for row in ratings.itertuples():
    rate_matrix[utag.index(row.userId)][mtag.index(row.movieId)] = row.rating

#Split the data obtained into traing and testing data by removing 5 ratings for each user
test_data = np.zeros(rate_matrix.shape)
train_data = rate_matrix.copy()
for u in range(rate_matrix.shape[0]):
    test_rate = np.random.choice(rate_matrix[u][:].nonzero()[0], size = 5, replace = False)
    train_data[u, test_rate] = 0
    test_data[u, test_rate] = rate_matrix[u, test_rate]
    
#print heads of test and train data
#print(test_data)
#print(train_data)

#calculating similarity matrix for user and movies and add an epsilon to avoid divide by zero error
u_sim = rate_matrix.dot(rate_matrix.T) + 1e-8
m_sim = rate_matrix.T.dot(rate_matrix) + 1e-8


#now the similarity data has to be normalized to have a standard
u_n_vector = np.array([np.sqrt(np.diagonal(u_sim))])
m_n_vector = np.array([np.sqrt(np.diagonal(m_sim))])
u_sim = (u_sim / u_n_vector / u_n_vector.T)
m_sim = (m_sim / m_n_vector / m_n_vector.T)

#users may be more inclined towards rating movies. ie., some users tend to rate more for all movies and some tend to rate less
u_sim = (u_sim - train_data.mean(axis = 1)[:, np.newaxis]).copy()
m_sim = (m_sim - train_data.mean(axis = 0)[np.newaxis, :]).copy()

print(u_sim)
print()
print(m_sim)

'''
To predict we find the top 'n' similar users and then predict.
To get the best value for 'n' we run a loop over the below function with different 'n' values
'''
def get_top_n_users_pred(data_train, simi, n):
    u_pred = np.zeros(data_train.shape)
    for i in range(data_train.shape[0]):
        top_n_users = [np.argsort(simi[:, i])[:-(n+1):-1]]
        for j in range(data_train.shape[1]):
            u_pred[i, j] = (simi[i, :][top_n_users].dot(data_train[:, j][top_n_users])) / np.sum(np.abs(simi[i, :][top_n_users]))
    return u_pred

def get_top_n_movies_pred(data_train, simi, n):
    m_pred = np.zeros(data_train.shape)
    for i in range(data_train.shape[1]):
        top_n_movies = [np.argsort(simi[:, i])[:-(n+1):-1]]
        for j in range(data_train.shape[0]):
            m_pred[j, i] = (simi[i, :][top_n_movies].dot(data_train[j, :][top_n_movies]).T) / np.sum(np.abs(simi[i, :][top_n_movies]))
    return m_pred

#u_pred = get_top_n_users_pred(train_data, u_sim, 50)
#m_pred = get_top_n_movies_pred(train_data, m_sim, 50)

'''
To find the validity of our algorithms, 
we find the mean square error values for them and check which 'n' value is best.
Below is the function to calculate MSE.
'''
def msqe(y, y_cap):
    return mean_squared_error(y[y_cap.nonzero()].flatten(), y_cap[y_cap.nonzero()].flatten())

#initialize a list of possible 
possible_n_values = [25, 50, 100, 200]
u_train_mse, u_test_mse, m_train_mse, m_test_mse = [], [], [], []

for n in possible_n_values:
    u_pred = get_top_n_users_pred(train_data, u_sim, n)
    m_pred = get_top_n_movies_pred(train_data, m_sim, n)
    
    #Mean square error for n top users 
    u_train_mse.append(msqe(u_pred, train_data))
    u_test_mse.append(msqe(u_pred, test_data))
    
    #Mean square error for n top movies
    m_train_mse.append(msqe(m_pred, train_data))
    m_test_mse.append(msqe(m_pred, test_data))

print("User Based Training Error : " ,u_train_mse)
print("User Based Testing Error : " ,u_test_mse)
print("Movie Based Training Error : " ,m_train_mse)
print("Movie Based Testing Error : " ,m_test_mse)

plt.plot(possible_n_values, u_train_mse)
plt.plot(possible_n_values, u_test_mse)
plt.plot(possible_n_values, m_train_mse)
plt.plot(possible_n_values, m_test_mse)
plt.legend(['User Based Training Error', 'User Based Testing Error', 'Movie Based Training Error', 'Movie Based Testing error'])
plt.show()