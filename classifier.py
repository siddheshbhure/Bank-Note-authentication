import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle 


def main():
	df = pd.read_csv('BankNote_Authentication.csv')
	X = df.iloc[:,:-1]
	y = df.iloc[:,-1]

	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)

	classifier = RandomForestClassifier()
	classifier.fit(X_train,y_train)

	pickle_out = open('classifier.pkl','wb')
	pickle.dump(classifier,pickle_out)
	pickle_out.close()




if __name__ == '__main__':
	main()