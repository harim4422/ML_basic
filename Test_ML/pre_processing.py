def pre(dataset):
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    X=dataset.drop(['style'],axis=1)
    y=dataset['style']
    X=X.values
    y = labelencoder.fit_transform(y)
    return X,y
def split(X,y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train , X_test , y_train, y_test

