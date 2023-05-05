import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score


def preprocess(file):
    data = pd.read_csv(file)
    df = data[['Cuisines', 'Name', 'Rating',
               'Address', 'Restaurant_Type', 'Category_Type']]
    df['Cost'] = data['Approx_Cost(For 2 Persons)']/2
    return df


def predict(budget, people, Cuisines):

    df = preprocess('Restaurants_DataSet.csv')

    encoder = LabelEncoder()
    df['ECuisines'] = encoder.fit_transform(df['Cuisines'])
    df['ECost'] = pd.to_numeric(df['Cost'])

    X_train, X_test, y_train, y_test = train_test_split(
        df[['ECuisines', 'ECost']], df['Name'], test_size=0.2
    )

    # Create and train random forest classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # import pickle
    # pickle.dump(model, open("Rsmodel.pkl", "wb"))

    # Make predictions on testing data and recommend top 10 restaurants

    predictions = model.predict(X_test)

    print(predictions)
    accuracy = model.score(X_test, y_test)
    print(f'Model accuracy: {accuracy:.2f}')
    onebudget = budget / people
    percen = 0.2 * onebudget
    a = df[(df['Cuisines'] == Cuisines) & (
        df['Cost'].between(onebudget - percen, onebudget + percen))]
    a.sort_values(by="Rating", ascending=False)
    return a[['Name', 'Rating', 'Address']].values.tolist()


budget = 12000
people = 6
Cuisines = 'BBQ'
print(predict(budget, people, Cuisines))
