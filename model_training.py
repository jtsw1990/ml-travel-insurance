import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics

def main():
    
    df = pd.read_csv('data/travel_insurance.csv')
    df = df[[
        #'Agency', 'Agency Type', 'Distribution Channel', 'Product Name', 'Destination', 'Gender',
        'Duration', 'Age', 'Claim'
    ]]
    x = df.drop('Claim', axis=1)
    y = df['Claim']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

    model = lgb.LGBMClassifier(learning_rate=0.05, max_depth=-5, random_state=123)
    model.fit(x_train, y_train, eval_set=[(x_test, y_test), (x_train, y_train)], verbose=20, eval_metric='logloss')

    print('Training accuracy {:.4f}'.format(model.score(x_train,y_train)))
    print('Testing accuracy {:.4f}'.format(model.score(x_test,y_test)))

    print(metrics.classification_report(y_test,model.predict(x_test)))


if __name__ == '__main__':
    main()