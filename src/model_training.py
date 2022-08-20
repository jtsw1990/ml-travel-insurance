import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from modules.data_reader import read_data
from modules.feature_transformer import one_hot_encode
from modules.data_wrangler import remove_non_positive_premiums, remove_outlier_ages, remove_outlier_duration

def main():
    
    df = read_data()
    df = remove_outlier_ages(df, 100)
    df = remove_outlier_duration(df, 547)
    df = remove_non_positive_premiums(df)
    df = df[[
        'Agency', 'Agency Type', 'Distribution Channel', 'Product Name', 'Destination', # 'Gender',
        'Duration', 'Age', 'Claim'
    ]]
    agency = one_hot_encode(df, 'Agency')
    agency_type = one_hot_encode(df, 'Agency Type')
    distribution_channel = one_hot_encode(df, 'Distribution Channel')
    product_name = one_hot_encode(df, 'Product Name')
    destination = one_hot_encode(df, 'Destination')

    df_encoded = pd.concat([df[['Duration', 'Age', 'Claim']], agency, agency_type, distribution_channel, product_name, destination], axis=1)
    
    x = df_encoded.drop('Claim', axis=1)
    y = df_encoded['Claim']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

    model = lgb.LGBMClassifier(learning_rate=0.05, max_depth=-5, random_state=123)
    model.fit(x_train, y_train, eval_set=[(x_test, y_test), (x_train, y_train)], eval_metric='logloss')

    print('Training accuracy {:.4f}'.format(model.score(x_train,y_train)))
    print('Testing accuracy {:.4f}'.format(model.score(x_test,y_test)))

    print(metrics.classification_report(y_test,model.predict(x_test)))


if __name__ == '__main__':
    main()