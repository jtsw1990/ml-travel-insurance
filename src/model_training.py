import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from modules.data_reader import read_data
from modules.feature_transformer import bin_destination, bin_product, one_hot_encode
from modules.data_wrangler import remove_non_positive_premiums, remove_outlier_ages, remove_outlier_duration
from modules.product_map import product_mapping

def main():

    # Read in data
    df = read_data()

    # Process and clean data
    df = remove_outlier_ages(df, 100)
    df = remove_outlier_duration(df, 547)
    df = remove_non_positive_premiums(df)

    # Feature selection and engineering
    df['destination_band'] = bin_destination(df)
    df['product_type'] = bin_product(df, product_mapping)
    df = df[[
        'Agency', 'Agency Type', 'Distribution Channel', 'product_type', 'destination_band', # 'Gender',
        'Duration', 'Age', 'Claim'
    ]]
    agency = one_hot_encode(df, 'Agency')
    agency_type = one_hot_encode(df, 'Agency Type')
    distribution_channel = one_hot_encode(df, 'Distribution Channel')
    product_name = one_hot_encode(df, 'product_type')
    destination = one_hot_encode(df, 'destination_band')

    df_encoded = pd.concat([df[['Duration', 'Age', 'Claim']], agency, agency_type, distribution_channel, product_name, destination], axis=1)
    
    x = df_encoded.drop('Claim', axis=1)
    y = df_encoded['Claim']

    # Train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

    # Model training
    model = lgb.LGBMClassifier(learning_rate=0.05, max_depth=-5, random_state=123)
    model.fit(x_train, y_train, eval_set=[(x_test, y_test), (x_train, y_train)], eval_metric='logloss')

    print('Training accuracy {:.4f}'.format(model.score(x_train, y_train)))
    print('Testing accuracy {:.4f}'.format(model.score(x_test, y_test)))

    print(metrics.classification_report(y_test,model.predict(x_test)))


if __name__ == '__main__':
    main()