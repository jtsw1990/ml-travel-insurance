# To add inference piece which links back to some arbitrary business case
# Maybe advertising costs for potential claimants?

import lightgbm as lgb
from datetime import datetime
from sklearn import metrics


def model_training_pipeline(x_train, x_test, y_train, y_test):

    seed = 123
    model = lgb.LGBMClassifier(learning_rate=0.05, max_depth=-5, random_state=seed)
    model.fit(x_train, y_train, eval_set=[(x_test, y_test), (x_train, y_train)], eval_metric='logloss')

    print('Training accuracy {:.4f}'.format(model.score(x_train, y_train)))
    print('Testing accuracy {:.4f}'.format(model.score(x_test, y_test)))

    results = metrics.classification_report(y_test, model.predict(x_test), output_dict=True)
    # Export model
    model.booster_.save_model(f"models/{datetime.now().strftime('%Y%m%d%H%M%S')}_lgbmc_base.txt")

    return(results)
