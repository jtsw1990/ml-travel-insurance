import os
import neptune.new as neptune
from modules.data_reader import read_data
from data_modelling import data_modelling_pipeline
from model_training import model_training_pipeline
from dotenv import load_dotenv


load_dotenv()


def train_model():

    # Instantiate neptune experiement within context
    with (neptune.init(project="jtsw1990/ml-travel", api_token=os.getenv('neptune_api_token'))) as run:

        # Read in data
        df = read_data()

        # Process and extract features
        x_train, x_test, y_train, y_test = data_modelling_pipeline(df)

        # Train model
        results = model_training_pipeline(x_train, x_test, y_train, y_test)

        # Output to neptune.ai
        run['classification_report'] = results


if __name__ == '__main__':
    train_model()
