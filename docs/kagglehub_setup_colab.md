# Google Colab Setup for `kagglehub` API keys

- Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings) and "Create New Token" on your API key
- Download and open the kaggle.json file

- In Google Colab, go to Secrets -> "Add New Secret"
- Create and enable KAGGLE_USERNAME and KAGGLE_KEY variables with the content of the JSON file

Datasets are now downloadable through `kagglehub.download_dataset(dataset_name)` in your Colab environment. 
