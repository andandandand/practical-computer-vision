# Google Colab Setup for `kagglehub` API keys

- Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings) and "Create New Token" on your API key
- Download and open the kaggle.json file

- In Google Colab, go to Secrets -> "Add New Secret"
- Create and enable KAGGLE_USERNAME and KAGGLE_KEY variables with the content of the JSON file

Datasets are now downloadable through `kagglehub.download_dataset(dataset_name)` in your Colab environment. 

A function like the following allows you to set up `kagglehub` and download a dataset.

```python
import kagglehub
def setup_kagglehub_credentials():
    """
    Configure KaggleHub credentials using Google Colab secrets.
    This allows authenticated access to Kaggle datasets.
    """
    # Get Kaggle credentials from Colab secrets
    kaggle_username = userdata.get('KAGGLE_USERNAME')
    kaggle_key = userdata.get('KAGGLE_KEY')

    if not kaggle_username or not kaggle_key:
        raise ValueError(
            "Kaggle credentials not found in Colab secrets. "
            "Please add your KAGGLE_USERNAME and KAGGLE_KEY to your Colab secrets."
        )

    # Set environment variables for kagglehub
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key

    print("✅ KaggleHub credentials configured successfully")

setup_kagglehub_credentials()

# Download latest version
path = kagglehub.dataset_download("andandand/presidential-dog")

print("Path to dataset files:", path)
```
