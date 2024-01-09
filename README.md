# BBC News Classification LSTM

In today’s world, data is power. With News companies having terabytes of data stored in
servers, everyone is in the quest to discover insights that add value to the organization.
With various examples to quote in which analytics is being used to drive actions, one that
stands out is news article classification.

Nowadays on the Internet there are a lot of sources that generate immense amounts of
daily news. In addition, the demand for information by users has been growing
continuously, so it is crucial that the news is classified to allow users to access the
information of interest quickly and effectively. This way, the machine learning model for
automated news classification could be used to identify topics of untracked news and/or
make individual suggestions based on the user’s prior interests.

**Approach:** Techniques like clustering and associating rule-based algorithms can be
applied to group together similar text. The ML algorithms learn the mapping function
between the text and the tags based on already categorized data. Algorithms such as
SVM, Neural Networks, Random Forest are commonly used for text classification.

**Results:** For a given news article, the system should be able to classify them according
to various categories like Finance, Sports etc.
You have to build a solution that should recognize and classify the news articles based
on their labels.


Text documents are one of the richest sources of data for businesses.We’ll use a public dataset from the BBC comprised of 2225 articles, each labeled under one of 5 categories: business, entertainment, politics, sport or tech.

The dataset is broken into 1490 records for training and 735 for testing. The goal will be to build a system that can accurately classify previously unseen news articles into the right category.

### [Click Here To See More About The Dataset](https://www.kaggle.com/c/learn-ai-bbc/data)   


##  ML Flow Experiments
[![mice_datapipeline](https://dagshub.com/SunilKumar-ugra/BBC_News_Classification_LSTM.mlflow/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&datasetsFilter=W10%3D&modelVersionFilter=All%20Runs&selectedColumns=attributes.%60Source%60,attributes.%60Models%60,attributes.%60Dataset%60&compareRunsMode=CHART&compareRunCharts=W3sidXVpZCI6IjE3MDQxMzY1NTUzNDEycTVvYTdxOSIsInR5cGUiOiJCQVIiLCJydW5zQ291bnRUb0NvbXBhcmUiOjEwLCJtZXRyaWNLZXkiOiJUZXN0IEFjY3VyYWN5In0seyJ1dWlkIjoiMTcwNDEzNjU1NTM0MXlsaTZudjc3IiwidHlwZSI6IkJBUiIsInJ1bnNDb3VudFRvQ29tcGFyZSI6MTAsIm1ldHJpY0tleSI6IlRlc3QgUmVjYWxsIn0seyJ1dWlkIjoiMTcwNDEzNjU1NTM0MXg4dHg3ODhzIiwidHlwZSI6IkJBUiIsInJ1bnNDb3VudFRvQ29tcGFyZSI6MTAsIm1ldHJpY0tleSI6IlRlc3QgTG9zcyJ9LHsidXVpZCI6IjE3MDQxMzY1NTUzNDFidG40c3d1ZCIsInR5cGUiOiJCQVIiLCJydW5zQ291bnRUb0NvbXBhcmUiOjEwLCJtZXRyaWNLZXkiOiJUZXN0IFByZWNpc2lvbiJ9LHsidXVpZCI6IjE3MDQxMzY1NTUzNDFibTI4a2VtcSIsInR5cGUiOiJCQVIiLCJydW5zQ291bnRUb0NvbXBhcmUiOjEwLCJtZXRyaWNLZXkiOiJUZXN0IEYxIFNjb3JlIn0seyJ1dWlkIjoiMTcwNDEzNjU1NTM0MTlhb2g3eDd2IiwidHlwZSI6IlBBUkFMTEVMIiwicnVuc0NvdW50VG9Db21wYXJlIjoxMCwic2VsZWN0ZWRQYXJhbXMiOltdLCJzZWxlY3RlZE1ldHJpY3MiOltdfV0%3D)](https://dagshub.com/SunilKumar-ugra/BBC_News_Classification_LSTM.mlflow/)




# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/SunilKumar-ugra/BBC_News_Classification_LSTM.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n bbc-news python=3.8 -y
```

```bash
conda activate bbc-news
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
http://127.0.0.1:80 #Open this url in the browser
```

## Project Demo

[demo.webm](https://github.com/SunilKumar-ugra/BBC_News_Classification_LSTM/assets/45965583/716ff1e2-e3d0-4e6c-a243-d71d39390cad)
