import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def extract_data(cols, df, rename=None):
    raw_dat = df[cols].values.flatten().copy()
    dat = []
    for x in raw_dat:
        try:
            if not np.isnan(x):
                dat.append(x)
        except Exception:
            dat.append(x)
    if rename is not None:
        dat = [rename(x) for x in dat]
    return dat

def rename_ide(x):
    if x == 'Jupyter (JupyterLab, Jupyter Notebooks, etc) ':
        return 'Jupyter'
    if x == ' RStudio ':
        return 'RStudio'
    if x == 'Visual Studio':
        return 'Visual Studio'
    if x == 'Visual Studio Code (VSCode)':
        return 'VSCode'
    if x == ' PyCharm ':
        return 'PyCharm'
    if x == '  Spyder  ':
        return 'Spyder'
    if x == '  Notepad++  ':
        return 'Notepad++'
    if x == '  Sublime Text  ':
        return 'Sublime Text'
    if x == '  Vim / Emacs  ':
        return 'Vim / Emacs'
    if x == ' MATLAB ':
        return 'MATLAB'
    if x == 'None':
        return 'None'
    if x == 'Other':
        return 'Other'
    return 'nan'

def rename_hosted(x):
    if x==' Amazon EMR Notebooks ':
        return 'Amazon EMR Notebooks'
    if x==' Amazon Sagemaker Studio ':
        return 'Amazon Sagemaker'
    if x==' Binder / JupyterHub ':
        return 'Binder'
    if x==' Code Ocean ':
        return 'Code Ocean'
    if x==' Databricks Collaborative Notebooks ':
        return 'Databricks Notebooks'
    if x==' IBM Watson Studio ':
        return 'Watson Studio'
    if x==' Kaggle Notebooks':
        return 'Kaggle Notebooks'
    if x==' Paperspace / Gradient ':
        return 'Paperspace / Gradient'
    if x=='Google Cloud AI Platform Notebooks ':
        return 'gCloud AI Platform Notebooks '
    if x=='Google Cloud Datalab Notebooks':
        return 'gCloud Datalab Notebooks'
    return x

def rename_computing(x):
    if x=='A cloud computing platform (AWS, Azure, GCP, hosted notebooks, etc)':
        return 'Cloud computing'
    if x=='A deep learning workstation (NVIDIA GTX, LambdaLabs, etc)':
        return 'DL workstation'
    if x=='A personal computer or laptop':
        return 'PC'
    return x

def rename_mlalgos(x):
    if x=='Convolutional Neural Networks':
        return 'CNN'
    if x=='Decision Trees or Random Forests':
        return 'CNN'
    if x=='Dense Neural Networks (MLPs, etc)':
        return 'MLP'
    if x=='Generative Adversarial Networks':
        return 'GAN'
    if x=='Gradient Boosting Machines (xgboost, lightgbm, etc)':
        return 'GBM'
    if x=='Recurrent Neural Networks':
        return 'RNN'
    if x=='Linear or Logistic Regression':
        return 'Linear/Logistic Regression'
    if x=='Transformer Networks (BERT, gpt-3, etc)':
        return 'Transformers'
    return x

def rename_cv(x):
    if x=='General purpose image/video tools (PIL, cv2, skimage, etc)':
        return 'General purpose tools'
    if x=='Generative Networks (GAN, VAE, etc)':
        return 'Generative Networks'
    if x=='Image classification and other general purpose networks (VGG, Inception, ResNet, ResNeXt, NASNet, EfficientNet, etc)':
        return 'Image classification'
    if x=='Image segmentation methods (U-Net, Mask R-CNN, etc)':
        return 'Image segmentation'
    if x=='Object detection methods (YOLOv3, RetinaNet, etc)':
        return 'Object detection'
    return x

def rename_nlp(x):
    if x=='Contextualized embeddings (ELMo, CoVe)':
        return 'Contextualized embeddings'
    if x=='Encoder-decorder models (seq2seq, vanilla transformers)':
        return 'Encoder-decorder models'
    if x=='Transformer language models (GPT-3, BERT, XLnet, etc)':
        return 'Transformers'
    if x=='Word embeddings/vectors (GLoVe, fastText, word2vec)':
        return 'Word embeddings/vectors'
    return x

def rename_cloudplat(x):
    if x==' Amazon Web Services (AWS) ':
        return 'AWS'
    if x==' Google Cloud Platform (GCP) ':
        return 'GCP'
    return x

def rename_mlprod(x):
    if x==' Azure Machine Learning Studio ':
        return ' Azure ML Studio '
    if x==' Google Cloud AI Platform / Google Cloud ML Engine':
        return 'gCloud AI Platform'
    if x==' Google Cloud Natural Language ':
        return 'gCloud NLP'
    if x==' Google Cloud Video AI ':
        return 'gCloud Video'
    if x==' Google Cloud Vision AI ':
        return 'gCloud Vision'
    return x

def rename_bigdata(x):
    if 'Google Cloud' in x:
        return 'gCloud '+x.split('Google Cloud')[-1]
    if x=='Microsoft Azure Data Lake Storage ':
        return 'Azure Data Lake'
    if x=='Microsoft SQL Server ':
        return 'Microsoft SQL'
    return x

def rename_bi(x):
    if x=='Einstein Analytics':
        return 'Einstein Anal.'
    if x=='Google Data Studio':
        return 'Google Data'
    if x=='Microsoft Power BI':
        return 'Microsoft BI'
    if x=='SAP Analytics Cloud ':
        return 'SAP Anal.'
    return x

def rename_automl(x):
    if x=='Automated data augmentation (e.g. imgaug, albumentations)':
        return 'Auto. data augmentation'
    if x=='Automated feature engineering/selection (e.g. tpot, boruta_py)':
        return 'Auto. feature engineering/selection'
    if x=='Automated hyperparameter tuning (e.g. hyperopt, ray.tune, Vizier)':
        return 'Auto. hyperparameter tuning'
    if x=='Automated model architecture searches (e.g. darts, enas)':
        return 'Auto. model architecture searches'
    if x=='Automated model selection (e.g. auto-sklearn, xcessiv)':
        return 'Auto. model selection'
    if x=='Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)':
        return 'Auto. full ML pipelines'
    return x

def rename_manage(x):
    if x=='Domino Model Monitor':
        return 'Domino Monitor'
    return x

def rename_share(x):
    if x=='I do not share my work publicly':
        return 'Not share'
    return x

def rename_courses(x):
    if x=='Cloud-certification programs (direct from AWS, Azure, GCP, or similar)':
        return 'Cloud-certification'
    if x=='University Courses (resulting in a university degree)':
        return 'University Degree'
    return x

def rename_socialplatform(x):
    platforms = [
       'Blogs (Towards Data Science, Analytics Vidhya, etc)',
       'Course Forums (forums.fast.ai, Coursera forums, etc)',
       "Email newsletters (Data Elixir, O'Reilly Data & AI, etc)",
       'Journal Publications (peer-reviewed journals, conference proceedings, etc)',
       'Kaggle (notebooks, forums, etc)', 'None', 'Other',
       'Podcasts (Chai Time Data Science, O’Reilly Data Show, etc)',
       'Reddit (r/machinelearning, etc)',
       'Slack Communities (ods.ai, kagglenoobs, etc)',
       'Twitter (data science influencers)',
       'YouTube (Kaggle YouTube, Cloud AI Adventures, etc)'
    ]

    for p in platforms:
        if x==p:
            return x.split('(')[0]

def process_dataframe(df):
    res = dict()
    res['prog_lang'] = extract_data(cols_programming, df)
    res['ide'] = extract_data(cols_ide, df, rename_ide)
    res['hosted'] = extract_data(cols_hosted_notebook, df, rename_hosted)
    res['computing'] = extract_data(['What type of computing platform do you use most often for your data science projects? - Selected Choice'], df, rename_computing)
    res['hardware'] = extract_data(cols_hardware, df)
    res['times_tpu'] = extract_data(['Approximately how many times have you used a TPU (tensor processing unit)?'], df)
    res['vis'] = extract_data(cols_vis, df)
    res['frameworks'] = extract_data(cols_frameworks, df)
    res['mlalgos'] = extract_data(cols_mlalgos, df, rename_mlalgos)
    res['cv'] = extract_data(cols_cv, df, rename_cv)
    res['nlp'] = extract_data(cols_nlp, df, rename_nlp)
    res['cloud_platform'] = extract_data(cols_cloud_platform, df, rename_cloudplat)
    res['ml_product'] = extract_data(cols_ml_product, df, rename_mlprod)
    res['bigdata'] = extract_data(cols_bigdata, df, rename_bigdata)
    res['bi'] = extract_data(cols_bi, df, rename_bi)
    res['automl'] = extract_data(cols_automl, df, rename_automl)
    res['manage_ml'] = extract_data(cols_manage_ml, df, rename_manage)
    res['share'] = extract_data(cols_share, df, rename_share)
    res['courses'] = extract_data(cols_courses, df, rename_courses)
    res['socialplatform'] = extract_data(cols_socialplatform, df, rename_socialplatform)
    
    return res

def map_team_size(x):
    if x=='0':
        return 0
    if x=='1-2':
        return 1.5
    if x=='3-4':
        return 3.5
    if x=='5-9':
        return 7
    if x=='10-14':
        return 12
    if x=='15-19':
        return 17
    if x=='20+':
        return 20
    return np.nan

def map_company_size(x):
    if '0-49 employees':
        return 25
    if '50-249 employees':
        return 150
    if '250-999 employees':
        return 625
    if '1000-9,999 employees':
        return 5500
    if '10,000 or more employees':
        return 10000
    return np.nan

def map_yrs_ml(x):
    if x=='I do not use machine learning methods':
        return 0
    if x=='Under 1 year':
        return 0.5
    if x=='1-2 years':
        return 1.5
    if x=='2-3 years':
        return 2.5
    if x=='3-4 years':
        return 3.5
    if x=='4-5 years':
        return 4.5
    if x=='5-10 years':
        return 7.5
    if x=='10-20 years':
        return 15
    if x=='20 or more years':
        return 20
    return np.nan

def map_yrs_prog(x):
    if x=='I have never written code':
        return 0
    if x=='< 1 years':
        return 0.5
    if x=='1-2 years':
        return 1.5
    if x=='3-5 years':
        return 4
    if x=='5-10 years':
        return 7.5
    if x=='10-20 years':
        return 15
    if x=='20+ years':
        return 20
    return np.nan

def generate_result(k, res_neophytes, res_expert, experience):
    labels = np.unique(res_neophytes[k]+res_expert[k])
    df = pd.DataFrame(np.zeros((labels.shape[0], 2)), columns=['neophytes', 'experts'], index=labels)
    for lang, counts in zip(*np.unique(res_neophytes[k], return_counts=True)):
        df.loc[lang, 'neophytes'] = counts/np.sum(experience['lvl'].values==0)
    for lang, counts in zip(*np.unique(res_expert[k], return_counts=True)):
        df.loc[lang, 'experts'] = counts/np.sum(experience['lvl'].values==1)
    fig, ax = plt.subplots(figsize=(20,8))
    plt.title(k)
    df.plot.bar(ax=ax, rot=0)
    plt.tight_layout()
    plt.show()

def generate_result_difference(k, res_neophytes, res_expert, experience):
    labels = np.unique(res_neophytes[k]+res_expert[k])
    df = pd.DataFrame(np.zeros((labels.shape[0], 2)), columns=['neophytes', 'experts'], index=labels)
    for lang, counts in zip(*np.unique(res_neophytes[k], return_counts=True)):
        df.loc[lang, 'neophytes'] = counts/np.sum(experience['lvl'].values==0)
    for lang, counts in zip(*np.unique(res_expert[k], return_counts=True)):
        df.loc[lang, 'experts'] = counts/np.sum(experience['lvl'].values==1)
    
    df['diff'] = df['experts']-df['neophytes']
    _, ax = plt.subplots(figsize=(15,8))
    #plt.title(k)
    colors = ['r' if x > 0 else 'b' for x in df['diff']]
    df['diff'].plot.barh(ax=ax, rot=0, color=colors)
    plt.tight_layout()
    plt.xlim(-0.3, 0.3)
    plt.savefig('./results/'+k+'.png')
    plt.close()




cols_programming = [
    'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Python',
    'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - R',
    'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - SQL',
    'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - C',
    'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - C++',
    'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Java',
    'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Javascript',
    'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Julia',
    'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Swift',
    'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Bash',
    'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - MATLAB',
    'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - None',
    'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Other',]
cols_ide = [
     "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice - Jupyter (JupyterLab, Jupyter Notebooks, etc) ",
    "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  RStudio ",
    "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Visual Studio / Visual Studio Code ",
    "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice - Click to write Choice 13",
    "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  PyCharm ",
    "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Spyder  ",
    "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Notepad++  ",
    "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Sublime Text  ",
    "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Vim / Emacs  ",
    "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  MATLAB ",
    "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice - None",
    "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice - Other",
]
cols_hosted_notebook = [
    'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Kaggle Notebooks',
    'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Colab Notebooks',
    'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Azure Notebooks',
    'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Paperspace / Gradient ',
    'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Binder / JupyterHub ',
    'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Code Ocean ',
    'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  IBM Watson Studio ',
    'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Amazon Sagemaker Studio ',
    'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Amazon EMR Notebooks ',
    'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Google Cloud AI Platform Notebooks ',
    'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Google Cloud Datalab Notebooks',
    'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Databricks Collaborative Notebooks ',
    'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - None',
    'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Other'
]
cols_hardware = [
    'Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - GPUs',
    'Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - TPUs',
    'Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - None',
    'Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - Other'
]
cols_vis = [
    'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Matplotlib ',
    'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Seaborn ',
    'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Plotly / Plotly Express ',
    'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Ggplot / ggplot2 ',
    'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Shiny ',
    'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  D3 js ',
    'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Altair ',
    'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Bokeh ',
    'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Geoplotlib ',
    'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Leaflet / Folium ',
    'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice - None',
    'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Other'
]
cols_frameworks = [
    'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -   Scikit-learn ',
    'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -   TensorFlow ',
    'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Keras ',
    'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  PyTorch ',
    'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Fast.ai ',
    'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  MXNet ',
    'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Xgboost ',
    'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  LightGBM ',
    'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  CatBoost ',
    'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Prophet ',
    'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  H2O 3 ',
    'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Caret ',
    'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Tidymodels ',
    'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  JAX ',
    'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice - None',
    'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice - Other'
]
cols_mlalgos = [
    'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Linear or Logistic Regression',
    'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Decision Trees or Random Forests',
    'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Gradient Boosting Machines (xgboost, lightgbm, etc)',
    'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Bayesian Approaches',
    'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Evolutionary Approaches',
    'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Dense Neural Networks (MLPs, etc)',
    'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Convolutional Neural Networks',
    'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Generative Adversarial Networks',
    'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Recurrent Neural Networks',
    'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Transformer Networks (BERT, gpt-3, etc)',
    'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - None',
    'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Other'
]
cols_cv = [
    'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - General purpose image/video tools (PIL, cv2, skimage, etc)',
    'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Image segmentation methods (U-Net, Mask R-CNN, etc)',
    'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Object detection methods (YOLOv3, RetinaNet, etc)',
    'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Image classification and other general purpose networks (VGG, Inception, ResNet, ResNeXt, NASNet, EfficientNet, etc)',
    'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Generative Networks (GAN, VAE, etc)',
    'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - None',
    'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Other'
]
cols_nlp = [
    'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Word embeddings/vectors (GLoVe, fastText, word2vec)',
    'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Encoder-decorder models (seq2seq, vanilla transformers)',
    'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Contextualized embeddings (ELMo, CoVe)',
    'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Transformer language models (GPT-3, BERT, XLnet, etc)',
    'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - None',
    'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Other'
]
cols_company = [
    'What is the size of the company where you are employed?',
    'Approximately how many individuals are responsible for data science workloads at your place of business?',
    'Does your current employer incorporate machine learning methods into their business?'
]
cols_roles = [
    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Analyze and understand data to influence product or business decisions',
    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',
    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build prototypes to explore applying machine learning to new areas',
    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run a machine learning service that operationally improves my product or workflows',
    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Experimentation and iteration to improve existing ML models',
    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Do research that advances the state of the art of machine learning',
    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - None of these activities are an important part of my role at work',
    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Other'
]

cols_cloud_platform = [
    'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Amazon Web Services (AWS) ',
    'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Microsoft Azure ',
    'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Platform (GCP) ',
    'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  IBM Cloud / Red Hat ',
    'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Oracle Cloud ',
    'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  SAP Cloud ',
    'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Salesforce Cloud ',
    'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  VMware Cloud ',
    'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Alibaba Cloud ',
    'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Tencent Cloud ',
    'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice - None',
    'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice - Other'
]

cols_cloud_product = [
    'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Amazon EC2 ',
    'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  AWS Lambda ',
    'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Amazon Elastic Container Service ',
    'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Azure Cloud Services ',
    'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Microsoft Azure Container Instances ',
    'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Azure Functions ',
    'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Compute Engine ',
    'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Functions ',
    'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Run ',
    'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud App Engine ',
    'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice - No / None',
    'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice - Other'
]

cols_ml_product = [
    'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Amazon SageMaker ',
    'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Amazon Forecast ',
    'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Amazon Rekognition ',
    'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Azure Machine Learning Studio ',
    'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Azure Cognitive Services ',
    'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud AI Platform / Google Cloud ML Engine',
    'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Video AI ',
    'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Natural Language ',
    'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Vision AI ',
    'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice - No / None',
    'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice - Other'
]

cols_bigdata = [
    'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - MySQL ',
    'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - PostgresSQL ',
    'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - SQLite ',
    'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Oracle Database ',
    'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - MongoDB ',
    'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Snowflake ',
    'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - IBM Db2 ',
    'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft SQL Server ',
    'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft Access ',
    'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft Azure Data Lake Storage ',
    'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Amazon Redshift ',
    'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Amazon Athena ',
    'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Amazon DynamoDB ',
    'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Google Cloud BigQuery ',
    'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Google Cloud SQL ',
    'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Google Cloud Firestore ',
    'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - None',
    'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Other',
]

cols_bi = [
    'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Amazon QuickSight',
    'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft Power BI',
    'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Google Data Studio',
    'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Looker',
    'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Tableau',
    'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Salesforce',
    'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Einstein Analytics',
    'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Qlik',
    'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Domo',
    'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - TIBCO Spotfire',
    'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Alteryx ',
    'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Sisense ',
    'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - SAP Analytics Cloud ',
    'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - None',
    'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Other']

cols_automl = [
    'Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automated data augmentation (e.g. imgaug, albumentations)',
    'Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automated feature engineering/selection (e.g. tpot, boruta_py)',
    'Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automated model selection (e.g. auto-sklearn, xcessiv)',
    'Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automated model architecture searches (e.g. darts, enas)',
    'Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automated hyperparameter tuning (e.g. hyperopt, ray.tune, Vizier)',
    'Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)',
    'Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - No / None',
    'Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Other',
    'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Google Cloud AutoML ',
    'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  H20 Driverless AI  ',
    'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Databricks AutoML ',
    'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  DataRobot AutoML ',
    'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Tpot ',
    'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Auto-Keras ',
    'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Auto-Sklearn ',
    'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Auto_ml ',
    'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Xcessiv ',
    'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   MLbox ',
    'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice - No / None',
    'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice - Other'
]

cols_manage_ml = [
    'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Neptune.ai ',
    'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Weights & Biases ',
    'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Comet.ml ',
    'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Sacred + Omniboard ',
    'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  TensorBoard ',
    'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Guild.ai ',
    'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Polyaxon ',
    'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Trains ',
    'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Domino Model Monitor ',
    'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice - No / None',
    'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice - Other'
]

cols_share = [
    'Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  Plotly Dash ',
    'Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  Streamlit ',
    'Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  NBViewer ',
    'Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  GitHub ',
    'Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  Personal blog ',
    'Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  Kaggle ',
    'Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  Colab ',
    'Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  Shiny ',
    'Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice - I do not share my work publicly',
    'Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice - Other'
]

cols_courses = [
    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Coursera',
    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - edX',
    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Kaggle Learn Courses',
    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - DataCamp',
    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Fast.ai',
    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udacity',
    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udemy',
    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - LinkedIn Learning',
    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Cloud-certification programs (direct from AWS, Azure, GCP, or similar)',
    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - University Courses (resulting in a university degree)',
    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - None',
    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Other'
]

cols_socialplatform = [
    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Twitter (data science influencers)',
    "Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Email newsletters (Data Elixir, O'Reilly Data & AI, etc)",
    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Reddit (r/machinelearning, etc)',
    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Kaggle (notebooks, forums, etc)',
    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Course Forums (forums.fast.ai, Coursera forums, etc)',
    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - YouTube (Kaggle YouTube, Cloud AI Adventures, etc)',
    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Podcasts (Chai Time Data Science, O’Reilly Data Show, etc)',
    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Blogs (Towards Data Science, Analytics Vidhya, etc)',
    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Journal Publications (peer-reviewed journals, conference proceedings, etc)',
    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Slack Communities (ods.ai, kagglenoobs, etc)',
    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - None',
    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Other'
]