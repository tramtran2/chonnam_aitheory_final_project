# FACIAL EMOTION RECOGNITION IN THE WILD USING HOG, LBP FEATURES WITH SVM CLASSIFICATION
Chonnam National University, Gwangju, South Korea<br/>
Professor: **이 칠 우**<br/>
Student: **Tran Nguyen Quynh Tram**<br/>
Subject: **AI Theory**<br/>

# Requirements
1. Python 3 with libraries sklearn, skimage, re, pandas, matplotlib, numpy
2. RAF-DB Basic Dataset (http://www.whdeng.cn/RAF/model1.html)
3. Jupyter notebook

# How to run source code
1. Install Python, Jupyter notebook and corresponding libraries
2. Download RAF-DB Basic Dataset and extract dataset into raf-db/base in data folder.
3. Run 01. rafdb_preprocessing.ipynb to create rafdb_basic.hdf5 indexing file.
4. Run 02. rafdb_explorer_data.ipynb to anlyze the dataset.
5. Run 03. rafdb_feature_extraction.ipynb to extract HOGs, LBPs features to data folder.
6. Run remaing jupyter notebooks to run classification on HOGs, LBPs features.

