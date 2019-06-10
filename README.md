# FACIAL EMOTION RECOGNITION IN THE WILD USING HOG, LBP FEATURES WITH SVM CLASSIFICATION
Chonnam National University, Gwangju, South Korea<br/>
Professor: **이 칠 우**<br/>
Student: **Tran Nguyen Quynh Tram**<br/>
Subject: **AI Theory**<br/>

## Requirements
1. Python 3 with libraries sklearn, skimage, re, pandas, matplotlib, numpy
2. RAF-DB Basic Dataset (http://www.whdeng.cn/RAF/model1.html) [2]
3. Jupyter notebook

## How to run source code
1. Install Python, Jupyter notebook and corresponding libraries
2. Download RAF-DB Basic Dataset and extract dataset into raf-db/base in data folder.
3. Run 01. rafdb_preprocessing.ipynb to create rafdb_basic.hdf5 indexing file.
4. Run 02. rafdb_explorer_data.ipynb to anlyze the dataset.
5. Run 03. rafdb_feature_extraction.ipynb to extract HOGs, LBPs features to data folder.
6. Run remaing jupyter notebooks to run SVM [5] classification on HOGs [3], LBPs [4] features for facial emotion recognition [1].

## RAF-DB Dataset information

## References
[1]  P. Ekman, “Facial expression and emotion.,”American psychologist, vol. 48, no. 4, p. 384, 1993.

[2]  S. Li and W. Deng, “Reliable crowdsourcing and deep locality-preserving learning for unconstrained facialexpression recognition,”IEEE Transactions on Image Processing, vol. 28, no. 1, pp. 356–370, 2019.

[3]  T. Surasak, I. Takahiro, C. H. Cheng, C. E. Wang, and P. Y. Sheng, “Histogram of oriented gradients forhuman detection in video,” inProceedings of 2018 5th International Conference on Business and IndustrialResearch: Smart Technology for Next Generation of Information, Engineering, Business and Social Science,ICBIR 2018, vol. 1, pp. 172–176, IEEE Computer Society, 2018.

[4]  T. Ojala, M. Pietik ̈ainen, and T. M ̈aenp ̈a ̈a, “Multiresolution gray-scale and rotation invariant texture classi-fication with local binary patterns,”IEEE Transactions on Pattern Analysis & Machine Intelligence, no. 7,pp. 971–987, 2002.

[5]  J. A. K. Suykens and J. Vandewalle, “Least squares support vector machine classifiers,”Neural processingletters, vol. 9, no. 3, pp. 293–300, 1999.
