# gender_stereotypes_NYT
This is the repository with the code for my final year thesis. I use facial recognition API and high-dimensional data to analyse gendered language of NYT articles and uncover nonverbal gender stereotypes.

## Abstract
Over the past decades, women have achieved progress in labor force participation, breaking the glass ceiling, and narrowing the gender wage gap. And even though women constitute approximately 50 percent of the world population, they are still underrepresented in many mathematically-intensive occupations and decision-making positions. Since media played an important role in shaping people's beliefs and creating unconscious biases, I examine whether women are portrayed in news media in accordance with traditional stereotypes and gender roles or not. The methodological idea is to test the association between textual and visual data of the major US newspaper with the help of machine learning tools, text analysis, and econometric models. This is the first attempt to combine both high-dimensional text analysis and large-sample visual analysis. I document a negative association between the tendency of an article being professional and the share of women depicted on article images. In particular, women are less likely to appear on images of science, politics, and economics articles and more likely in appearance, fashion, and family articles. This result sheds more light on factors influencing educational and the occupational choices of women.

## Description of files:
* data_collection.py\
The code for collecting the raw data from the New York Times Developer Network, which provides a convenient NYT API for free non-commercial uses. In particular, I access Archive API returning JSON file with all NYT article metadata for given month and year. The code also provides a solution for collecting full texts of each article via the automated web scraper.\
* face_detection.py\
The code for dealing with Microsoft Azure which provides the artificial intelligence service that identifies faces in images and, for each face, detects attributes such as gender. This method has several limitations: image resolution, size and angle of faces.
* text_analysis.py\
The code for dealing with high-dimensional data (digital texts), causal inference and descriptive analysis. First, it includes the code for text preprocessing (tokenization, lemmatization, etc.). Second, it creates a bag-of-words and estimates lasso logistic regression to solve the classification and prediction problems.
