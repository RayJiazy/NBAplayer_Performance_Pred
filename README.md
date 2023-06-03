# NBAplayer_Performance_Pred
Predict players’ performance (e.g. pts, rebs, and ast (average number of points,  rebounds, and assists)) this season based on the statistics in previous seasons.

## The goal of this project
1.Data preprocessing and feature engineering: Clean and prepare the dataset for analysis, identify 
relevant features, and potentially create new ones to improve model performance.

2.Exploratory Data Analysis (EDA): Analyze the dataset to gain insights into player performance 
trends and relationships between various features.

3.Model selection and training: Select appropriate machine learning algorithms, split the data into 
training and testing sets, and train the models to make predictions.

4.Model Evaluation and Comparison: Evaluate the performance of the trained models using 
appropriate metrics and compare their accuracy to determine the most suitable model for predicting 
players’ performance. This step also involves fine-tuning the models to optimize their accuracy, 
as needed.

By achieving these goals, we hope to provide a reliable tool for predicting NBA player 
performance, which can be used to inform team strategies and personnel decisions.

## Data augmentation
We standardized the data and performed data augmentation on the original dataset. 
Specifically, for players with N historical data points (N > n > 3), we split their original dataset 
into N-n+1 subsets, each containing n data points, where n is an optimizable hyperparameter. The 
starting index of each subset was within the range [0, N-n] from the original dataset. The results 
showed a significant improvement in the model's performance on the augmented dataset.

### Feature Engineering
| feature of pts prediction  | feature of reb prediction  | feature of ast prediction |
|:----:|:----:|:----:|
|  age  |   age   |   age   |
| avg pg | avg pg  | avg pg  |
| avg pts | avg rebs  | avg ast  |
| pts of lastest 1 year | rebs of lastest 1 year  | ast of lastest 1 year  |
| pts of lastest 2 year | rebs of lastest 2 year  | ast of lastest 2 year  |
| avg net_rating | avg net_rating  | avg net_rating  |
| avg usg_pct | avg oreb_pct | avg usg_pct |
| avg ast_pct | avg dreb_pct | avg ast_pct |
| season_no | season_no | season_no |

Correlation of each feature

![image](https://github.com/RayJiazy/NBAplayer_Performance_Pred/blob/main/images/1685768354627.jpg)
