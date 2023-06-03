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
| 表格  | 第一列  | 第二列  |
| --- |:----:|:----:|
|  age  |   age   |   age   |
| avg pg | avg pg  | avg pg  |
| avg pts | avg rebs  | avg ast  |
| pts of lastest 1 year | rebs of lastest 1 year  | ast of lastest 1 year  |
| pts of lastest 2 year | rebs of lastest 2 year  | ast of lastest 2 year  |
| avg net_rating | avg net_rating  | avg net_rating  |
| avg usg_pct | avg oreb_pct | avg usg_pct |
| avg ast_pct | avg dreb_pct | avg ast_pct |
| #th_season | season_no | season_no |


feature 1: age    
feature 2: avg pg  
feature 3: avg ast  
feature 4: ast of lastest 1 year  
feature 5: ast of lastest 2 year  
feature 6: avg net_rating  
feature 7: avg usg_pct  
feature 8: avg ast_pct  
feature 9: #th_season

### input feature of reb prediction
feature 1: age    
feature 2: avg pg  
feature 3: avg rebs  
feature 4: rebs of lastest 1 year  
feature 5: rebs of lastest 2 year  
feature 6: avg net_rating  
feature 7: avg oreb_pct  
feature 8: avg dreb_pct  
feature 9: season_no

### input feature of ast prediction
feature 1: age    
feature 2: avg pg  
feature 3: avg ast  
feature 4: ast of lastest 1 year  
feature 5: ast of lastest 2 year  
feature 6: avg net_rating  
feature 7: avg usg_pct  
feature 8: avg ast_pct  
feature 9: season_no
