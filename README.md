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

## Feature Engineering
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

*Correlation of each feature*
![image](https://github.com/RayJiazy/NBAplayer_Performance_Pred/blob/main/images/1685768354627.jpg)

## Results
*RMSE/MAE/R-squared evaluation of different models*
<table>
    <tr>
        <th></th><th colspan="3">RMSE</th><th colspan="3">MAE</th><th colspan="3">R-squared</th>
    </tr>
    <tr>
        <td></td><td>pts</td><td>reb</td><td>ast</td><td>pts</td><td>reb</td><td>ast</td><td>pts</td><td>reb</td><td>ast</td>
    </tr>
    <tr>
        <td>Trivial system</td><td>6.936</td><td>2.500</td><td>1.851</td><td>4.835</td><td>1.803</td><td>1.503</td><td>\</td><td>\</td><td>\</td>
    </tr>
    <tr>
        <td>Linear Regression(baseline)</td><td>2.948</td><td>1.168</td><td>0.812</td><td>2.145</td><td>0.837</td><td>0.549</td><td>0.540</td><td>0.532</td><td>0.556</td>
    </tr>
    <tr>
        <td>Ridge Regression</td><td>2.946</td><td>1.164</td><td>0.809</td><td>2.144</td><td>0.837</td><td>0.549</td><td>0.538</td><td>0.534</td><td>0.563</td>
    </tr>
    <tr>
        <td>Decision Tree</td><td>3.183</td><td>1.208</td><td>0.823</td><td>2.412</td><td>0.893</td><td>0.590</td><td>0.502</td><td>0.517</td><td>0.555</td>
    </tr>
    <tr>
        <td>Random Forest</td><td>2.978</td><td>1.178</td><td>0.797</td><td>2.268</td><td>0.840</td><td>0.568</td><td>0.534</td><td>0.529</td><td>0.570</td>
    </tr>
    <tr>
        <td>SVR</td><th>2.830</th><th>1.159</th><td>0.790</td><th>2.127</th><th>0.840</th><td>0.541</td><th>0.558</th><th>0.537</th><td>0.573</td>
    </tr>
    <tr>
        <td>MLP-4layers</td><td>2.864</td><td>1.161</td><th>0.782</th><td>2.153</td><td>0.837</td><th>0.538</th><td>0.552</td><td>0.536</td><th>0.578</th>
    </tr>
</table>

*Compare RMSE loss with augmented and non-augmented dataset*

<table>
    <tr>
        <th></th><th colspan="3">Origin dataset</th><th colspan="3">Augmented dataset</th>
    </tr>
    <tr>
        <td></td><td>pts</td><td>reb</td><td>ast</td><td>pts</td><td>reb</td><td>ast</td>
    </tr>
    <tr>
        <td>Linear Regression(baseline)</td><td>3.246</td><td>1.322</td><td>0.959</td><td>2.948</td><td>1.168</td><td>0.812</td>
    </tr>
    <tr>
        <td>Ridge Regression</td><td>3.246</td><td>1.322</td><th>0.959</th><td>2.946</td><td>1.164</td><td>0.809</td>
    </tr>
    <tr>
        <td>Decision Tree</td><td>3.266</td><td>1.405</td><td>1.032</td><td>3.183</td><td>1.208</td><td>0.823</td>
    </tr>
    <tr>
        <td>Random Forest</td><td>3.127</td><td>1.332</td><td>0.988</td><td>2.978</td><td>1.178</td><td>0.797</td>
    </tr>
    <tr>
        <td>SVR</td><th>3.106</th><th>1.297</th><td>0.971</td><th>2.830</th><th>1.159</th><td>0.791</td>
    </tr>
    <tr>
        <td>MLP-4layers</td><td>3.189</td><td>1.359</td><td>0.968</td><td>2.864</td><td>1.161</td><th>0.782</th>
    </tr>
</table>

## Instruction
## Environment
* Pytorch
* Pandas
* Sklearn
* Numpy

## References
[1] "NBA Players," [Online]. Available: [https://www.kaggle.com/datasets/justinas/nba-players-data](https://www.kaggle.com/datasets/justinas/nba-players-data)

[2] "College Students' NBA Performance Analysis," [Online]. Available: [https://www.kaggle.com/datasets/justinas/nba-players-data](https://www.kaggle.com/datasets/justinas/nba-players-data)

[3]	“Decision Trees in Machine Learning,” [Online].Available: [https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052](https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052)

[4]	Vangelis Sarlis a, Vasilis Chatziilias b, Christos Tjortjis a, Dimitris Mandalidis b, A Data Science approach analysing the Impact of Injuries on Basketball Player and Team Performance, Information Systems, Volume 99, July 2021, 101750

