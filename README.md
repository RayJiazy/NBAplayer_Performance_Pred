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
| 第一行 | 第一列  | 第二列  |
| 第一行 | 第一列  | 第二列  |
| 第一行 | 第一列  | 第二列  |
| 第一行 | 第一列  | 第二列  |
| 第一行 | 第一列  | 第二列  |
| 第一行 | 第一列  | 第二列  |
| 第一行 | 第一列  | 第二列  |
| 第一行 | 第一列  | 第二列  |
<table class="MsoTableGrid" border="1" cellspacing="0" cellpadding="0" width="601" style="width:451.05pt;border-collapse:collapse;border:none;mso-border-top-alt:
 solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;mso-yfti-tbllook:
 1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt;mso-border-insidev:none">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes;page-break-inside:avoid;
  height:1.0cm">
  <td width="94" style="width:70.85pt;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="ZH-CN" style="mso-bidi-font-size:10.5pt;line-height:106%;color:black;mso-themecolor:
  text1">sub-model<o:p></o:p></span></p>
  </td>
  <td width="205" style="width:153.6pt;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="ZH-CN" style="mso-bidi-font-size:10.5pt;line-height:106%;color:black;mso-themecolor:
  text1">pts Prediction<o:p></o:p></span></p>
  </td>
  <td width="151" style="width:113.3pt;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="ZH-CN" style="mso-bidi-font-size:10.5pt;line-height:106%;color:black;mso-themecolor:
  text1">reb Prediction<o:p></o:p></span></p>
  </td>
  <td width="151" style="width:113.3pt;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0cm">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="ZH-CN" style="mso-bidi-font-size:10.5pt;line-height:106%;color:black;mso-themecolor:
  text1">ast Prediction<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:1;page-break-inside:avoid;height:14.2pt;mso-height-rule:
  exactly">
  <td width="94" valign="top" style="width:70.85pt;border:none;mso-border-top-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt;mso-height-rule:
  exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">Feature 1</span><span lang="ZH-CN" style="font-size:10.0pt;
  font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black"><o:p></o:p></span></p>
  </td>
  <td width="205" valign="top" style="width:153.6pt;border:none;mso-border-top-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt;mso-height-rule:
  exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">age</span><span lang="EN-US" style="font-size:10.0pt;
  font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
  <td width="151" valign="top" style="width:113.3pt;border:none;mso-border-top-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt;mso-height-rule:
  exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">age</span><span lang="EN-US" style="font-size:10.0pt;
  font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
  <td width="151" valign="top" style="width:113.3pt;border:none;mso-border-top-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt;mso-height-rule:
  exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">age</span><span lang="ZH-CN" style="font-size:10.0pt;
  font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2;page-break-inside:avoid;height:14.2pt;mso-height-rule:
  exactly">
  <td width="94" valign="top" style="width:70.85pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">Feature 2</span><span lang="EN-US" style="font-size:10.0pt;
  font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
  <td width="205" valign="top" style="width:153.6pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">avg pg</span><span lang="EN-US" style="font-size:10.0pt;
  font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
  <td width="151" valign="top" style="width:113.3pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">avg pg</span><span lang="EN-US" style="font-size:10.0pt;
  font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
  <td width="151" valign="top" style="width:113.3pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">avg pg</span><span lang="EN-US" style="font-size:10.0pt;
  font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:3;page-break-inside:avoid;height:14.2pt;mso-height-rule:
  exactly">
  <td width="94" valign="top" style="width:70.85pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">Feature 3</span><span lang="EN-US" style="font-size:10.0pt;
  font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
  <td width="205" valign="top" style="width:153.6pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">avg pts</span><span lang="EN-US" style="font-size:10.0pt;
  font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
  <td width="151" valign="top" style="width:113.3pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">avg reb</span><span lang="EN-US" style="font-size:10.0pt;
  font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
  <td width="151" valign="top" style="width:113.3pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">avg ast</span><span lang="EN-US" style="font-size:10.0pt;
  font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:4;page-break-inside:avoid;height:14.2pt;mso-height-rule:
  exactly">
  <td width="94" valign="top" style="width:70.85pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">Feature 4<o:p></o:p></span></p>
  </td>
  <td width="205" valign="top" style="width:153.6pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="EN-US" style="font-size:9.0pt;color:black;
  mso-themecolor:text1;mso-ansi-language:EN-US">pts of the last season</span><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman Regular&quot;;
  mso-bidi-font-family:&quot;Times New Roman Regular&quot;;color:black;mso-ansi-language:
  EN-US"><o:p></o:p></span></p>
  </td>
  <td width="151" valign="top" style="width:113.3pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="left" style="margin-left:0cm;text-align:left;
  line-height:normal"><span class="SpellE"><span lang="EN-US" style="font-size:
  9.0pt;color:black;mso-themecolor:text1;mso-ansi-language:EN-US">reb</span></span><span lang="EN-US" style="font-size:9.0pt;color:black;mso-themecolor:text1;
  mso-ansi-language:EN-US"> of the last season</span><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:
  &quot;Times New Roman Regular&quot;;color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
  <td width="151" valign="top" style="width:113.3pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span class="SpellE"><span lang="EN-US" style="font-size:
  9.0pt;color:black;mso-themecolor:text1;mso-ansi-language:EN-US">ast</span></span><span lang="EN-US" style="font-size:9.0pt;color:black;mso-themecolor:text1;
  mso-ansi-language:EN-US"> of the last season</span><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:
  &quot;Times New Roman Regular&quot;;color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:5;page-break-inside:avoid;height:14.2pt;mso-height-rule:
  exactly">
  <td width="94" valign="top" style="width:70.85pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">Feature 5<o:p></o:p></span></p>
  </td>
  <td width="205" valign="top" style="width:153.6pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="EN-US" style="font-size:9.0pt;color:black;
  mso-themecolor:text1;mso-ansi-language:EN-US">pts of the last <span class="SpellE">last</span> season</span><span lang="EN-US" style="font-size:10.0pt;
  font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
  <td width="151" valign="top" style="width:113.3pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span class="SpellE"><span lang="EN-US" style="font-size:
  9.0pt;color:black;mso-themecolor:text1;mso-ansi-language:EN-US">reb</span></span><span lang="EN-US" style="font-size:9.0pt;color:black;mso-themecolor:text1;
  mso-ansi-language:EN-US"> of the last <span class="SpellE">last</span> season</span><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman Regular&quot;;
  mso-bidi-font-family:&quot;Times New Roman Regular&quot;;color:black;mso-ansi-language:
  EN-US"><o:p></o:p></span></p>
  </td>
  <td width="151" valign="top" style="width:113.3pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span class="SpellE"><span lang="EN-US" style="font-size:
  9.0pt;color:black;mso-themecolor:text1;mso-ansi-language:EN-US">ast</span></span><span lang="EN-US" style="font-size:9.0pt;color:black;mso-themecolor:text1;
  mso-ansi-language:EN-US"> of the last <span class="SpellE">last</span> season</span><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman Regular&quot;;
  mso-bidi-font-family:&quot;Times New Roman Regular&quot;;color:black;mso-ansi-language:
  EN-US"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:6;page-break-inside:avoid;height:14.2pt;mso-height-rule:
  exactly">
  <td width="94" valign="top" style="width:70.85pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">Feature 6<o:p></o:p></span></p>
  </td>
  <td width="205" valign="top" style="width:153.6pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">avg net_rating</span><span lang="EN-US" style="font-size:
  10.0pt;font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
  <td width="151" valign="top" style="width:113.3pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">avg net_rating</span><span lang="ZH-CN" style="font-size:
  10.0pt;font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black"><o:p></o:p></span></p>
  </td>
  <td width="151" valign="top" style="width:113.3pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">avg net_rating</span><span lang="EN-US" style="font-size:
  10.0pt;font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:7;page-break-inside:avoid;height:14.2pt;mso-height-rule:
  exactly">
  <td width="94" valign="top" style="width:70.85pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">Feature 7<o:p></o:p></span></p>
  </td>
  <td width="205" valign="top" style="width:153.6pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">avg ts_pct</span><span lang="EN-US" style="font-size:
  10.0pt;font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
  <td width="151" valign="top" style="width:113.3pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">avg oreb_pct</span><span lang="EN-US" style="font-size:
  10.0pt;font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
  <td width="151" valign="top" style="width:113.3pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">avg usg_pct</span><span lang="EN-US" style="font-size:
  10.0pt;font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:8;page-break-inside:avoid;height:14.2pt;mso-height-rule:
  exactly">
  <td width="94" valign="top" style="width:70.85pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">Feature 8<o:p></o:p></span></p>
  </td>
  <td width="205" valign="top" style="width:153.6pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1"><span style="mso-spacerun:yes">&nbsp;</span>avg usg_pct</span><span lang="EN-US" style="font-size:10.0pt;font-family:&quot;Times New Roman Regular&quot;;
  mso-bidi-font-family:&quot;Times New Roman Regular&quot;;color:black;mso-ansi-language:
  EN-US"><o:p></o:p></span></p>
  </td>
  <td width="151" valign="top" style="width:113.3pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">avg dreb_pct</span><span lang="EN-US" style="font-size:
  10.0pt;font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
  <td width="151" valign="top" style="width:113.3pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">avg ast_pct</span><span lang="EN-US" style="font-size:
  10.0pt;font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:9;mso-yfti-lastrow:yes;page-break-inside:avoid;
  height:14.2pt;mso-height-rule:exactly">
  <td width="94" valign="top" style="width:70.85pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">Feature 9<o:p></o:p></span></p>
  </td>
  <td width="205" valign="top" style="width:153.6pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">season_no</span><span lang="EN-US" style="font-size:10.0pt;
  font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
  <td width="151" valign="top" style="width:113.3pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">season_no</span><span lang="EN-US" style="font-size:10.0pt;
  font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
  <td width="151" valign="top" style="width:113.3pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt;mso-height-rule:exactly">
  <p class="BodyText1" align="center" style="margin-left:0cm;text-align:center;
  line-height:normal"><span lang="ZH-CN" style="font-size:9.0pt;color:black;
  mso-themecolor:text1">season_no</span><span lang="EN-US" style="font-size:10.0pt;
  font-family:&quot;Times New Roman Regular&quot;;mso-bidi-font-family:&quot;Times New Roman Regular&quot;;
  color:black;mso-ansi-language:EN-US"><o:p></o:p></span></p>
  </td>
 </tr>
</tbody></table>

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
