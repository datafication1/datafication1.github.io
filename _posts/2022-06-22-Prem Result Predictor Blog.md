---
layout: post
title:  "Predicting Premier League Results with Random Forest"
author: Thomas
categories: [ Jekyll, tutorial ]
image: assets/images/posts/Prem_Result_Predictor/stadium.jpg
---

Introduction

The sports betting industry is worth X million pounds and is only going to increase with the proliferation of online gambling. There are streams of pundits on twitter that offer predictions based on intensive studying of form.Most data models and pundits will use 'team data' to predict results such as team goals scored, team xG and team recent form. What this data fails to take into account is that team selection often changes. Analysing only 'team data' does not account for:
<ul>
<li>Injuries to key players</li>
<li>Resting players for bigger games in the future</li>
<li>New signings bedding into the team</li>
</ul>

The idea behind this blog is to utilise individual player data. This involves averaging eac player on the starting 11s previous stats, to give a more accurate team average. The stats are taken from Fantrax and are calculated by finding the total value of the player stat for the season up until that gameweek (eg number of goals scored up until that gameweek) , and dividing by the number of games the player has played. For each averaged stat, 3 metrics are produced; the home teams average, the away teams average and the difference in the home and away teams average. 


Data Preprocessing 

Firstly, fixture data was imported as csv's (one containing Bet 365 result odds and one containing result by gameweek). A Game ID was created in, as well as a Home Team ID and Away Team ID for matching the corresposnding lineup and player stats. Bet 365 odds columns were utilised to find what the bookmaker believed was the most likely result, and a further column was created to indicate if this was the correct prediction. 


```python
# Bookies predictions
Odds_df.loc[(Odds_df["B365H"]<Odds_df["B365A"]) & (Odds_df["B365H"]<Odds_df["B365D"]),["B365_Prediction"]] = "H"
Odds_df.loc[(Odds_df["B365D"]<Odds_df["B365A"]) & (Odds_df["B365D"]<Odds_df["B365H"]),["B365_Prediction"]] = "D"
Odds_df.loc[(Odds_df["B365A"]<Odds_df["B365D"]) & (Odds_df["B365A"]<Odds_df["B365H"]),["B365_Prediction"]] = "A"

Odds_df.loc[(Odds_df["B365_Prediction"] == Odds_df["FTR"]),["B365_Correct"]] = 1
Odds_df.loc[(Odds_df["B365_Prediction"] != Odds_df["FTR"]),["B365_Correct"]] = 0


# Creating an ID for each game
Odds_df['Date'] = pd.to_datetime(Odds_df['Date'], dayfirst = True)
Odds_df['Game_ID'] = Odds_df['HomeTeam'] + "-" + Odds_df['AwayTeam'] + "-" + Odds_df['Date'].apply(str)


# Combine results and odds dataframe
Results_Merge_df =  Results_df.merge(Odds_df, how='left', on='Game_ID')

# Creating Team ID for joining player stats later
Results_Merge_df['Home Team ID'] = Results_Merge_df['Wk'].apply(str)+'-'+Results_Merge_df['Home Team'] + '-' + Results_Merge_df['Start Year'].apply(str)
Results_Merge_df['Away Team ID'] = Results_Merge_df['Wk'].apply(str)+'-'+Results_Merge_df['Away Team'] + '-' + Results_Merge_df['Start Year'].apply(str)
```

The player data was extracted from fantrax in csv form for each gameweek in the entire 2020/21 season and in the 2021/22 season up until mid-January. There are 3 different views in fantrax that were utilised to obtain all necessary stats for the model. A mixture of quick analysis and domain knowledge gave a selection of stats to utilise in the model. These were:
<ul>
<li>Goals</li>
<li>Assists</li>
<li>Key Passes</li>
<li>Shot</li>
<li>Shots on Target</li>
<li>Clean Sheets</li>
<li>Saves</li>
</ul>
Three folders were created to hold the sets of CSV's. Each folder is passed seperately to the import_files function in order to create a df. Fantrax data contains a unique identifier for each player, which was combined with gameweek number and the season start year to ensure accurate merging of the 3 csv's.   


```python
#Finding local file path for import function
current_path = os.path.abspath(os.path.dirname("__file__"))

#Function to combine csvs for each gameweek into a df
def import_files(subfolder):
    Years = ["2021","2020"]
    csv_list = []
    for Year in Years:
        file_path = os.path.join(current_path, Year, subfolder)
        all_files = glob.glob(file_path + "/*.csv")
        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0, engine='python')
            df['GW'] = filename[-6:-4]
            df['Start Year'] = Year
            csv_list.append(df)
    return pd.concat(csv_list, axis=0, ignore_index=True)

#creating a df of player info and if they started that GW
lineup_frame = import_files("Lineup per GW")

lineup_frame = lineup_frame[['ID','Player','Team','Position','GS','GW','Start Year']]

#removing players that arent in the starting lineup
lineup_frame = lineup_frame.drop(lineup_frame[lineup_frame.GS < 1].index)

#creating player ID for merging of df
lineup_frame['Identifier']= lineup_frame['ID']+"-"+lineup_frame['GW'] + "-" + lineup_frame['Start Year']
lineup_frame
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Player</th>
      <th>Team</th>
      <th>Position</th>
      <th>GS</th>
      <th>GW</th>
      <th>Start Year</th>
      <th>Identifier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>*04amh*</td>
      <td>Aaron Ramsdale</td>
      <td>ARS</td>
      <td>G</td>
      <td>1</td>
      <td>10</td>
      <td>2021</td>
      <td>*04amh*-10-2021</td>
    </tr>
    <tr>
      <th>1</th>
      <td>*02t20*</td>
      <td>Alex McCarthy</td>
      <td>SOU</td>
      <td>G</td>
      <td>1</td>
      <td>10</td>
      <td>2021</td>
      <td>*02t20*-10-2021</td>
    </tr>
    <tr>
      <th>2</th>
      <td>*04rhc*</td>
      <td>Vicente Guaita</td>
      <td>CRY</td>
      <td>G</td>
      <td>1</td>
      <td>10</td>
      <td>2021</td>
      <td>*04rhc*-10-2021</td>
    </tr>
    <tr>
      <th>3</th>
      <td>*05oje*</td>
      <td>Edouard Mendy</td>
      <td>CHE</td>
      <td>G</td>
      <td>1</td>
      <td>10</td>
      <td>2021</td>
      <td>*05oje*-10-2021</td>
    </tr>
    <tr>
      <th>4</th>
      <td>*02ln1*</td>
      <td>David de Gea</td>
      <td>MUN</td>
      <td>G</td>
      <td>1</td>
      <td>10</td>
      <td>2021</td>
      <td>*02ln1*-10-2021</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>39804</th>
      <td>*051kd*</td>
      <td>Jamal Lewis</td>
      <td>NEW</td>
      <td>D</td>
      <td>1</td>
      <td>W9</td>
      <td>2020</td>
      <td>*051kd*-W9-2020</td>
    </tr>
    <tr>
      <th>39805</th>
      <td>*0372y*</td>
      <td>Matt Targett</td>
      <td>AVL</td>
      <td>D</td>
      <td>1</td>
      <td>W9</td>
      <td>2020</td>
      <td>*0372y*-W9-2020</td>
    </tr>
    <tr>
      <th>39806</th>
      <td>*05a6d*</td>
      <td>Tariq Lamptey</td>
      <td>BHA</td>
      <td>D</td>
      <td>1</td>
      <td>W9</td>
      <td>2020</td>
      <td>*05a6d*-W9-2020</td>
    </tr>
    <tr>
      <th>39807</th>
      <td>*02ln2*</td>
      <td>Jonny Evans</td>
      <td>LEI</td>
      <td>D</td>
      <td>1</td>
      <td>W9</td>
      <td>2020</td>
      <td>*02ln2*-W9-2020</td>
    </tr>
    <tr>
      <th>39808</th>
      <td>*03jab*</td>
      <td>Federico Fernandez</td>
      <td>NEW</td>
      <td>D</td>
      <td>1</td>
      <td>W9</td>
      <td>2020</td>
      <td>*03jab*-W9-2020</td>
    </tr>
  </tbody>
</table>
<p>12221 rows × 8 columns</p>
</div>




```python
# creating a df of player stats
stats_frame = import_files("Stats for season")

# creating player ID for merging of df
stats_frame['Identifier']= stats_frame['ID']+"-"+stats_frame['GW'] + "-" + stats_frame['Start Year']

stats_frame = stats_frame[["GP","G","KP","AT","SOT","CS","Sv","Identifier"]]

stats_frame

shots_frame = import_files("Minutes for season")

shots_frame['Identifier']= shots_frame['ID']+"-"+ shots_frame['GW'] + "-" + shots_frame['Start Year']

shots_frame = shots_frame[["S","Identifier"]]

shots_frame

#merging the starting lineup df with the stats df and shots df
initial_merge = lineup_frame.merge(stats_frame, how='left', on='Identifier')

Players_df = initial_merge.merge(shots_frame, how='left', on='Identifier')
Players_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Player</th>
      <th>Team</th>
      <th>Position</th>
      <th>GS</th>
      <th>GW</th>
      <th>Start Year</th>
      <th>Identifier</th>
      <th>GP</th>
      <th>G</th>
      <th>KP</th>
      <th>AT</th>
      <th>SOT</th>
      <th>CS</th>
      <th>Sv</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>*04amh*</td>
      <td>Aaron Ramsdale</td>
      <td>ARS</td>
      <td>G</td>
      <td>1</td>
      <td>10</td>
      <td>2021</td>
      <td>*04amh*-10-2021</td>
      <td>6</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>16.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>*02t20*</td>
      <td>Alex McCarthy</td>
      <td>SOU</td>
      <td>G</td>
      <td>1</td>
      <td>10</td>
      <td>2021</td>
      <td>*02t20*-10-2021</td>
      <td>9</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>22.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>*04rhc*</td>
      <td>Vicente Guaita</td>
      <td>CRY</td>
      <td>G</td>
      <td>1</td>
      <td>10</td>
      <td>2021</td>
      <td>*04rhc*-10-2021</td>
      <td>9</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>26.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>*05oje*</td>
      <td>Edouard Mendy</td>
      <td>CHE</td>
      <td>G</td>
      <td>1</td>
      <td>10</td>
      <td>2021</td>
      <td>*05oje*-10-2021</td>
      <td>8</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>26.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>*02ln1*</td>
      <td>David de Gea</td>
      <td>MUN</td>
      <td>G</td>
      <td>1</td>
      <td>10</td>
      <td>2021</td>
      <td>*02ln1*-10-2021</td>
      <td>9</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>28.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12216</th>
      <td>*051kd*</td>
      <td>Jamal Lewis</td>
      <td>NEW</td>
      <td>D</td>
      <td>1</td>
      <td>W9</td>
      <td>2020</td>
      <td>*051kd*-W9-2020</td>
      <td>8</td>
      <td>0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12217</th>
      <td>*0372y*</td>
      <td>Matt Targett</td>
      <td>AVL</td>
      <td>D</td>
      <td>1</td>
      <td>W9</td>
      <td>2020</td>
      <td>*0372y*-W9-2020</td>
      <td>7</td>
      <td>0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>12218</th>
      <td>*05a6d*</td>
      <td>Tariq Lamptey</td>
      <td>BHA</td>
      <td>D</td>
      <td>1</td>
      <td>W9</td>
      <td>2020</td>
      <td>*05a6d*-W9-2020</td>
      <td>8</td>
      <td>1</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>12219</th>
      <td>*02ln2*</td>
      <td>Jonny Evans</td>
      <td>LEI</td>
      <td>D</td>
      <td>1</td>
      <td>W9</td>
      <td>2020</td>
      <td>*02ln2*-W9-2020</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12220</th>
      <td>*03jab*</td>
      <td>Federico Fernandez</td>
      <td>NEW</td>
      <td>D</td>
      <td>1</td>
      <td>W9</td>
      <td>2020</td>
      <td>*03jab*-W9-2020</td>
      <td>8</td>
      <td>0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>12221 rows × 16 columns</p>
</div>



Various clean up exercises were done post merge including removing unnecessary columns, changing incorrect teams for certain players on certain gameweeks and a creation of a 'Team ID' column to enable grouping by players team and gameweek.

The incorrect teams for certain players is an error in the fantrax data in which if a player changes club during the season, that players team is updated to the new team on all previous gameweeks. A dictionary was produced from a CSV containing the team changes required and in what gameweek they need to take effect from. This was then updated via a for loop as shown below.


```python
Players_df = Players_df[['ID', 'Player', 'Team', 'Position', 'GS', 'GW', 'Identifier',
        'GP', 'G', 'KP', 'AT','SOT','CS', 'Sv','S','Start Year']]
```


```python
#Reading CSV of Players with incorrect teams (normally due to Late Summer or January Transfer)
change_team_dict = {}

with open('Player Team Changes.csv', mode='r') as i:
    reader = csv.reader(i)
    for rows in reader:
        list1 = (rows[1], rows[2], rows[3], rows[4],rows[5])
        change_team_dict[rows[0]] = list1

#Changing incorrect teams for players that moved in January window/late summer window
for k,v in change_team_dict.items():
    if k == "ID":
        continue
    else:
        Players_df.loc[(Players_df["GW"] >= int(v[2])) & (Players_df["ID"]==v[4])& (Players_df["Start Year"]==v[3]),["Team"]] = v[1]

#creation of Team ID for merging fixture dataframe and grouping by game 
Players_df['Team ID'] = Players_df['GW'].apply(str)+"-"+Players_df['Team Name'] + '-' + Players_df['Start Year'].apply(str)
```


Once joined, the Players_df gives a view of each player that started a particular gameweek and there stats up until that gameweek. Below is an example of the Southampton starting lineup in GW 9 of the 2020/21 season


```python
# An example of the Southampton starting lineup in GW 9 of the 2020/21 season
Players_df.loc[Players_df["Team ID"]=="9-Southampton-2020"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Player</th>
      <th>Team</th>
      <th>Position</th>
      <th>GS</th>
      <th>GW</th>
      <th>Identifier</th>
      <th>GP</th>
      <th>G</th>
      <th>KP</th>
      <th>AT</th>
      <th>SOT</th>
      <th>CS</th>
      <th>Sv</th>
      <th>S</th>
      <th>Start Year</th>
      <th>Team Abv</th>
      <th>Team Name</th>
      <th>Team ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>885</th>
      <td>*02t20*</td>
      <td>Alex McCarthy</td>
      <td>SOU</td>
      <td>G</td>
      <td>1</td>
      <td>9</td>
      <td>*02t20*-W9-2020</td>
      <td>8</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>2020</td>
      <td>SOU</td>
      <td>Southampton</td>
      <td>9-Southampton-2020</td>
    </tr>
    <tr>
      <th>1223</th>
      <td>*0522s*</td>
      <td>Che Adams</td>
      <td>SOU</td>
      <td>F</td>
      <td>1</td>
      <td>9</td>
      <td>*0522s*-W9-2020</td>
      <td>8</td>
      <td>3</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>4</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>2020</td>
      <td>SOU</td>
      <td>Southampton</td>
      <td>9-Southampton-2020</td>
    </tr>
    <tr>
      <th>1224</th>
      <td>*02lzn*</td>
      <td>Theo Walcott</td>
      <td>SOU</td>
      <td>M</td>
      <td>1</td>
      <td>9</td>
      <td>*02lzn*-W9-2020</td>
      <td>4</td>
      <td>0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>2020</td>
      <td>SOU</td>
      <td>Southampton</td>
      <td>9-Southampton-2020</td>
    </tr>
    <tr>
      <th>1225</th>
      <td>*0522r*</td>
      <td>Moussa Djenepo</td>
      <td>SOU</td>
      <td>M</td>
      <td>1</td>
      <td>9</td>
      <td>*0522r*-W9-2020</td>
      <td>5</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>2020</td>
      <td>SOU</td>
      <td>Southampton</td>
      <td>9-Southampton-2020</td>
    </tr>
    <tr>
      <th>1226</th>
      <td>*02m79*</td>
      <td>Ryan Bertrand</td>
      <td>SOU</td>
      <td>D</td>
      <td>1</td>
      <td>9</td>
      <td>*02m79*-W9-2020</td>
      <td>7</td>
      <td>0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2020</td>
      <td>SOU</td>
      <td>Southampton</td>
      <td>9-Southampton-2020</td>
    </tr>
    <tr>
      <th>1227</th>
      <td>*04rd1*</td>
      <td>Stuart Armstrong</td>
      <td>SOU</td>
      <td>M</td>
      <td>1</td>
      <td>9</td>
      <td>*04rd1*-W9-2020</td>
      <td>6</td>
      <td>1</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>4</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>2020</td>
      <td>SOU</td>
      <td>Southampton</td>
      <td>9-Southampton-2020</td>
    </tr>
    <tr>
      <th>1228</th>
      <td>*03zsq*</td>
      <td>Kyle Walker-Peters</td>
      <td>SOU</td>
      <td>D</td>
      <td>1</td>
      <td>9</td>
      <td>*03zsq*-W9-2020</td>
      <td>8</td>
      <td>0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>2020</td>
      <td>SOU</td>
      <td>Southampton</td>
      <td>9-Southampton-2020</td>
    </tr>
    <tr>
      <th>1229</th>
      <td>*02m87*</td>
      <td>Oriol Romeu</td>
      <td>SOU</td>
      <td>M</td>
      <td>1</td>
      <td>9</td>
      <td>*02m87*-W9-2020</td>
      <td>8</td>
      <td>1</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>4</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>2020</td>
      <td>SOU</td>
      <td>Southampton</td>
      <td>9-Southampton-2020</td>
    </tr>
    <tr>
      <th>1230</th>
      <td>*04s60*</td>
      <td>Jannik Vestergaard</td>
      <td>SOU</td>
      <td>D</td>
      <td>1</td>
      <td>9</td>
      <td>*04s60*-W9-2020</td>
      <td>7</td>
      <td>2</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>4</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>2020</td>
      <td>SOU</td>
      <td>Southampton</td>
      <td>9-Southampton-2020</td>
    </tr>
    <tr>
      <th>1231</th>
      <td>*02oq8*</td>
      <td>James Ward-Prowse</td>
      <td>SOU</td>
      <td>M</td>
      <td>1</td>
      <td>9</td>
      <td>*02oq8*-W9-2020</td>
      <td>8</td>
      <td>3</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>4</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>2020</td>
      <td>SOU</td>
      <td>Southampton</td>
      <td>9-Southampton-2020</td>
    </tr>
    <tr>
      <th>1232</th>
      <td>*04fk9*</td>
      <td>Jan Bednarek</td>
      <td>SOU</td>
      <td>D</td>
      <td>1</td>
      <td>9</td>
      <td>*04fk9*-W9-2020</td>
      <td>8</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>4</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>2020</td>
      <td>SOU</td>
      <td>Southampton</td>
      <td>9-Southampton-2020</td>
    </tr>
  </tbody>
</table>
</div>



The next step is to find the mean stat value for the season for each player. Since Players_df already contains the player stats up until the designated GW, we added a new average column for each stat. This was done by dividing the particular pl\yer stat by the number of games the players has played up until that GW.


```python
Stat_Columns = ['G', 'KP', 'AT','SOT', 'CS', 'Sv', 'S']

for i in Stat_Columns:
    Players_df[i+'_per_game'] = Players_df[i]/Players_df['GP']
    Players_df[i+'_per_game'] = Players_df[i+'_per_game'].fillna(0)

```


```python
# Remvoing superfluous columns in players dataframe
Players_df = Players_df[['ID', 'Player', 'Team', 'Position', 'GS', 'GW', 'Identifier', 'GP', 'Team Abv',
                         'Team Name', 'Team ID','G_per_game', 'KP_per_game', 'AT_per_game', 'SOT_per_game',
                         'CS_per_game', 'Sv_per_game', 'S_per_game']]
```

The pandas groupby function was used to find the mean value of player stat for each team, up until that particular GW. Esentially this finds the overall mean value of all 11 starting players mean stat value. This gave us Team_Stats_df, a new df of mean team stats.


```python
# Group player stats by the GW the team played in
# This groupby function returns the mean value of the 11 players that played in that teams fixture
Team_Stats_df = Players_df.groupby('Team ID', as_index=False).mean()

Team_Stats_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team ID</th>
      <th>GS</th>
      <th>GW</th>
      <th>GP</th>
      <th>G_per_game</th>
      <th>KP_per_game</th>
      <th>AT_per_game</th>
      <th>SOT_per_game</th>
      <th>CS_per_game</th>
      <th>Sv_per_game</th>
      <th>S_per_game</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10-Arsenal-2020</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>7.181818</td>
      <td>0.044553</td>
      <td>0.511003</td>
      <td>0.085859</td>
      <td>0.191558</td>
      <td>0.256494</td>
      <td>0.252525</td>
      <td>0.639177</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10-Arsenal-2021</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>6.727273</td>
      <td>0.113636</td>
      <td>0.711905</td>
      <td>0.051768</td>
      <td>0.376768</td>
      <td>0.347042</td>
      <td>0.242424</td>
      <td>1.171176</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10-Aston Villa-2020</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>8.000000</td>
      <td>0.187500</td>
      <td>0.912500</td>
      <td>0.187500</td>
      <td>0.462500</td>
      <td>0.500000</td>
      <td>0.287500</td>
      <td>1.175000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10-Aston Villa-2021</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>6.909091</td>
      <td>0.123196</td>
      <td>0.629149</td>
      <td>0.103896</td>
      <td>0.312951</td>
      <td>0.266595</td>
      <td>0.272727</td>
      <td>0.909812</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10-Brentford-2021</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>6.818182</td>
      <td>0.068687</td>
      <td>0.599747</td>
      <td>0.088889</td>
      <td>0.231566</td>
      <td>0.214646</td>
      <td>0.000000</td>
      <td>0.709848</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1097</th>
      <td>9-West Brom-2020</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>5.909091</td>
      <td>0.065260</td>
      <td>0.490260</td>
      <td>0.038961</td>
      <td>0.221104</td>
      <td>0.162338</td>
      <td>0.363636</td>
      <td>0.733009</td>
    </tr>
    <tr>
      <th>1098</th>
      <td>9-West Ham-2020</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>7.400000</td>
      <td>0.091667</td>
      <td>0.648214</td>
      <td>0.141786</td>
      <td>0.245238</td>
      <td>0.395357</td>
      <td>0.250000</td>
      <td>0.859881</td>
    </tr>
    <tr>
      <th>1099</th>
      <td>9-West Ham-2021</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>7.000000</td>
      <td>0.167208</td>
      <td>0.983766</td>
      <td>0.165584</td>
      <td>0.474026</td>
      <td>0.275974</td>
      <td>0.238636</td>
      <td>1.391234</td>
    </tr>
    <tr>
      <th>1100</th>
      <td>9-Wolves-2020</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>6.727273</td>
      <td>0.103896</td>
      <td>0.616775</td>
      <td>0.057143</td>
      <td>0.274892</td>
      <td>0.470130</td>
      <td>0.227273</td>
      <td>0.911797</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>9-Wolves-2021</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>6.727273</td>
      <td>0.088636</td>
      <td>0.658766</td>
      <td>0.052273</td>
      <td>0.239610</td>
      <td>0.210065</td>
      <td>0.272727</td>
      <td>0.925000</td>
    </tr>
  </tbody>
</table>
<p>1102 rows × 11 columns</p>
</div>




```python
# Grouping player stats by the GW the team played in but returning the sum rather than the mean
# This is only done for validation that each team has 11 players in each GW
Games_Played_df = Players_df[['Team ID', 'GS']]
Games_Played_df = Games_Played_df.groupby('Team ID', as_index=False).sum()

# Combing the mean player stats with the games played df
Team_Stats_df = Team_Stats_df.merge(Games_Played_df, how='left', left_on='Team ID', right_on='Team ID',suffixes=('_mean','_sum'))


Team_Stats_df.columns

```




    Index(['Team ID', 'GS_mean', 'GW', 'GP', 'G_per_game', 'KP_per_game',
           'AT_per_game', 'SOT_per_game', 'CS_per_game', 'Sv_per_game',
           'S_per_game', 'GS_sum'],
          dtype='object')



We then combine this new df with the fixture df created earlier. We first complete a left outer join using the Home Team ID of the Results_Merge_df and the Team ID of the Team_Stats_df. 


```python
# Merging the results df with the teams stats df for the home teams 
Final_Stats_df = Results_Merge_df.merge(Team_Stats_df, how='left', left_on='Home Team ID', right_on='Team ID')
```


```python
# Renaming columns to clarify they relate to the home team
Final_Stats_df = Final_Stats_df.rename(columns={col: 'Home_'+col 
                        for col in Team_Stats_df.columns if col.endswith('per_game')})

```

Then repeat the same process but for the Away Teams.


```python
# Merging the final stats df with the teams stats df for the away teams 
Final_Stats_df = Final_Stats_df.merge(Team_Stats_df, how='left', left_on='Away Team ID', right_on='Team ID')
```


```python
# Renaming columns to clarify they relate to the away team
Final_Stats_df = Final_Stats_df.rename(columns={col: 'Away_'+col 
                        for col in Team_Stats_df.columns if (col.endswith('per_game') == True) and (col.startswith('Home') == False)})
```


```python

```

The last set of columns used for the model are differential columns, ie Subtracting the away teams stats from the home teams stats. The idea for this comes from the paper X LINK PAPER. The basis for it is that a large positive difference would highlight the home side has a significantly better mean value for that stat, and vice versa if there was a large netaive difference the away team would have a significantly better value. This is something the basic home and away mean value may not show as clearly.


```python
for i in Stat_Columns:
    Final_Stats_df[i+'_diff'] = Final_Stats_df['Home_'+i+'_per_game']-Final_Stats_df['Away_'+i+'_per_game']

    
```


```python
# Final_Stats_df.to_csv('finals.csv')
```

Games in which there were NA values for any of the team stats and in which there was not 11 players found for each team, were removed. As well as this games which occured before GW in the season were removed. The idea here is that it would take at least 8 games for player stats to even out due to varying strength of schedule in the early season. By GW 8 it is assumed most teams have played some 'good' teams, and some 'not as good' teams.


```python
#Removing rows with N/A values and matches in which both teams dont have valid starting 11's
Final_Stats_df = Final_Stats_df.dropna()
Final_Stats_df = Final_Stats_df.loc[(Final_Stats_df['GS_sum_x'] == 11)  & (Final_Stats_df['GS_sum_y'] == 11)]

#Removing games which occured before GW 8 in the season
Final_Stats_df = Final_Stats_df.loc[Final_Stats_df['Wk'] > 7]
Final_Stats_df.columns
```




    Index(['Match Number', 'Round Number', 'Date_x', 'Location', 'Home Team',
           'Away Team', 'Result', 'Start Year', 'Wk', 'Home Score', 'Away Score',
           'Game_ID', 'Date_y', 'HomeTeam', 'AwayTeam', 'FTR', 'B365H', 'B365D',
           'B365A', 'B365_Prediction', 'B365_Correct', 'Home Team ID',
           'Away Team ID', 'Team ID_x', 'GS_mean_x', 'GW_x', 'GP_x',
           'Home_G_per_game', 'Home_KP_per_game', 'Home_AT_per_game',
           'Home_SOT_per_game', 'Home_CS_per_game', 'Home_Sv_per_game',
           'Home_S_per_game', 'GS_sum_x', 'Team ID_y', 'GS_mean_y', 'GW_y', 'GP_y',
           'Away_G_per_game', 'Away_KP_per_game', 'Away_AT_per_game',
           'Away_SOT_per_game', 'Away_CS_per_game', 'Away_Sv_per_game',
           'Away_S_per_game', 'GS_sum_y', 'G_diff', 'KP_diff', 'AT_diff',
           'SOT_diff', 'CS_diff', 'Sv_diff', 'S_diff'],
          dtype='object')




```python
print("B365 accuracy - " + str((Final_Stats_df["B365_Correct"].sum())/(Final_Stats_df.shape[0])))
```

    B365 accuracy - 0.5171339563862928
    

Demonstrating the benefit of using the differential columns, we can see a subest of the data below showing the top 10 negative values of clean sheet difference ie the away team keeps much more clean sheets than the home team. We can see of the 10 that 9 resulted in away wins, with 7 of them involving the away team keeping a clean sheet and scoring 3 or more goals. This one stat potentially shows us the clean sheet likelihood of both teams as well as the likelihood for a high scoring win. 


```python
Final_Stats_df[['Home Team','Away Team','Result','CS_diff']].sort_values(by=['CS_diff'], ascending=True).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Home Team</th>
      <th>Away Team</th>
      <th>Result</th>
      <th>CS_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>143</th>
      <td>Watford</td>
      <td>Man City</td>
      <td>1 - 3</td>
      <td>-0.531624</td>
    </tr>
    <tr>
      <th>94</th>
      <td>Newcastle</td>
      <td>Chelsea</td>
      <td>0 - 3</td>
      <td>-0.492424</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Man Utd</td>
      <td>Liverpool</td>
      <td>0 - 5</td>
      <td>-0.482143</td>
    </tr>
    <tr>
      <th>397</th>
      <td>West Brom</td>
      <td>Man City</td>
      <td>0 - 5</td>
      <td>-0.455934</td>
    </tr>
    <tr>
      <th>110</th>
      <td>Leicester</td>
      <td>Chelsea</td>
      <td>0 - 3</td>
      <td>-0.448439</td>
    </tr>
    <tr>
      <th>131</th>
      <td>Watford</td>
      <td>Chelsea</td>
      <td>1 - 2</td>
      <td>-0.429565</td>
    </tr>
    <tr>
      <th>344</th>
      <td>West Brom</td>
      <td>Aston Villa</td>
      <td>0 - 3</td>
      <td>-0.403874</td>
    </tr>
    <tr>
      <th>179</th>
      <td>Leicester</td>
      <td>Liverpool</td>
      <td>1 - 0</td>
      <td>-0.400629</td>
    </tr>
    <tr>
      <th>166</th>
      <td>Newcastle</td>
      <td>Man City</td>
      <td>0 - 4</td>
      <td>-0.395483</td>
    </tr>
    <tr>
      <th>70</th>
      <td>Watford</td>
      <td>Liverpool</td>
      <td>0 - 5</td>
      <td>-0.386797</td>
    </tr>
  </tbody>
</table>
</div>



Below is various box plots for the differntial columns showing the maximum value (the top horizontal line), the third quartile (the top of the coloured box), the median value (the middle horizontal line), the first quartile (the bottom of the coloured box) and the minimum value ( bottom horizontal line). The third and first quartiles are the 'middle' values between the maximum and the median, and the minimum and the median, respectively. we can also see some outliers in the plots (the diamonds) that are calcualted as outliers using a function of the inter-quartile range. https://seaborn.pydata.org/generated/seaborn.boxplot.html


```python
diff_list = {'G_diff', 'KP_diff',
         'AT_diff',  'SOT_diff',
         'CS_diff', 'Sv_diff', 'S_diff' }

for i in diff_list:
    sb.catplot(x="FTR", y=i, kind="box", data=Final_Stats_df)
    
```


    
![png](output_43_0.png)
    



    
![png](output_43_1.png)
    



    
![png](output_43_2.png)
    



    
![png](output_43_3.png)
    



    
![png](output_43_4.png)
    



    
![png](output_43_5.png)
    



    
![png](output_43_6.png)
    


We can see that in most cases the data is distributed as expected for home and away wins (the home win quantiles are of greater value than the away win quantiles). We expect this as postive differnetial value signifies the home teams stat is higher, and vice versa for a ngetaive differential value. The exceprion here is save differences which is reversed. This is also expected as its likely a poorer performing teams keeper would need to make more saves due to the number of shots being faced on average per game. 

The data distribution for draws is slightly less predictable. The median value for draws in each stat generlly lies somewhere between the home and away value, but there is overlap in the quantile values for draws and that of the home and away wins. Therefore it may be harder for a model based on this data to predict draws, perhaps something highlighted in the Bet365 prediction data.


```python
print(list(Final_Stats_df))
```

    ['Match Number', 'Round Number', 'Date_x', 'Location', 'Home Team', 'Away Team', 'Result', 'Start Year', 'Wk', 'Home Score', 'Away Score', 'Game_ID', 'Date_y', 'HomeTeam', 'AwayTeam', 'FTR', 'B365H', 'B365D', 'B365A', 'B365_Prediction', 'B365_Correct', 'Home Team ID', 'Away Team ID', 'Team ID_x', 'GS_mean_x', 'GW_x', 'GP_x', 'Home_G_per_game', 'Home_KP_per_game', 'Home_AT_per_game', 'Home_SOT_per_game', 'Home_CS_per_game', 'Home_Sv_per_game', 'Home_S_per_game', 'GS_sum_x', 'Team ID_y', 'GS_mean_y', 'GW_y', 'GP_y', 'Away_G_per_game', 'Away_KP_per_game', 'Away_AT_per_game', 'Away_SOT_per_game', 'Away_CS_per_game', 'Away_Sv_per_game', 'Away_S_per_game', 'GS_sum_y', 'G_diff', 'KP_diff', 'AT_diff', 'SOT_diff', 'CS_diff', 'Sv_diff', 'S_diff']
    

Below is a pie chart of the final dataset and what result occured in each match, Home win is the most common result with 38.9% of the share , but it is only just ahead of away wins which consitute 37.1% of the dataset. Draws are the least common reslut with only 24% of the result.


```python
Result_List = Final_Stats_df['FTR']
Bet365_List = Final_Stats_df['B365_Prediction']

Final_Stats_df.groupby('FTR').size().plot(kind='pie',autopct='%1.1f%%')
```




    <AxesSubplot:ylabel='None'>




    
![png](output_47_1.png)
    


The bet365 result predicitons look a little different from the actual results, We can see they favour Home results, predciciting it 58.3% of the time. They also predicted away wins more often than what actually occured. Whats interesting here is that of all games in the dataset, they didnt predict a draw once. This is most likely due to betting companies altering prices to match the way the market is betting in order to cover losses and maximise profit. Its uncommon for an average better to place money on the draw and so these odds would naturally be higher. 


```python
Final_Stats_df.groupby('B365_Prediction').size().plot(kind='pie',autopct='%1.1f%%')
```




    <AxesSubplot:ylabel='None'>




    
![png](output_49_1.png)
    


With regards to accuracy, we can see Bet365 correctly predicted the correct match result 52% of the time. The confusion matrix shows us again the tendency of Bet365 to predict home wins over away wins. However, when doing so the success rate of a correct prediction is similar ( 53% for away and 51% for home).


```python
b365_pred = round(100*((Final_Stats_df["B365_Correct"].sum())/(Final_Stats_df.shape[0])))
print("Bet 365 Result Prediciton Accuracy - " f"{b365_pred}%")

Conf_Matrix = confusion_matrix(Result_List, Bet365_List,labels=['A','D','H'])

disp = ConfusionMatrixDisplay(Conf_Matrix,display_labels=['A','D','H'])
disp.plot()

plt.show()
```

    Bet 365 Result Prediciton Accuracy - 52%
    


    
![png](output_51_1.png)
    


# Data Modelling

Once all data had been cleaned and analysed I began to optimise the model. I elected to use only a Random Forrest for this project as it . A random forest is made up of many decison trees which are trained on different subsets of the training data. Decision Trees work by placing all of the training data selected into a root node, and splitting it into various sub nodes based on a ruleset of the features, with the goal of making each sub-node more homogenous than the root node. This process repeats until each node is pure (not recommended as this can lead to over-fitting), or when a predefined criteria has been met eg max depth of tree has been reached. 

*add in tree graphic

https://www.javatpoint.com/machine-learning-random-forest-algorithm


```python

cols = ['Home_G_per_game', 'Home_KP_per_game', 'Home_AT_per_game', 
         'Home_SOT_per_game',     'Home_CS_per_game', 'Home_Sv_per_game', 'Home_S_per_game',
         'Away_G_per_game',
         'Away_KP_per_game', 'Away_AT_per_game', 'Away_SOT_per_game',  'Away_CS_per_game',
         'Away_Sv_per_game', 'Away_S_per_game',   'G_diff', 'KP_diff',
         'AT_diff',  'SOT_diff',
         'CS_diff', 'Sv_diff', 'S_diff']

X_all = Final_Stats_df[cols]
y_all = Final_Stats_df['FTR']




c = X_all.columns

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, random_state=42,
                                                    test_size = 0.2)

```


```python

```


```python

```


```python

```


```python


```


```python

```


```python


```


```python

```


```python

```


```python

```


```python

```


```python

```


```python
# feature_importances = clf.feature_importances_
# importances = list(zip(feature_importances, cols))
# importances.sort(reverse=True)
# importances
```


```python

```


```python

```


```python

```


```python

```


```python

```


```python


```

A pipeline was created for the model so we could tune the model with optimal parameters as well as finding the relevant features for result classification.

The number of trees was kept consistent at 100 and the gini impurity was used as the split criterion. The Gini impurity works by calculating what ruleset would produce subnodes with data that has the lowest likelihood of misclasssification when randomnly labelled. A fully pure node would produce a gini index  of 0 as there is no chance of a sample being misclassified. Therefore each node is split based on what feature based ruleset gives the split the lowest gini index.

However, other parameters such as max depth (the maximum number of vertical layers each branch can have), max features (the number of features considered at each node for the split) and min samples per leaf (the minimum number of samples required of both children nodes for the parent node to be split) were tuned in the cross validated random grid search. - https://scikit-learn.org/stable/modules/ensemble.html#forest

As well as this, a SelectKBest parameter was tuned in the random grid search in which the top k features (ANOVA F test) are used in fitting the model. The random grid search also includes a stratified K fold cross validation step to ensure no overfitting occured. 

An 80/20 split of training to test data was chosen giving us a training set size of 256 games.


```python
RF = RandomForestClassifier(random_state=42)


pipeline = Pipeline(
    [('selector',SelectKBest()),
     ('model',RF)
    ]
)


```


```python
# Creation of Paramater Grid for Randomised Search CV
random_grid = {'selector__k':[8,9,10,11,12,13,14,15, 16, 17, 18, 19, 20],
   'model__max_depth': [1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,30,50,75,100],
   'model__max_features': ['sqrt', 'log2'],
    'model__min_samples_leaf' : [1, 2, 4, 6, 8,10,15,20]}



clf = RandomizedSearchCV(estimator = pipeline , param_distributions = random_grid, n_iter=400,
                         cv = 4, verbose=2,scoring ='accuracy' , n_jobs = -1)

clf.fit(X_train, y_train)


y_pred = clf.predict_proba(X_test)
df3 = pd.DataFrame({'Actual': y_test, 'Prediction': clf.predict(X_test), 'Prediction%': y_pred[:,1]*100})
df3
```

    Fitting 4 folds for each of 400 candidates, totalling 1600 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed:    9.8s
    [Parallel(n_jobs=-1)]: Done 146 tasks      | elapsed:   16.9s
    [Parallel(n_jobs=-1)]: Done 349 tasks      | elapsed:   28.7s
    [Parallel(n_jobs=-1)]: Done 632 tasks      | elapsed:   46.3s
    [Parallel(n_jobs=-1)]: Done 997 tasks      | elapsed:  1.1min
    [Parallel(n_jobs=-1)]: Done 1442 tasks      | elapsed:  1.5min
    [Parallel(n_jobs=-1)]: Done 1600 out of 1600 | elapsed:  1.7min finished
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual</th>
      <th>Prediction</th>
      <th>Prediction%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>366</th>
      <td>H</td>
      <td>A</td>
      <td>26.754163</td>
    </tr>
    <tr>
      <th>305</th>
      <td>A</td>
      <td>A</td>
      <td>29.717722</td>
    </tr>
    <tr>
      <th>411</th>
      <td>H</td>
      <td>H</td>
      <td>22.349317</td>
    </tr>
    <tr>
      <th>80</th>
      <td>H</td>
      <td>H</td>
      <td>27.996628</td>
    </tr>
    <tr>
      <th>188</th>
      <td>H</td>
      <td>H</td>
      <td>25.565425</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>454</th>
      <td>A</td>
      <td>A</td>
      <td>23.550599</td>
    </tr>
    <tr>
      <th>136</th>
      <td>A</td>
      <td>A</td>
      <td>23.611830</td>
    </tr>
    <tr>
      <th>542</th>
      <td>A</td>
      <td>A</td>
      <td>27.494971</td>
    </tr>
    <tr>
      <th>510</th>
      <td>A</td>
      <td>A</td>
      <td>28.280432</td>
    </tr>
    <tr>
      <th>287</th>
      <td>A</td>
      <td>A</td>
      <td>28.261162</td>
    </tr>
  </tbody>
</table>
<p>65 rows × 3 columns</p>
</div>




```python
score = clf.score(X_test, y_test)
print(clf.best_estimator_)
print(score)
```

    Pipeline(steps=[('selector', SelectKBest(k=17)),
                    ('model',
                     RandomForestClassifier(max_depth=2, max_features='sqrt',
                                            min_samples_leaf=6, random_state=42))])
    0.676923076923077
    

 

The following hypreparameters were returned:

max_depth=2, min_samples_leaf=10, max_features: 'sqrt'

The Select KBest returned 17 features out of the available 21, The features excldued were:
Home_G_per_game
Home_KP_per_game
Home_AT_per_game
KP_diff


```python
#Collating the optimal features as chosen by k select
Selected_feature_names = np.array(cols)[clf.best_estimator_.named_steps['selector'].get_support(indices=True)]
Selected_feature_names
```




    array(['Home_SOT_per_game', 'Home_CS_per_game', 'Home_Sv_per_game',
           'Home_S_per_game', 'Away_G_per_game', 'Away_KP_per_game',
           'Away_AT_per_game', 'Away_SOT_per_game', 'Away_CS_per_game',
           'Away_S_per_game', 'G_diff', 'KP_diff', 'AT_diff', 'SOT_diff',
           'CS_diff', 'Sv_diff', 'S_diff'], dtype='<U17')



The cross validated model was then tested on the test set and produced an impressive accuracy of 68%. This is higher than the Bet365 accuracy of the same Test set of 62%. Most pundits struglle to obtain a prediciton accuracy of anything over 60% throughout the season.As Analysed in this blog post - http://eightyfivepoints.blogspot.com/2017/08/lawrenson-and-merson-dream-team.html - from the 2014/15 season through to the 2016/17 season, pundits Paul Merson and Mark Lawrenson had a result prediction accuracy of just over 50%. Considering our model was fairly simple, trained on just over 250 games and can not take into account any 'human' element that pundits caan, an accuracy of 68% is a significant result.


```python

cols = ['Home_SOT_per_game', 'Home_CS_per_game',
       'Home_Sv_per_game', 'Home_S_per_game',
       'Away_G_per_game', 'Away_KP_per_game',
       'Away_AT_per_game', 'Away_SOT_per_game',
       'Away_CS_per_game', 'Away_Sv_per_game',
       'Away_S_per_game', 'G_diff', 'AT_diff',
       'SOT_diff', 'CS_diff', 'Sv_diff', 'S_diff']


X_all = Final_Stats_df[cols]
y_all = Final_Stats_df['FTR']




c = X_all.columns


X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, random_state=42,
                                                    test_size = 0.2)

```


```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler
from sklearn.pipeline import make_pipeline

clf = RandomForestClassifier(max_depth=2, min_samples_leaf=10,
                                        random_state=42)
clf.fit(X_train,y_train)
print(f"{round(100*clf.score(X_test,y_test))}%")
y_pred = clf.predict_proba(X_test)
df3 = pd.DataFrame({'Actual': y_test, 'Prediction': clf.predict(X_test)})
predictions_df = df3.join(Final_Stats_df)
predictions_df.to_csv('Predictions.csv')
```

    68%
    


```python
predictions_df.columns
```




    Index(['Actual', 'Prediction', 'Match Number', 'Round Number', 'Date_x',
           'Location', 'Home Team', 'Away Team', 'Result', 'Start Year', 'Wk',
           'Home Score', 'Away Score', 'Game_ID', 'Date_y', 'HomeTeam', 'AwayTeam',
           'FTR', 'B365H', 'B365D', 'B365A', 'B365_Prediction', 'B365_Correct',
           'Home Team ID', 'Away Team ID', 'Team ID_x', 'GS_mean_x', 'GW_x',
           'GP_x', 'Home_G_per_game', 'Home_KP_per_game', 'Home_AT_per_game',
           'Home_SOT_per_game', 'Home_CS_per_game', 'Home_Sv_per_game',
           'Home_S_per_game', 'GS_sum_x', 'Team ID_y', 'GS_mean_y', 'GW_y', 'GP_y',
           'Away_G_per_game', 'Away_KP_per_game', 'Away_AT_per_game',
           'Away_SOT_per_game', 'Away_CS_per_game', 'Away_Sv_per_game',
           'Away_S_per_game', 'GS_sum_y', 'G_diff', 'KP_diff', 'AT_diff',
           'SOT_diff', 'CS_diff', 'Sv_diff', 'S_diff'],
          dtype='object')




```python
print("B365 Test accuracy - " + str((predictions_df["B365_Correct"].sum())/(predictions_df.shape[0])))
```

    B365 Test accuracy - 0.6153846153846154
    


```python
z = ['Actual', 'Prediction', 'Round Number', 'Date_x',
        'Home Team', 'Away Team', 'Result', 'B365_Prediction', 'B365_Correct',
        'Home_G_per_game', 'Home_KP_per_game', 'Home_AT_per_game',
       'Home_SOT_per_game', 'Home_CS_per_game', 'Home_Sv_per_game',
       'Home_S_per_game',  'Away_G_per_game', 'Away_KP_per_game', 'Away_AT_per_game',
       'Away_SOT_per_game', 'Away_CS_per_game', 'Away_Sv_per_game',
       'Away_S_per_game', 'G_diff', 'KP_diff', 'AT_diff',
       'SOT_diff', 'CS_diff', 'Sv_diff', 'S_diff']
```

The test data ctually contained a slightly different split than the overall data as shown below. 


```python
predictions_df.groupby('Actual').size().plot(kind='pie',autopct='%1.1f%%')
```




    <AxesSubplot:ylabel='None'>




    
![png](output_86_1.png)
    



```python

```




    256



The model prodcued the following confusion matrix with the corresposnding result split.


```python
Conf_Matrix = confusion_matrix(y_test,clf.predict(X_test),labels=['A','D','H'])

disp = ConfusionMatrixDisplay(Conf_Matrix,display_labels=['A','D','H'])
disp.plot()

plt.show()
```


    
![png](output_89_0.png)
    



```python
predictions_df.groupby('Prediction').size().plot(kind='pie',autopct='%1.1f%%')
```




    <AxesSubplot:ylabel='None'>




    
![png](output_90_1.png)
    


Its noted that in a simlar fashion to the Bet365 predicitions, the model favours the predicition of away wins, but the prediction accuracy for both is much the same (69% for away wins and 66% for home wins). Our model also correctly classified a draw! This game was Wolves v Burnley 


```python
cols2 = ['Home_G_per_game', 'Home_KP_per_game', 'Home_AT_per_game',
       'Home_SOT_per_game', 'Home_CS_per_game', 'Home_Sv_per_game',
       'Home_S_per_game',  'Away_G_per_game', 'Away_KP_per_game', 'Away_AT_per_game',
       'Away_SOT_per_game', 'Away_CS_per_game', 'Away_Sv_per_game',
       'Away_S_per_game', 'G_diff', 'KP_diff', 'AT_diff',
       'SOT_diff', 'CS_diff', 'Sv_diff', 'S_diff']
```

Interestingly, there were 10 games in the 65 game test set that the random forest model predicited correctly, yet Bet365 predcited wrong. This included 4 games where Arsenal lost in the 2020/21 season, 3 of which were at the emirates. This could be a market bias in which 'big teams' are favoured.


```python
predictions_df.loc[(predictions_df['Actual'] == predictions_df['Prediction']) & (predictions_df['B365_Correct'] == 0)][z]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual</th>
      <th>Prediction</th>
      <th>Round Number</th>
      <th>Date_x</th>
      <th>Home Team</th>
      <th>Away Team</th>
      <th>Result</th>
      <th>B365_Prediction</th>
      <th>B365_Correct</th>
      <th>Home_G_per_game</th>
      <th>...</th>
      <th>Away_CS_per_game</th>
      <th>Away_Sv_per_game</th>
      <th>Away_S_per_game</th>
      <th>G_diff</th>
      <th>KP_diff</th>
      <th>AT_diff</th>
      <th>SOT_diff</th>
      <th>CS_diff</th>
      <th>Sv_diff</th>
      <th>S_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>305</th>
      <td>A</td>
      <td>A</td>
      <td>10</td>
      <td>2020-11-29</td>
      <td>Arsenal</td>
      <td>Wolves</td>
      <td>1 - 2</td>
      <td>H</td>
      <td>0.0</td>
      <td>0.044553</td>
      <td>...</td>
      <td>0.399315</td>
      <td>0.212121</td>
      <td>0.883983</td>
      <td>-0.027417</td>
      <td>-0.184488</td>
      <td>0.021465</td>
      <td>-0.108405</td>
      <td>-0.142821</td>
      <td>0.040404</td>
      <td>-0.244805</td>
    </tr>
    <tr>
      <th>530</th>
      <td>A</td>
      <td>A</td>
      <td>33</td>
      <td>2021-04-23</td>
      <td>Arsenal</td>
      <td>Everton</td>
      <td>0 - 1</td>
      <td>H</td>
      <td>0.0</td>
      <td>0.050505</td>
      <td>...</td>
      <td>0.234469</td>
      <td>0.265152</td>
      <td>0.883066</td>
      <td>-0.067760</td>
      <td>-0.037241</td>
      <td>-0.001802</td>
      <td>-0.170107</td>
      <td>0.017656</td>
      <td>-0.037879</td>
      <td>-0.125354</td>
    </tr>
    <tr>
      <th>133</th>
      <td>D</td>
      <td>D</td>
      <td>14</td>
      <td>2021-12-01</td>
      <td>Wolves</td>
      <td>Burnley</td>
      <td>0 - 0</td>
      <td>H</td>
      <td>0.0</td>
      <td>0.078322</td>
      <td>...</td>
      <td>0.097009</td>
      <td>0.287879</td>
      <td>0.845205</td>
      <td>-0.037118</td>
      <td>0.003886</td>
      <td>0.007664</td>
      <td>-0.075497</td>
      <td>0.206099</td>
      <td>0.005828</td>
      <td>-0.109658</td>
    </tr>
    <tr>
      <th>425</th>
      <td>H</td>
      <td>H</td>
      <td>23</td>
      <td>2021-02-06</td>
      <td>Aston Villa</td>
      <td>Arsenal</td>
      <td>1 - 0</td>
      <td>A</td>
      <td>0.0</td>
      <td>0.133818</td>
      <td>...</td>
      <td>0.339927</td>
      <td>0.132231</td>
      <td>0.907078</td>
      <td>0.028338</td>
      <td>0.319567</td>
      <td>0.020895</td>
      <td>0.071236</td>
      <td>0.106548</td>
      <td>0.195041</td>
      <td>0.186279</td>
    </tr>
    <tr>
      <th>121</th>
      <td>A</td>
      <td>A</td>
      <td>13</td>
      <td>2021-11-27</td>
      <td>Crystal Palace</td>
      <td>Aston Villa</td>
      <td>1 - 2</td>
      <td>H</td>
      <td>0.0</td>
      <td>0.115358</td>
      <td>...</td>
      <td>0.271320</td>
      <td>0.264463</td>
      <td>0.860941</td>
      <td>0.034465</td>
      <td>0.048898</td>
      <td>-0.006631</td>
      <td>-0.024439</td>
      <td>-0.009750</td>
      <td>0.015840</td>
      <td>-0.058668</td>
    </tr>
    <tr>
      <th>77</th>
      <td>A</td>
      <td>A</td>
      <td>8</td>
      <td>2021-10-17</td>
      <td>Everton</td>
      <td>West Ham</td>
      <td>0 - 1</td>
      <td>H</td>
      <td>0.0</td>
      <td>0.116883</td>
      <td>...</td>
      <td>0.119048</td>
      <td>0.246753</td>
      <td>1.387446</td>
      <td>-0.062771</td>
      <td>-0.004329</td>
      <td>-0.022727</td>
      <td>-0.193723</td>
      <td>0.173160</td>
      <td>0.041126</td>
      <td>-0.465368</td>
    </tr>
    <tr>
      <th>581</th>
      <td>H</td>
      <td>H</td>
      <td>38</td>
      <td>2021-05-23</td>
      <td>Aston Villa</td>
      <td>Chelsea</td>
      <td>2 - 1</td>
      <td>A</td>
      <td>0.0</td>
      <td>0.136176</td>
      <td>...</td>
      <td>0.435192</td>
      <td>0.169697</td>
      <td>1.011055</td>
      <td>0.040117</td>
      <td>-0.060717</td>
      <td>0.015226</td>
      <td>0.056380</td>
      <td>-0.055985</td>
      <td>0.164455</td>
      <td>0.106164</td>
    </tr>
    <tr>
      <th>542</th>
      <td>A</td>
      <td>A</td>
      <td>34</td>
      <td>2021-05-01</td>
      <td>Everton</td>
      <td>Aston Villa</td>
      <td>1 - 2</td>
      <td>H</td>
      <td>0.0</td>
      <td>0.092300</td>
      <td>...</td>
      <td>0.381953</td>
      <td>0.332386</td>
      <td>1.051807</td>
      <td>-0.015517</td>
      <td>-0.168283</td>
      <td>0.018586</td>
      <td>-0.108702</td>
      <td>-0.118797</td>
      <td>-0.066932</td>
      <td>-0.280317</td>
    </tr>
    <tr>
      <th>510</th>
      <td>A</td>
      <td>A</td>
      <td>31</td>
      <td>2021-04-09</td>
      <td>Fulham</td>
      <td>Wolves</td>
      <td>0 - 1</td>
      <td>H</td>
      <td>0.0</td>
      <td>0.046666</td>
      <td>...</td>
      <td>0.258956</td>
      <td>0.233333</td>
      <td>1.032152</td>
      <td>-0.024766</td>
      <td>-0.202464</td>
      <td>-0.038630</td>
      <td>-0.165782</td>
      <td>-0.017058</td>
      <td>0.057576</td>
      <td>-0.409362</td>
    </tr>
    <tr>
      <th>287</th>
      <td>A</td>
      <td>A</td>
      <td>8</td>
      <td>2020-11-08</td>
      <td>Arsenal</td>
      <td>Aston Villa</td>
      <td>0 - 3</td>
      <td>H</td>
      <td>0.0</td>
      <td>0.104762</td>
      <td>...</td>
      <td>0.469697</td>
      <td>0.303030</td>
      <td>1.257576</td>
      <td>-0.107359</td>
      <td>-0.509307</td>
      <td>-0.099567</td>
      <td>-0.168831</td>
      <td>-0.112987</td>
      <td>-0.069264</td>
      <td>-0.548052</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 30 columns</p>
</div>



There were 6 instances in which the reverse was true ( Bet365 predicted correctly and the model predicted incorrectly). 5 of the 6 were home wins predicted as away wins. This is likely the model not taking into account the true effect of home field advantage.


```python
predictions_df.loc[(predictions_df['Actual'] != predictions_df['Prediction']) & (predictions_df['B365_Correct'] == 1)][z]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual</th>
      <th>Prediction</th>
      <th>Round Number</th>
      <th>Date_x</th>
      <th>Home Team</th>
      <th>Away Team</th>
      <th>Result</th>
      <th>B365_Prediction</th>
      <th>B365_Correct</th>
      <th>Home_G_per_game</th>
      <th>...</th>
      <th>Away_CS_per_game</th>
      <th>Away_Sv_per_game</th>
      <th>Away_S_per_game</th>
      <th>G_diff</th>
      <th>KP_diff</th>
      <th>AT_diff</th>
      <th>SOT_diff</th>
      <th>CS_diff</th>
      <th>Sv_diff</th>
      <th>S_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>366</th>
      <td>H</td>
      <td>A</td>
      <td>17</td>
      <td>2021-01-01</td>
      <td>Man Utd</td>
      <td>Aston Villa</td>
      <td>2 - 1</td>
      <td>H</td>
      <td>1.0</td>
      <td>0.145910</td>
      <td>...</td>
      <td>0.501290</td>
      <td>0.292208</td>
      <td>1.258242</td>
      <td>-0.033119</td>
      <td>-0.147124</td>
      <td>-0.017063</td>
      <td>-0.109355</td>
      <td>-0.262903</td>
      <td>-0.117383</td>
      <td>-0.230217</td>
    </tr>
    <tr>
      <th>162</th>
      <td>H</td>
      <td>A</td>
      <td>17</td>
      <td>2021-12-15</td>
      <td>Arsenal</td>
      <td>West Ham</td>
      <td>2 - 0</td>
      <td>H</td>
      <td>1.0</td>
      <td>0.085529</td>
      <td>...</td>
      <td>0.238578</td>
      <td>0.272727</td>
      <td>1.082314</td>
      <td>-0.044016</td>
      <td>0.086419</td>
      <td>-0.054720</td>
      <td>-0.180656</td>
      <td>0.169450</td>
      <td>0.083916</td>
      <td>-0.260688</td>
    </tr>
    <tr>
      <th>156</th>
      <td>H</td>
      <td>A</td>
      <td>16</td>
      <td>2021-12-12</td>
      <td>Leicester</td>
      <td>Newcastle</td>
      <td>4 - 0</td>
      <td>H</td>
      <td>1.0</td>
      <td>0.088274</td>
      <td>...</td>
      <td>0.161609</td>
      <td>0.303030</td>
      <td>0.926761</td>
      <td>-0.004353</td>
      <td>-0.026093</td>
      <td>0.038202</td>
      <td>0.031478</td>
      <td>-0.106919</td>
      <td>0.018182</td>
      <td>-0.016659</td>
    </tr>
    <tr>
      <th>75</th>
      <td>H</td>
      <td>A</td>
      <td>8</td>
      <td>2021-10-16</td>
      <td>Southampton</td>
      <td>Leeds</td>
      <td>1 - 0</td>
      <td>H</td>
      <td>1.0</td>
      <td>0.018182</td>
      <td>...</td>
      <td>0.135931</td>
      <td>0.363636</td>
      <td>0.910823</td>
      <td>-0.030303</td>
      <td>-0.334848</td>
      <td>-0.049351</td>
      <td>-0.182035</td>
      <td>0.069264</td>
      <td>-0.116883</td>
      <td>-0.315368</td>
    </tr>
    <tr>
      <th>543</th>
      <td>A</td>
      <td>H</td>
      <td>34</td>
      <td>2021-05-02</td>
      <td>Newcastle</td>
      <td>Arsenal</td>
      <td>0 - 2</td>
      <td>A</td>
      <td>1.0</td>
      <td>0.080678</td>
      <td>...</td>
      <td>0.187930</td>
      <td>0.160839</td>
      <td>0.779232</td>
      <td>0.007684</td>
      <td>0.134334</td>
      <td>0.021523</td>
      <td>0.024630</td>
      <td>-0.057849</td>
      <td>0.214161</td>
      <td>-0.010845</td>
    </tr>
    <tr>
      <th>285</th>
      <td>H</td>
      <td>A</td>
      <td>8</td>
      <td>2020-11-08</td>
      <td>Leicester</td>
      <td>Wolves</td>
      <td>1 - 0</td>
      <td>H</td>
      <td>1.0</td>
      <td>0.188312</td>
      <td>...</td>
      <td>0.648701</td>
      <td>0.233766</td>
      <td>1.131169</td>
      <td>0.017316</td>
      <td>-0.412771</td>
      <td>0.007576</td>
      <td>0.020779</td>
      <td>-0.372727</td>
      <td>0.025974</td>
      <td>-0.546753</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 30 columns</p>
</div>


Overall, the results produced from a relatively small training set and with minimal model optimisation are very impressive. Football is notoriously hard to predict and pundits are often caught out in their predictions. A 68% accuracy in a season and a half of premier league football is hard to match from even the most experienced football fans.

We can see that the novel approach of predicting results with player data has been successful, and with a larger volume of training data, a more robust model and incorporating team data (such as home advantage, historic form, league position), the predicition accuracy could be improved further.
