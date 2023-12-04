import pandas as pd

if __name__=='__main__':
    df = pd.read_csv("2019_kbo_for_kaggle_v2.csv")
    df.set_index('batter_name', inplace=True)

    print("Problem 1 : ")
    print("H Top 10 Players : ")
    print(df[(2015 <= df.year) & (df.year <= 2018)].groupby('year').apply(lambda x : x.nlargest(10, ['H']))[['H']], end="\n\n")
    print("avg Top 10 Players : ")
    print(df[(2015 <= df.year) & (df.year <= 2018)].groupby('year').apply(lambda x : x.nlargest(10, ['avg']))[['avg']], end="\n\n")
    print("HR Top 10 Players : ")
    print(df[(2015 <= df.year) & (df.year <= 2018)].groupby('year').apply(lambda x : x.nlargest(10, ['HR']))[['HR']], end="\n\n")
    print("OBP Top 10 Players : ")
    print(df[(2015 <= df.year) & (df.year <= 2018)].groupby('year').apply(lambda x : x.nlargest(10, ['OBP']))[['OBP']], end="\n\n\n")
    print("Problem 2 : ")
    print(df[df.year == 2018].groupby('cp').apply(lambda x : x.nlargest(1, 'war'))[['war']], end="\n\n")
    print("Problem 3 : ")
    print(df[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']].corr()['salary'].drop('salary').sort_values(ascending=False).index[0], end="\n\n")