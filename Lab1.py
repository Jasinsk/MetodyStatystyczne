import numpy as np
import matplotlib.pyplot as plot
import pandas

# Wczytanie danych
data = pandas.read_csv("data/owid-covid-data.csv")

# Ogaraniczenie zakresu do pola zainteresowania
data = data.loc[data["location"] == "Poland"]
data = data[["date", "total_cases", "new_cases", "total_deaths", "new_deaths", "total_cases_per_million", "new_cases_per_million", "total_deaths_per_million", "new_deaths_per_million", ]]

# Opisanie danych
metadata = data.describe()
#print(metadata)

# Grupowanie danych w tygodnie
data["date"] = pandas.to_datetime(data["date"], format='%Y%m%d', infer_datetime_format=True)
data_week = data.resample("W-MON", on="date").sum()
#print(data_week)

# Rysowanie wykresów
data_week.plot.line(y="new_cases")
data_week.plot.line(y=["total_cases_per_million", "new_deaths_per_million"])
plot.show(block=True)
