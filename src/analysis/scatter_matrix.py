import pandas as pd
import seaborn as sns
sns.set_theme(style="ticks")


train_data = pd.read_csv("./data/example_imputed.csv")
plot = sns.pairplot(train_data)# hue="diagnosis")
fig = plot.figure
fig.savefig("./plots/train_scatter.png") 