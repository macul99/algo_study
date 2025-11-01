import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

tips = sns.load_dataset("tips")
flights = sns.load_dataset("flights")
iris = sns.load_dataset("iris")

sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", style="sex")
plt.show()

sns.lmplot(data=tips, x="total_bill", y="tip", hue="smoker", height=4, aspect=1.2, ci=95)
plt.show()

sns.lineplot(data=flights, x="year", y="passengers", estimator="mean", errorbar=('ci', 95))
plt.show()

sns.barplot(data=tips, x="day", y="total_bill", hue="sex", estimator="mean", errorbar=('ci', 95))
plt.show()

sns.boxplot(data=tips, x="day", y="total_bill", hue="smoker")
plt.show()

sns.countplot(data=tips, x="day", hue="sex")
plt.show()

sns.histplot(data=tips, x="total_bill", bins=20, kde=True, hue="sex", element="step")
plt.show()

sns.kdeplot(data=tips, x="total_bill", hue="sex", fill=True, common_norm=False, alpha=.4)
plt.show()

sns.pairplot(iris, hue="species", diag_kind="kde")
plt.show()

corr = tips.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()

sns.clustermap(corr, cmap="vlag", annot=True)

sns.catplot(data=tips, x="day", y="total_bill", hue="sex", col="time", kind="box")
plt.show()

g = sns.FacetGrid(tips, col="time", row="sex", margin_titles=True)
g.map(sns.scatterplot, "total_bill", "tip")

# stacked bar plot, bar based on grouped key, stacked with 'total_bill' and 'tip'
tips.groupby('sex')[['total_bill', 'tip']].sum().plot(kind='bar', stacked=True)
plt.show()

tsla[['close', 'Adjusted Close']].loc["2016":].plot(figsize=(12, 8), subplots = True)
plt.show()

# plot using plt directly
plt.figure(figsize=(14, 7))
plt.plot(df_tag["timestamp_local"], df_tag["Value"], label="Value", color="blue")
plt.plot(df_tag["timestamp_local"], df_tag["Q2"], label="Q2", color="green")
anomalies = df_tag[df_tag["Anomaly_Label"]]
plt.scatter(anomalies["timestamp_local"], anomalies["Value"], color="black", label="Anomalies", zorder=3)
smoothed = df_tag[df_tag["Smoothed_Alarm"]]
plt.scatter(smoothed["timestamp_local"], smoothed["Value"], color="red", label="Smoothed_Alarm", zorder=4)
plt.xlabel("Timestamp")
plt.ylabel("Value")
plt.title(f"Time Series Plot for Tag: {tag_name}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# plot using ax
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(f, Pxx)
ax.set_xlabel('Frequency')
ax.set_ylabel('Power Spectral Density')
ax.set_title('Periodogram of Daily Solar Power')
ax.grid(True, alpha=0.3)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0], axes[1]

fig, axes = plt.subplots(2, 2, figsize=(10, 5))
# axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

fig = plt.figure()
ax = fig.add_subplot(111)
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
fig = plt.figure()
ax11 = fig.add_subplot(221)
ax12 = fig.add_subplot(222)
ax21 = fig.add_subplot(223)
ax22 = fig.add_subplot(224)

### Double y-axis
x = np.linspace(0, 10, 200)
y_left = np.sin(x) # left y-axis data
y_right = 0.1 * np.exp(x/3) # right y-axis data

fig, ax = plt.subplots()

Left axis
color_left = 'tab:blue'
ax.plot(x, y_left, color=color_left, label='sin(x)')
ax.set_xlabel('x')
ax.set_ylabel('Left: sin(x)', color=color_left)
ax.tick_params(axis='y', labelcolor=color_left)
ax.spines['left'].set_color(color_left)

Right axis
ax2 = ax.twinx() # shares x with ax
color_right = 'tab:red'
ax2.plot(x, y_right, color=color_right, label='0.1*exp(x/3)')
ax2.set_ylabel('Right: exp scale', color=color_right)
ax2.tick_params(axis='y', labelcolor=color_right)
ax2.spines['right'].set_color(color_right)

Combined legend
lines = ax.get_lines() + ax2.get_lines()
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='upper left')

plt.tight_layout()
plt.show()
