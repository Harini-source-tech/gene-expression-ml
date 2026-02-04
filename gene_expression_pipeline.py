import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from scipy.stats import ttest_ind
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

X_df = pd.read_csv(
    r"C:\Users\harim\OneDrive\Desktop\autoimmune_gene_expression_project\data\data_set_ALL_AML_train.csv"
)

y_df = pd.read_csv(
    r"C:\Users\harim\OneDrive\Desktop\autoimmune_gene_expression_project\data\actual.csv"
)

X_numeric = X_df.select_dtypes(include=[np.number])
X_numeric = X_numeric.T

genes = X_numeric.columns

labels = y_df.iloc[:, 1].astype(str).values
labels = labels[: X_numeric.shape[0]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Label": labels
})

plt.figure(figsize=(8, 6))

for label in np.unique(labels):
    subset = pca_df[pca_df["Label"] == label]
    plt.scatter(subset["PC1"], subset["PC2"], label=label, alpha=0.7)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Gene Expression Dataset")
plt.legend()
plt.tight_layout()
plt.savefig("figures/pca_plot.png")
plt.close()

expr = X_numeric.copy()
labels_series = pd.Series(labels, index=expr.index)

group1 = expr[labels_series == "ALL"]
group2 = expr[labels_series == "AML"]

pvals = []
logfc = []

eps = 1e-6

for gene in genes:
    g1 = np.clip(group1[gene].astype(float), 0, None)
    g2 = np.clip(group2[gene].astype(float), 0, None)

    stat, p = ttest_ind(g1, g2, equal_var=False)

    m1 = g1.mean()
    m2 = g2.mean()

    fc = np.log2(m2 + eps) - np.log2(m1 + eps)

    pvals.append(p)
    logfc.append(fc)

diff_df = pd.DataFrame({
    "Gene": genes,
    "log2FC": logfc,
    "p_value": pvals
}).sort_values("p_value")

diff_df.to_csv("data/differential_expression_results.csv", index=False)

diff_df["-log10(p)"] = -np.log10(diff_df["p_value"] + 1e-300)

plt.figure(figsize=(8, 6))
plt.scatter(diff_df["log2FC"], diff_df["-log10(p)"], alpha=0.4)

sig = (diff_df["p_value"] < 0.05) & (abs(diff_df["log2FC"]) > 1)

plt.scatter(
    diff_df.loc[sig, "log2FC"],
    diff_df.loc[sig, "-log10(p)"],
    alpha=0.7
)

plt.axvline(1, linestyle="--")
plt.axvline(-1, linestyle="--")
plt.axhline(-np.log10(0.05), linestyle="--")

plt.xlabel("log2 Fold Change")
plt.ylabel("-log10(p-value)")
plt.title("Volcano Plot")

plt.tight_layout()
plt.savefig("figures/volcano_plot.png")
plt.close()

top_genes = diff_df.head(20)
top_genes.to_csv("data/top20_genes.csv", index=False)

top30 = diff_df.head(30)["Gene"]
heatmap_data = expr[top30]

cg = sns.clustermap(
    heatmap_data,
    row_colors=labels_series.map({"ALL": "blue", "AML": "red"}),
    standard_scale=1,
    cmap="vlag",
    figsize=(12, 10)
)

cg.savefig("figures/top30_heatmap.png")
plt.close()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

importances = pd.Series(model.feature_importances_, index=genes)

importances.sort_values(ascending=False).head(20).to_csv(
    "data/top20_ml_genes.csv"
)