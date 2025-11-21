from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

# Cargar dataset
wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

# Generando reglas del árbol
rules = export_text(tree, feature_names=wine.feature_names)
print(rules)

exactitud = tree.score(X_test, y_test)
print("~ La precisión fue de:", exactitud)
