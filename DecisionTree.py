import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns

# Dataset de dígitos 3x3 (0 a 4)
cero =   [1,1,1, 1,0,1, 1,1,1]
uno =    [0,1,0, 0,1,0, 0,1,0]
dos =    [1,1,0, 0,1,0, 0,1,1]
tres =   [1,1,0, 1,1,0, 1,1,0]
cuatro = [1,0,1, 1,1,1, 0,0,1]

# Salida de las clases 
Y_onehot = np.array([
    [0,0,0,0,1], # 0
    [0,0,0,1,0], # 1
    [0,0,1,0,0], # 2
    [0,1,0,0,0], # 3
    [1,0,0,0,0]  # 4
])

X_base = np.array([cero, uno, dos, tres, cuatro], dtype=float)

# Aumentar con ruido gaussiano
def agregar_ruido(X, Y, repeticiones=20, ruido=0.3, seed=42):
    rng = np.random.default_rng(seed)
    X_out, Y_out = [], []
    for i in range(len(X)):
        X_out.append(X[i])
        Y_out.append(Y[i])
        for _ in range(repeticiones):
            ruido_sample = X[i] + rng.normal(0, ruido, X[i].shape)
            ruido_sample = np.clip(ruido_sample, 0, 1)
            X_out.append(ruido_sample)
            Y_out.append(Y[i])
    return np.array(X_out), np.array(Y_out)

X_total, Y_total = agregar_ruido(X_base, Y_onehot, repeticiones=20, ruido=0.25)
y_labels = np.argmax(Y_total, axis=1)  # convertir a etiquetas 0–4

print("Tamaño dataset:", X_total.shape)

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X_total, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

print("Train:", X_train.shape, "Test:", X_test.shape)

# Entrenar el modelo de árbol de decisión
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Evaluar el modelo
y_pred = clf.predict(X_test)

# Visualizar el árbol de decisión
plt.figure(figsize=(12,6))
plot_tree(clf, filled=True, class_names=["4","3","2","1","0"], feature_names=[f"x{i}" for i in range(9)])
plt.title("Árbol de Decisión (dígitos 3x3)")
plt.show()

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy en Test: {acc*100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, digits=3))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
plt.xlabel("Predicciones")
plt.ylabel("Reales")
plt.title("Matriz de Confusión - Test")
plt.show()