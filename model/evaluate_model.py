import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path, test_dir, labels):
    from tensorflow.keras.models import load_model
    model = load_model(model_path)

    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_dir, target_size=(150,150), batch_size=32,
        class_mode='categorical', shuffle=False
    )

    loss, acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {acc*100:.2f}%")

    y_true = test_gen.classes
    preds = np.argmax(model.predict(test_gen), axis=1)

    cm = confusion_matrix(y_true, preds)

    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.show()

    print(classification_report(y_true, preds, target_names=labels))
