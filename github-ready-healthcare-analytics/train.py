from src.model_utils import train_and_save_model


if __name__ == "__main__":
    metrics = train_and_save_model()
    print("Best model:", metrics["best_model"])
    print("Accuracy:", metrics["test_accuracy"])
    print("Macro F1:", metrics["test_macro_f1"])
