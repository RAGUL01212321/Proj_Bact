import matplotlib.pyplot as plt

# Assuming `history` is the object returned by model.fit()
def plot_learning_curve(history):
    plt.figure(figsize=(8, 5))
    
    # Plot training & validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid()
    
    plt.show()

# Call the function after training your model
plot_learning_curve(history)
