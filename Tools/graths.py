import matplotlib.pyplot as plt

# Данные обучения
train_loss = [0.3303, 0.0251, 0.0274, 0.0220, 0.0090, 0.0041, 0.0064, 0.0410, 0.1125, 0.0165,
              0.0154, 0.0062, 0.0046, 0.0022, 0.0063, 0.0040, 0.0084, 0.0024, 0.0017, 0.0067]

# Данные валидации
val_loss = [4.4102, 7.8068, 8.0145, 2.8204, 3.0090, 4.2454, 0.8982, 0.0012, 6.1998e-04, 0.0035,
            0.0066, 0.0401, 0.0488, 0.0255, 0.0170, 0.0148, 0.0107, 0.0228, 0.0926, 0.2508]

# Эпохи
epochs = range(1, len(train_loss) + 1)

# График потерь на обучающей и валидационной выборках
plt.plot(epochs, train_loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()