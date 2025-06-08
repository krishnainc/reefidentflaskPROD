from keras.models import load_model
model = load_model('model/reefident_final_model.keras')
print(model.input_shape)
