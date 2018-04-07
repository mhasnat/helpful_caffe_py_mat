model = Model(inputs=_input, outputs=_loss)
model.summary()

# Load weights
model.load_weights(model_wt_path)

# Set trainable to freeze layers
for layer in model.layers[:22]:
    layer.trainable = False
