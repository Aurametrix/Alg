From ludwig import LudwigModel

# train a model
model_definition = {…}
model = LudwigModel(model_definition)
train_stats = model.train(training_dataframe)
# or load a model
model = LudwigModel.load(model_path)

# obtain predictions
predictions = model.predict(test_dataframe)

model.close()
