from keras.models import load_model, Model

class ModelUtils:
    @staticmethod
    def load_model_from_disk(model_path: str):
        return load_model(model_path, compile=False)

    def save_model_to_disk(model, model_trained, model_name: str):
        print('Saving model to disk...')
        model.save('res/%s' % model_name)
        print('Saved')

        print('Saving loss history to disk...')
        json.dump(model_trained.history, open('res/%s.json' % model_name, 'w'))
        print('Saved')
