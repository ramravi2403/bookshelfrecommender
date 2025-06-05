from ultralytics import YOLO


class YoloModel:
    def __init__(self, model_path:str):
        self.__model = YOLO(model_path)

    def predict(self, path, save:bool=True, show:bool=False):
        return self.__model.predict(path, save=save, show=show)

    def get_model(self):
        return self.__model
