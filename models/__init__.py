import importlib
from models.base_model import BaseModel

#对model的多继承,选择合适的model进行import
def find_model_class_by_name(model_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    model_filename = "models." + model_name + "_model"#In general: mvs_points_volumetric_model
    modellib = importlib.import_module(model_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print(
            "In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase."
            % (model_filename, target_model_name))
        exit(0)
    return model


def get_option_setter(model_name):
    model_class = find_model_class_by_name(model_name)
    return model_class.modify_commandline_options

# In general :mvs_points_volumetric_model
def create_model(opt):
    model = find_model_class_by_name(opt.model)
    instance = model()
    instance.initialize(opt)
    print("model [{}] was created".format(instance.name()))
    return instance
