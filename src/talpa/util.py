from os import path

SourceROI = 15
SourceEditorDialog = 16


def get_resource(filename):
    return path.join(path.dirname(path.realpath(__file__)), "res", filename)
