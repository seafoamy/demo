from os.path import exists

def test_model_exists():
    file_exists = exists("model.pkl")
    assert file_exists