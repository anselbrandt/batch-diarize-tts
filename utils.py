import pickle


def saveToPkl(filepath, data):
    with open(
        filepath,
        "wb",
    ) as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def loadPkl(filepath):
    with open(filepath, "rb") as handle:
        data = pickle.load(handle)
    return data
