
def init(data, id):
    prs={"id" : id,
    "gender" : [],
    "bag" : [],
    "hat" : [],
    "trajectory" : []
    }
    while len(data["people"]) < id:
        data["people"].append(None)
    data["people"][id-1] = prs
    return data