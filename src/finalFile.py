# support file for the classification

def init(data, id):
    prs={"id" : id,
    "gender" : [],
    "bag" : [],
    "hat" : [],
    "trajectory" : []
    }
    while len(data["people"]) < id:
        data["people"].append(prs)
    return data

def append(data, id, gender, bag, hat):
    person = data["people"][id-1]
    person["gender"].append(gender)
    person["bag"].append(bag)
    person["hat"].append(hat)
    return data

def classify_gender(final_file):
    male_count = 0
    female_count = 0
    for person in final_file["people"]:
        male_count = 0
        female_count = 0
        if person is not None and "gender" in person and person["gender"] is not None:
            # Itera su ogni valore nella lista "gender"
            for gender in person["gender"]:
                if gender == "Male":
                    male_count += 1
                elif gender == "Female":
                    female_count += 1
        dominant_gender = "Male" if male_count > female_count else "Female"
        print(person["id"])
        person["gender"] = dominant_gender  # Sostituisci il valore con "Male" o "Female"
    return final_file


def classify_hat(final_file):
    for person in final_file["people"]:
        yes_count = 0
        no_count = 0
        if "hat" in person:
            # Itera su ogni valore nella lista "hat"
            for hat in person["hat"]:
                if hat == "Yes":
                    yes_count += 1
                elif hat == "No":
                    no_count += 1
        dominant_hat = "Yes" if yes_count > no_count else "No"
        person["hat"] = dominant_hat  # Sostituisci il valore con "Yes" o "No"
    return final_file


def classify_bag(final_file):
    for person in final_file["people"]:
        yes_count = 0
        no_count = 0
        if "bag" in person:
            # Itera su ogni valore nella lista "bag"
            len_bag = int(len(person["bag"])/2)
            for bag in person["bag"][len_bag-5:len_bag+5]:
                if bag == "Yes":
                    yes_count += 1
                elif bag == "No":
                    no_count += 1
        dominant_bag = "Yes" if yes_count > no_count else "No"
        person["bag"] = dominant_bag  # Sostituisci il valore con "Yes" o "No"
    return final_file

def classify(final_file):
    final_file = classify_gender(final_file)
    final_file = classify_hat(final_file)
    final_file = classify_bag(final_file)
    return final_file


# final = {
#     "people" : []
# }

# final = init(final,1)
# final = append(final,1,"male","Yes","No",1)
# final = append(final,1,"male","Yes","Yes",1)
# final = append(final,1,"female","No","Yes",1)
# final = classify(final)

# print(final)


