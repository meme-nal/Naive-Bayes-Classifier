def generate_results(model_results:dict)->None:
    with open("./results/results.txt", mode="w") as file:
        file.write("MODEL RESULTS\n")
        
        for res in model_results:
            file.write("a\n")