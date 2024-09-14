from train_v3_submodel import train_all_submodels
from train_v3_metamodel import train_metamodel

def main_train(label_condition):
    submodels = train_all_submodels(input_dir="../../REFAIT", label_condition=label_condition)
    best_logloss = train_metamodel(input_dir="../../REFAIT", label_condition=label_condition, submodels=submodels)
    return best_logloss

if __name__ == "__main__":
    b1 = main_train(label_condition="Spinal Canal Stenosis")
    b2 = main_train(label_condition="Left Neural Foraminal Narrowing")
    b3 = main_train(label_condition="Right Neural Foraminal Narrowing")
    b4 = main_train(label_condition="Left Subarticular Stenosis")
    b5 = main_train(label_condition="Right Subarticular Stenosis")

    print("B1=", b1)
    print("B2=", b2)
    print("B3=", b3)
    print("B4=", b4)
    print("B5=", b5)
