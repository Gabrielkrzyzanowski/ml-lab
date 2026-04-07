from data.dataset import generate_linear_data
from models.linear_regression import train_model

def main(): 
    x,y = generate_linear_data() 

    model = train_model(x,y) 

    print(f'Slope:{model.coef_}')
    print(f'Intercept:{model.intercept_}') 

if __name__=="__main__": 
    main()