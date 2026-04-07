from data.dataset import generate_linear_data
from models.linear_regression import train_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def main(): 
    x,y = generate_linear_data() 

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = train_model(x,y) 

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)

    print(f'Slope:{model.coef_}')
    print(f'Intercept:{model.intercept_}') 
    print(f'MSE:{mse}') 

if __name__=="__main__": 
    main()