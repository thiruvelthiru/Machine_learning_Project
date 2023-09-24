from pickle import load
# load the model
model = load(open('model.pkl', 'rb'))
# load the scaler
scaler = load(open('scaler.pkl', 'rb'))
#predict user age 20 and salary 900000
age=int(input("Enter your age: "))
salary=int(input("Enter your salary: "))
user_age_salary=[[age,salary]]
scaled_result = scaler.transform(user_age_salary)
res=model.predict(scaled_result)
if res==1:
    print("He can buy the car")
else:
    print("He can't buy the car")