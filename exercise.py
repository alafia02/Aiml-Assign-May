#Numpy 
#Question1 
import numpy as np
marks = np.array([78, 85, 90, 66, 72, 88, 95, 70, 60, 80])
print("Mean:", np.mean(marks))
print("Median:", np.median(marks))
print("Max:", np.max(marks))
print("Min:", np.min(marks))
print("Std Dev:", np.std(marks))

#Question2 & 3
import numpy as np
arr = np.arange(1, 21)
print(arr.reshape(4, 5))
print(arr.shape)
print(arr.ndim)

print("First 5:", arr[:5])
print("Last 5:", arr[-5:])
print("Alternate:", arr[::2])

#Ques4
import numpy as np  
a = np.array([10, 20, 30])
b = np.array([1, 2, 3])
print("Add:", a + b)
print("Sub:", a - b)
print("Mult:", a * b)
print("Div:", a / b)


#Pandas
#Ques5
import pandas as pd
df = pd.read_csv("pandas.py\\titanic.csv")
print(df)

#Ques6
import pandas as pd
df = pd.read_csv("pandas.py\\titanic.csv", encoding="latin-1")
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())

#Ques7
import pandas as pd
df = pd.read_csv("pandas.py\\titanic.csv", encoding="latin-1")
print("Total passengers:", len(df))
print("Male passengers:", (df['Sex'] == 'male').sum())
print("Female passengers:", (df['Sex'] == 'female').sum())
print("Average age:", df['Age'].mean())
print("Maximum fare:", df['Fare'].max())
print("Minimum fare:", df['Fare'].min())
print(df.describe())

#Q8
import pandas as pd
df = pd.read_csv("pandas.py\\titanic.csv", encoding="latin-1")
print(df[df['Age'] > 30])
print(df[df['Sex'] == 'female'])
print(df[df['Survived'] == 1])
print(df[df['Pclass'] == 1])

#Q 9&10
import pandas as pd
df = pd.read_csv("pandas.py\\titanic.csv", encoding="latin-1")
print(df.isnull())
df['Age'].fillna(df['Age'].mean(), inplace=True)
df.dropna(subset=['Embarked'], inplace=True)
print(df)


#matplotlib     
#Q11
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("pandas.py\\titanic.csv", encoding="latin-1")
counts = df['Pclass'].value_counts().sort_index()
plt.bar(counts.index, counts.values)
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.title("Passenger Count by Class")
plt.show()

#Q12
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("pandas.py\\titanic.csv", encoding="latin-1")
counts = df['Sex'].value_counts()
plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
plt.title("Male vs Female Passengers")
plt.show()

#Q13
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("pandas.py\\titanic.csv", encoding="latin-1")
plt.hist(df['Age'].dropna())
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution")
plt.show()

#Q14
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("pandas.py\\titanic.csv", encoding="latin-1")
plt.scatter(df['Age'], df['Fare'])
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Age vs Fare")
plt.show() 