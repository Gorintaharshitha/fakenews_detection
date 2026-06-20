
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
fake_data=pd.read_csv("News_dataset/Fake.csv")
fake_data["label"]=0
true_data=pd.read_csv("News_dataset/True.csv")
true_data["label"]=1
combine_data=pd.concat([fake_data,true_data])
print(combine_data["label"].value_counts())
print("Lables checked")
fake=combine_data[combine_data.label==0]
real=combine_data[combine_data.label==1]
min_size= min(len(fake),len(real))
fake_sample=resample(fake,replace=False,n_samples=min_size)
real_sample=resample(real,replace=False,n_samples=min_size)
combine_data=pd.concat([fake_sample,real_sample])
combine_data=combine_data.sample(frac=1).reset_index(drop=True)
combine_data["content"]=combine_data["title"]+" "+combine_data["text"]


print("First rows")
print(combine_data.head())
print("shape of data")
print(combine_data.shape)
x=combine_data["content"].astype(str)
y=combine_data["label"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()
x_train_vec=vectorizer.fit_transform(x_train)
x_test_vec=vectorizer.transform(x_test)
print("vector shape:",x_train_vec.shape)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000)
model.fit(x_train_vec,y_train)
y_pred = model.predict(x_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Training completed")
import joblib
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("saved successfully")
