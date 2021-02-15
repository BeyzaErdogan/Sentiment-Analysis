import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import nltk #stop wordlerden kurtulmak için kullanıldı.
nltk.download('stopwords')
stop_word_list = nltk.corpus.stopwords.words('turkish')    

#Verisetinin Hazırlanması
df = pd.read_excel('veriseti.xlsx')
df = df[0:2200].values 

y = []
x = []
for i in df:
    for k in i:
        sonkrktr = k.rfind(",")
        y.append(k[sonkrktr+1:])
        x.append(k[:sonkrktr])
  
X = pd.Series(x)
Y = pd.Series(y)
data = dict(x=X, y=Y)
df = pd.DataFrame(data)

for index,i in enumerate(y):
    if i=="Tarafsız" or i == " ceb": 
        df.drop(index,axis=0, inplace=True)
    
        
df.reset_index(drop=True,inplace=True)
#print(df["y"].value_counts())

x = df.x
y = df.y

#Stopwordlerin Temizlenmesi
def remove_stopword(tokens):
 
 filtered_tokens = [token for token in tokens if token not in stop_word_list]#stop word'lerden temizlenir.  
 return filtered_tokens
 
#Vectorizer Uygulanması
def tfidf_features(sentence):
    Tfidf_Vector = TfidfVectorizer(max_features=2000)
    Tfidf_Matrix = Tfidf_Vector.fit_transform(sentence)
    Tfidf_Matrix = Tfidf_Matrix.toarray()
    features = Tfidf_Vector.get_feature_names()
    return Tfidf_Matrix,features,Tfidf_Vector
    
   
#Önişleme Adımları
X = []
for sentence in x:

    review = re.sub('\W+', ' ', sentence)
    review = re.sub("\d+", "", review)
    review = review.strip()
    review = review.replace('I','ı') #lower() yapıldığı zaman I harfi i olarak çevirildiğinden dolayı replace ile düzeltildi.
    review = review.replace('İ','i') 
    review = review.lower()
    review = review.split()
    tokens = remove_stopword(review)
    corrected_text = " ".join(tokens)
    X.append(corrected_text)

#Vectorizer Uygulanıldı.
Tfidf_Matrix,features,Tfidf_Vector = tfidf_features(X)

#Çıktı değerleri kategorik old. için sayısal değerlere çevrildi.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

#Modele verilmesi için train test olarak bölündü.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(Tfidf_Matrix, y, test_size=0.33,random_state =42)

#Multinomial sınıflandırıcısı ile model oluşturuldu. Accuracy değeri: 0.87
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Multinomial Accuracy Value: "+str(accuracy_score(y_test,y_pred)))


#Xgboost sınıflandırıcısı ile model oluşturuldu.Accuracy değeri: 0.83
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Xgboost Accuracy Value: "+str(accuracy_score(y_test,y_pred)))


#Deneme
review = ["o ürünü beğenmiştim ama artık beğenmiyorum"]

vec = Tfidf_Vector.transform(review)

print(clf.predict(vec))






