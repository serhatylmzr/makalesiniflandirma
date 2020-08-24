# -*- coding: utf-8 -*-
import jpype as jp
import nltk
import pandas as pd
import numpy as np
import dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
nltk.download('stopwords')
ZEMBEREK_PATH = 'zemberek-full.jar'
jp.startJVM(jp.getDefaultJVMPath(), 'ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH),ignoreUnrecognized=True)
TurkishSentenceExtractor = jp.JClass('zemberek.tokenization.TurkishSentenceExtractor')
TurkishMorphology = jp.JClass('zemberek.morphology.TurkishMorphology')
TurkishSpellChecker = jp.JClass('zemberek.normalization.TurkishSpellChecker')
TurkishTokenizer = jp.JClass('zemberek.tokenization.TurkishTokenizer')
TurkishLexer = jp.JClass('zemberek.tokenization.antlr.TurkishLexer')

extractor = TurkishSentenceExtractor.DEFAULT
morphology = TurkishMorphology.createWithDefaults()
tokenizer = TurkishTokenizer.ALL
spell = TurkishSpellChecker(morphology)
makale_sayisi = 0
inputs = []
outputs = []
tip = 0
#Makale kategorilerini getir
for i in dataset.label_data:
#Kategorideki makaleleri getir
    for j in range(0, 15):
        #Makaledeki yazım yanlışlarını bulup düzelt
        tokens = tokenizer.tokenize(i[j])
        def analyze_token(token) -> bool:
            t = token.getType()
            return (t != TurkishLexer.NewLine and
                    t != TurkishLexer.SpaceTab and
                    t != TurkishLexer.Punctuation and
                    t != TurkishLexer.RomanNumeral and
                    t != TurkishLexer.UnknownWord and
                    t != TurkishLexer.Unknown)
        corrected_document = ''
        for token in tokens:
            text = token.getText()
            if (analyze_token(token) and not spell.check(text)):
                suggestions = spell.suggestForWord(token.getText())
                if suggestions:
                    correction = suggestions.get(0)
                    corrected_document += (correction)
                else:
                    corrected_document += (text)
            else:
                corrected_document += (text)
        #Makaleyi Cümlelerine ayır
        sentences = extractor.fromParagraph(corrected_document)
        #Makalenin cümlelerini kelimelerine ayır ve kelime köklerini diziye at
        word_roots = []
        for sentence in sentences:
            analysis = morphology.analyzeAndDisambiguate(sentence).bestAnalysis()
            for word in analysis:
                word_roots.append(word.getLemmas()[0])

        stop_word_list = nltk.corpus.stopwords.words('turkish')
        #Kelime kökleri dizisinde istenmeyen noktalama işaretlerini ve stop_word leri diziden at
        word_roots = [e for e in word_roots if e not in (',', '.', '"', ";", ":", "?", "!", "$", "#",
                                                         "/", "UNK", "(", ")")]
        word_roots = [token for token in word_roots if token not in stop_word_list]
        word_roots = str(word_roots)
        #print("Makaledeki kelime sayısı:", len(word_roots))
        makale_sayisi += 1
        #Kelime Kökleri dizisi girişler dizisine atılır
        inputs += [word_roots]
        #Makalenin çıktısı çıkışlar dizisine atılır
        if tip == 0:
           outputs += ["label_sport"]
        elif tip == 1:
            outputs += ["label_health"]
        elif tip == 2:
            outputs += ["label_technology"]
        elif tip == 3:
            outputs += ["label_science"]
        elif tip == 4:
            outputs += ["label_art"]
        elif tip == 5:
            outputs += ["label_history"]
        elif tip == 6:
            outputs += ["label_economy"]
        elif tip == 7:
            outputs += ["label_travel"]
    tip += 1
print("MAKALE SAYISI: ", makale_sayisi)

#Çıktılar Sayısallaştırılır
Encoder = LabelEncoder()
outputs = Encoder.fit_transform(outputs)

#Makalelerin %70 i deneme %30 u test verisi olarak ayrılır.
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(inputs, outputs, test_size=0.3, random_state=69)
#Girişler  sayısallaştırılır
tfidf_vector = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2), max_features=5000)
tfidf_vector.fit(inputs)
Train_X_Tfidf = tfidf_vector.transform(Train_X)
Test_X_Tfidf = tfidf_vector.transform(Test_X)

#**************** DIŞARIDAN EKLENEN BİR TEST MAKALESİNİN KELİME KÖKLERİNE AYIRILMASI VE TFIDF İLE TRANSFORM EDİLMESİ**************************
paragraph = """	
GALATASARAY'DA TRANSFER KRİZİ 
Göreve geldiği ilk günden itibaren bütçede yaptığı atılımlarla dikkatleri üzerine çeken Galatasaray Yönetimi’nin en zorlu dönemi bu sezon olacak. Kulübün harcama limitleri diğer kulüplerden daha yüksek olsa da, yıldız oyuncularının maaşları sarı-kırmızılı idarecilerin belini bükecek. Daha da önemlisi, bu sezon pandemi nedeniyle taraftarın tribünlere ne zaman döneceği belirsizliğini koruduğu için tribün gelirleri ve Şampiyonlar Ligi’ne gidilmemesi nedeniyle iki önemli gelir kaynağından da mahrum kalınacak. Takımın en önemli kozu olan kaleci Fernando Muslera performansıyla kazancının karşılığını verirken, diğer isimler için bunları söylemek çok zor. Özellikle Falcao bu futbolcular arasında başı çekiyor. Yıldız forvetin 5 milyon euro net maaşı bulunuyor. Cim-Bom’da 22 maça çıkan tecrübeli futbolcu, 11 gol attı, 1 asist yaptı.Belhanda ve Feghouli de 3 milyon euronun üstünde kazancı olmasına rağmen geçen sezon beklentilerin çok uzağında kaldı. Bu isimlerin yanında kiralıktan dönen Babel ve Diagne’nin de maaşları kulübe büyük bir fatura olarak çıkacak. Galatasaray Yönetimi, geçen sezon 60 milyon euro tutarındaki gider kalemini 40-42 milyon euro aralığına çekmek istese de, satış olmayınca ve gönderilmesi düşünülen isimlerle yollar ayrılamayınca Aslan’ın işi zora girdi. Pandemi nedeniyle futbolcuların bazıları ücretlerinde indirim yapsa da özellikle gözden çıkarılan Belhanda ve Feghouli’nin yüksek maaşlarını düşünerek ayrılmak istememesi ve ‘kendine kulüp bul’ mesajı verilen Babel’in hala bir talibinin çıkmaması sarı-kırmızılıları çok zorlayacak.

"""
tokens = tokenizer.tokenize(paragraph)
def analyze_token (token) -> bool:
    t = token.getType()
    return (t != TurkishLexer.NewLine and
            t != TurkishLexer.SpaceTab and
            t != TurkishLexer.Punctuation and
            t != TurkishLexer.RomanNumeral and
            t != TurkishLexer.UnknownWord and
            t != TurkishLexer.Unknown)
corrected_document = ''

for token in tokens:
    text = token.getText()
    if (analyze_token(token) and not spell.check(text)):
        suggestions = spell.suggestForWord(token.getText())

        if suggestions:
            correction = suggestions.get(0)
            corrected_document += (correction)
        else:
            corrected_document += (text)
    else:
        corrected_document += (text)

sentences = extractor.fromParagraph(corrected_document)
word_roots = []
for sentence in sentences:
    analysis = morphology.analyzeAndDisambiguate(sentence).bestAnalysis()
    for word in analysis:
        word_roots.append(word.getLemmas()[0])

stop_word_list = nltk.corpus.stopwords.words('turkish')
word_roots = [e for e in word_roots if e not in (',', '.', '"', ";", ":", "?", "!", "$", "#", "/", "UNK")]
word_roots = [token for token in word_roots if token not in stop_word_list]
word_roots = str(word_roots)
input = [word_roots]
test = tfidf_vector.transform(input)
#**************** TEST MAKALESİNİN HAZIRLIĞI BİTTİ *********************
def getArticleType(par):
    case = {
        0:  "Sanat Makalesi",
        1:  "Ekonomi Makalesi",
        2:  "Sağlık Makalesi",
        3:  "Tarih Makalesi",
        4:  "Bilim Makalesi",
        5:  "Spor Makalesi",
        6:  "Teknoloji Makalesi",
        7:  "Seyehat Makalesi"
    }
    return case.get(par, "Makale bulunamadı")
#### NAİVE BAYES ALGORİTMASI *************

# Naive Bayes Algoritması
Naive = MultinomialNB()
Naive.fit(Train_X_Tfidf, Train_Y)
predictions_NB = Naive.predict(Test_X_Tfidf)
predictions_NB2 = Naive.predict(test)

print("Naive Bayes Algoritmasına göre test verisinin sonucu :", getArticleType(predictions_NB2[0]))
print("Naive Bayes Algoritması Doğruluk Skoru -> ", accuracy_score(predictions_NB, Test_Y)*100)
print("\n")
print("Naive Bayes Algoritması için karmaşıklık matrisi")
print(confusion_matrix(Test_Y, predictions_NB))
print("\n")
print("Naive Bayes Algoritması için Sınıflandırma Raporu")
print(classification_report(Test_Y, predictions_NB))

#SUPPORT VECTOR MACHINE ALGORİTMASI *******************HAZIR KÜTÜPHANE İLE*********************
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf, Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
predictions_SVM2 = SVM.predict(test)
print("SVM Algoritmasına göre test verisinin sonucu :", getArticleType(predictions_SVM2[0]))
print("Support Vector Machine Algoritması Doğruluk Skoru -> ", accuracy_score(predictions_SVM, Test_Y)*100)
print("\n")
print("Support Vector Machine Algoritması için karmaşıklık matrisi")
print(confusion_matrix(Test_Y, predictions_SVM))
print("\n")
print("Support Vector Machine Algoritması için Sınıflandırma Raporu")
print(classification_report(Test_Y, predictions_SVM))
jp.shutdownJVM()