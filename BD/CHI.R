.
#Tweets SelChile
chi <- read.csv("c.csv", sep = ";", stringsAsFactors = FALSE)
chi <- chi[1:764,] #Se eliminan todas las observaciones vacias
chi$Sentiment <- factor(chi$Sentiment) #Conversion a factores de variable relevante
chi$Sentiment
str(chi)
library(tm)

#Limpieza datos
txt = chi$Tweet.Text
# remueve retweets
txtclean = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", txt)
# remueve @otragente
txtclean = gsub("@\\w+", "", txtclean)
# remueve simbolos de puntuación
txtclean = gsub("[[:punct:]]", "", txtclean)
# remove números
txtclean = gsub("[[:digit:]]", "", txtclean)
# remueve links
txtclean = gsub("http\\w+", "", txtclean)


chi_corpus <- VCorpus(VectorSource(txtclean))
#Creacion corpus de texto para analisis
print(chi_corpus)
chi_corpus_clean <- tm_map(chi_corpus, content_transformer(tolower))
#Transformacion de todos los tweets a minusculas, para eliminar posibles duplicados en el analisis

chi_corpus_clean <- tm_map(chi_corpus_clean, removeWords, c(stopwords("spanish"), "que", "un", "una", "por", "la", "el"))
#Remueve stopwords y palabras irrelevantes del corpus

sw <- readLines("C:/Users/NICOLAS GARZON/Downloads/Nueva carpeta (2)/BD/stopwords.es.txt",encoding="UTF-8")
sw = iconv(sw, to="ASCII//TRANSLIT")
#Archivo con nuevas stopwords en espanol

chi_corpus_clean <- tm_map(chi_corpus_clean, removeWords, sw) #Remueve stopwords faltantes
chi_corpus_clean <- tm_map(chi_corpus_clean, stripWhitespace) #Elimina espacios vacios resultantes
as.character(chi_corpus_clean[[3]])

#Tokenizacion
chi_dtm <- DocumentTermMatrix(chi_corpus_clean)
chi_dtm
#Conjuntos de entrenamiento y prueba
chi_dtm_train <- chi_dtm[1:458, ]
chi_dtm_test <- chi_dtm[459:764, ]

#Vectores de sentimiento
chi_train_labels <- chi[1:458, ]$Sentiment
chi_test_labels <- chi[459:764, ]$Sentiment

prop.table(table(chi_train_labels))
#Las proporciones entre las clases de tweets muestran una tendencia hacia la reaccion positiva (0.62), con las reacciones neutrales como
#segunda opcion (0.36) y la negatividad relegada con solo (0.01)

prop.table(table(chi_test_labels))
#Para el conjunto de prueba se observa una distribucion algo diferente con las reacciones positivas en una proporcion de 0.33, las neutrales con 0.64 y las negativas escasas con 0.022

wordcloud(chi_corpus_clean, min.freq=10, random.order=FALSE)
#Nube de palabras de corpus

pos <- subset(chi, Sentiment=="1")
neg <- subset(chi, Sentiment=="-1")
neu <- subset(chi, Sentiment=="0")
#Separacion de mensajes según su sentimiento

wordcloud(pos$Tweet.Text, scale=c(3, 0.5))
#Nube de palabras tweets positivos

wordcloud(neg$Tweet.Text, scale=c(3, 0.5))
#Nube de palabras tweets negativos

wordcloud(neu$Tweet.Text, scale=c(3, 0.5))
#Nube de palabras tweets neutros

#Terminos frecuentes
findFreqTerms(chi_dtm_train, 5)
chi_freq_words <- findFreqTerms(chi_dtm_train, 5)

#Eliminacion terminos irrelevantes o poco frecuentes del modelo
chi_dtm_freq_train <- chi_dtm_train[ , chi_freq_words]
chi_dtm_freq_test <- chi_dtm_test[ , chi_freq_words]

#Funcion para indicar si los tweets contienen o no terminos frecuentes
convert_counts <- function(x){
  x<- ifelse(x>0, "Yes", "No")
}

chi_train <- apply(chi_dtm_freq_train, MARGIN=2, convert_counts)
chi_test <- apply(chi_dtm_freq_test, MARGIN=2, convert_counts)


#Modelo de prediccion
chi_classifier <- naiveBayes(chi_train, chi_train_labels)
chi_test_pred <- predict(chi_classifier, chi_test)
CrossTable(chi_test_pred, chi_test_labels, prop.chisq=FALSE, prop.t=FALSE, dnn=c('Prediccion', 'Real'))

#El desempeño del modelo es bastante regular, en primer lugar sobreestima las reacciones neutrales ya que predice 262 cundo realmente son 196;
#reduce considerablemente las reacciones positivas prediciendo solo 37 de las 103 originales, y finalemnte acierta con los tweets negativos al predecir 
# los 7 que realmente son encontrados.
#Modelo 2
chi_classifier2 <- naiveBayes(chi_train, chi_train_labels, laplace=1)
chi_test_pred2 <- predict(chi_classifier2, chi_test)
CrossTable(chi_test_pred2, chi_test_labels, prop.chisq=FALSE, prop.t=FALSE, dnn=c('Prediccion', 'Real'))

#Agregando un estimador de Laplace, se observa que dada la escasez de tweets negativos, el modelo directamente los elimina
#creando un sesgo hacia la neutralidad y las emociones positivas, lo que dificulta el trabajo con este metodo.

